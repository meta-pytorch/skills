---
name: orchestrate-stable-abi-migration
description: Execute a PyTorch stable-ABI migration on a third-party extension repo by dispatching sub-skills in dependency order with build+test verification between each step. Use when the user wants to actually perform the migration (after running assess-abi-migration, or with --skip-assess to discover on the fly).
allowed-tools: Read, Bash, Edit, Write, Glob, Grep
license: BSD-3-Clause
---

# Orchestrate Stable-ABI Migration

Coordinate a full stable-ABI migration of a third-party PyTorch C++/CUDA extension. This skill does not rewrite code itself — it dispatches purpose-built sub-skills in the right order and verifies each step.

## Prerequisites

- A migration plan exists. Run `assess-abi-migration` first to get a `<repo_root>/.abi-migration-plan.json`.
- PyTorch 2.10+ installed and matches the build target version (verify: `python3 -c "import torch; print(torch.__version__)"`).
- The target extension repo builds and the test_cmd tests pass **before** migration begins. If they don't, stop and have the user fix that first — you need a green baseline.

## Inputs

- `plan_path`: absolute path to `.abi-migration-plan.json` (default: `<repo_root>/.abi-migration-plan.json`).
- `build_cmd` and `test_cmd` (overrides plan values if provided).

## End-state goal

The successful exit state — also the motivation users should hear up front — is **one wheel** that:

- Works against any PyTorch ≥ the chosen `TORCH_TARGET_VERSION` (no rebuild per torch minor).
- Optionally works against any CPython ≥ the chosen `Py_LIMITED_API` (no rebuild per Python version).

Without ABI stability, the build matrix is (Python versions) × (torch versions) wheels. With both flags set, that matrix collapses to a single binary.

## Dispatch order

Run **only** the sub-skills listed in the plan's `dispatch_order`, in this canonical sequence (skip any that don't apply to this repo):

1. **`pybind-to-torch-library`** — if any file uses `PYBIND11_MODULE` or pybind `m.def`. Must come first because later steps assume registrations go through `TORCH_LIBRARY`, not pybind. Optionally, this skill can enable `Py_LIMITED_API` as its closing step (if removing pybind is what unlocks the Limited API), giving the user a partial-credit single-wheel win even if they stop here.
2. **`migrate-meta-fns-to-python`** — if any C++ Meta dispatch impls exist. Move them out of C++ into Python before splitting/migrating the remaining C++.
3. **`migrate-autograd-fns-to-python`** — if any C++ Autograd dispatch impls exist (`TORCH_LIBRARY_IMPL(..., Autograd, m)`, `torch::autograd::Function` subclasses), move them into Python. Independent of step 2; safe to run in either order, but both should complete before `migrate-to-stable-apis`.
4. **`scaffold-stable-target`** — runs only for **incremental** migrations (assess Step 7). Adds a second `ext_modules` entry (stable, empty `sources=`, with `TORCH_TARGET_VERSION` always set, and `Py_LIMITED_API` if the user opts in) alongside the legacy one. Both entries share the same `csrc/` directory and register into the same `torch.ops.<libname>` namespace; no files move on disk. The legacy target keeps building until decommissioned at the end. For **one-shot** migrations, this skill is skipped — `migrate-to-stable-apis` converts the existing extension in place.
5. **`migrate-to-stable-apis`** — the per-file work, in one of two shapes depending on whether step 4 ran:
   - **Incremental** (step 4 ran): rewrite each file in place (includes, APIs, registration macros) and flip its path from the legacy entry's `sources=` to the stable entry's `sources=`. One file at a time, build + test between each. Stop when the legacy entry's `sources=` is empty, then drop the legacy entry.
   - **One-shot** (step 4 skipped): add `TORCH_TARGET_VERSION` (and optionally `Py_LIMITED_API`) to the existing `ext_modules` entry, rewrite every in-scope file's APIs + registration macros, build, test. Done in one pass.

The rationale for this ordering: each step shrinks or relocates the surface the next step has to deal with. Doing API rewrites before splitting would force you to revisit files twice.

## Per-step protocol

For each sub-skill in `dispatch_order`:

1. **Pre-step**: confirm the working tree is clean (or warn the user — easier to bisect failures).
2. **Invoke** the sub-skill with:
   - `plan_path`
   - the list of files from the plan that have this step's concern
3. **Build** using `build_cmd` (or the plan's recorded value).
   - If build fails: stop, report the error, do not advance. The sub-skill should be re-runnable to fix issues.
4. **Test** using `test_cmd`.
   - If tests fail: stop, report which tests, leave changes for inspection. Do not auto-revert.
5. **Commit checkpoint** (optional, ask user): a git commit per successful step makes bisecting trivial later. Recommend the user enable this on first run.
6. **Update plan**: mark this sub-skill as `completed` in the plan JSON. This makes the orchestrator resumable.

## Resume behavior

If `.abi-migration-plan.json` has any `dispatch_order` entries already marked `completed`, skip them. Start from the first incomplete step.

## Outputs

- A migrated repo whose `dispatch_order` is fully completed in the plan.
- A short final summary: number of files touched per step, any remaining coverage gaps from the assessment that were left as TODOs.

## Stopping criteria

Stop and ask the user before continuing if:
- A sub-skill reports it cannot complete (e.g., coverage gap with no workaround).
- Build or test failures persist after one re-run.
- The plan's `blockers` array is non-empty.

## Handoff

When all steps complete and the project's existing tests pass, summarize for the user:

- `TORCH_TARGET_VERSION` pinned at `<value>` — binary requires PyTorch ≥ that version.
- If applicable, `Py_LIMITED_API` pinned at `<value>` — binary works on CPython ≥ that version.
- Single-wheel goal: achieved / not achieved (if any sub-skill was skipped or partial).
- Coverage gaps left in UNSTABLE translation units, if any — these need follow-up when the stable ABI expands.

Recommend an empirical ABI verification: install a different PyTorch version and re-run the project's tests **without rebuilding** the extension. A pass there proves the migration delivered on its promise.
