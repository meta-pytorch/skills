---
name: assess-abi-migration
description: Audit a PyTorch C++/CUDA extension repo for ABI-stable migration readiness. Use when the user wants to know whether their library can be made ABI stable with libtorch, what work is required, and where coverage gaps in the stable ABI exist. Produces an ordered migration plan as both a JSON file and a markdown summary.
allowed-tools: Read, Bash, Glob, Grep
license: BSD-3-Clause
---

# Assess ABI Migration

Inspect a third-party PyTorch extension repo (e.g., flash-attention, vllm, a custom CUDA op library) and determine:

1. Whether the project is migrateable to the stable ABI for the PyTorch version installed.
2. What concrete steps are required, in dependency order.
3. Where the stable ABI has **missing coverage** for this project — APIs the project uses that have no `torch::stable` or `torch::headeronly` equivalent yet.

This skill does not modify code. It produces a machine-readable plan that `orchestrate-stable-abi-migration` consumes.

## Prerequisites

- The user's extension repo (the migration target) is checked out locally.
- PyTorch 2.10+ is installed in the active environment (this is the `build_torch_version`):
  ```bash
  python3 -c "import torch; print(torch.__version__)"
  ```
  If below 2.10, stop and tell the user to upgrade — the stable ABI surface is much more complete in 2.10+.

## Inputs

- `repo_root`: absolute path to the extension repo root.

- `build_torch_version`: the PyTorch version the extension will be **compiled against**. Determines which headers are present during the build and which symbols the assess step can grep for. **Should match the currently installed torch** (`python3 -c 'import torch; print(torch.__version__)'`). If not provided, default to the installed version and confirm with the user, e.g.:

  > "I'll assess against the installed torch `<x.y.z>` as the build-time version. Use a different version? (If so, install it in the active env first — the assess step greps real headers.)"

  Constraint: must equal the installed torch version. If they differ, stop and ask the user to either install the requested version or accept the installed one.

- `target_torch_version`: the **minimum runtime** PyTorch version the resulting wheel should support — gets encoded as `TORCH_TARGET_VERSION` in the build. Lower = wider compatibility (one wheel covers more torch releases); higher = more stable APIs available. **If not provided, ask** before continuing:

  > "What's the lowest PyTorch version you want this wheel to support at runtime? Common answers: `2.10` (widest), `2.11` (standard), or `<build_torch_version>` (match the build, narrowest support window). This becomes `TORCH_TARGET_VERSION` and determines what counts as a coverage gap."

  Constraints:
  - ≥ 2.9 (older versions don't have a usable stable surface).
  - ≤ `build_torch_version` (can't promise runtime support for a torch newer than what we compiled against).

### Why two versions

- The **build** version determines what's available in headers right now, on this machine, for the migration to use.
- The **target** version determines what's available in headers on **every other user's machine** when they install the resulting wheel. It's the lowest common denominator the binary must work with at runtime.

A coverage gap is "API exists in build, doesn't exist at target" — those are APIs you literally cannot use even though they're sitting in the headers, because the wheel must also run on older torch installations. Step 4 checks for this by greping the build headers and then verifying each found symbol's `#if TORCH_FEATURE_VERSION >= TORCH_VERSION_X_Y_0` guard against the target.

## Discovery steps

### 1. Identify the build system

Look for, in priority order:
- `setup.py` with `CUDAExtension` / `CppExtension` imported from `torch.utils`.
- `CMakeLists.txt` referencing `Torch` / `find_package(Torch)`.
- `pyproject.toml` with a build backend.

Record which is present.

### 2. Find C++/CUDA source files in scope

Ignore test files and any files (e.g., in an `experimental` directory) that are not part of the shipped library. Look for files matching `*.cpp`, `*.cc`, `*.cu`, `*.cuh`, usually in a `csrc` directory, that include any of the following headers:
- `<torch/extension.h>`
- `<ATen/...>`
- `<c10/...>`
- `<torch/library.h>`
- `<pybind11/...>`

#### 2a. Vendored / FetchContent / submodule sources are in scope if compiled into the shipped wheel

A "vendored" source is any file from an external project (git submodule, CMake `FetchContent`, vendored copy under `third_party/`, `csrc/external/`, etc.) that the build system **compiles into the shipped shared library**. From the resulting `.so`'s perspective, vendored code is indistinguishable from first-party code — both contribute symbols and both must be ABI-stable.

To find them, read the build files (`CMakeLists.txt`, `setup.py`, sub-CMakeLists includes) and look for:
- CMake variables resolved to `${repo-foo_SOURCE_DIR}/.../*.cu` listed in `SOURCES` / `add_library(...)` / `Python_add_library(...)`.
- `file(GLOB ... ${repo-foo_SOURCE_DIR}/...)` patterns that pick up many vendored files at once.
- `add_subdirectory(${repo-foo_SOURCE_DIR} ...)` that builds the vendored project as a static lib linked into the wheel.
- `setup.py` extensions whose `sources=[...]` list includes paths outside `csrc/`.

For each vendored source compiled into the wheel:
- If the files are checked out locally (e.g. `build/_deps/repo-foo-src/...` exists, or under `third_party/`), grep them the same way as first-party sources for the concerns in Step 3 and the gaps in Step 4. Add them to the `files` list with a `vendored: true` marker and the upstream repo URL + pinned commit/tag.
- **If they are not checked out locally** (`build/_deps/` empty, no `third_party/` copy) and you cannot fetch them (no network access to GitHub, no permission to run `cmake configure` to populate `_deps`), surface this as a **blocker** of type `vendored-source-unfetched`. Record the file list (with `${repo-foo}` placeholders), the upstream repo+tag from `FetchContent_Declare`, and the reason. Do NOT silently drop them — the wheel cannot ship as ABI-stable until they are migrated too.

Remediation options to suggest with each `vendored-source-unfetched` blocker:
1. Re-run the assess skill on a host with network access after `cmake -S . -B build` populates `build/_deps/`.
2. Upstream the migration via PRs to the vendored project (and bump the `GIT_TAG` in `FetchContent_Declare` afterwards).
3. Fork-and-patch locally under `csrc/third_party/` with the migration applied, replacing the `FetchContent_Declare` URL.

**Things that are NOT in scope even if vendored:**
- Header-only libraries (e.g. cutlass, fmt) — no compiled `.cu`/`.cpp` from them; they only affect compile time, not the resulting symbol table.
- Static libs that don't call libtorch (e.g. a vendored NCCL/mscclpp built via `add_subdirectory`). The static lib gets linked into the wheel but its own source doesn't touch the unstable libtorch ABI. **Verify** by checking the vendored project's source for `torch::`/`at::`/`c10::` — if absent, it's fine; if present, it's in scope after all.
- Python-only artifacts installed via CMake `install(DIRECTORY ...)` (e.g. a vendored Triton kernel directory). No C++ compilation, no migration.

#### 2b. Cross-backend / multi-wheel projects

Many projects ship multiple wheels from the same source tree — typically a CUDA wheel plus alternate-accelerator wheels (CPU, ROCm, MUSA/Moore Threads, XPU, Metal). Look for sibling build files like `pyproject_cpu.toml`, `pyproject_rocm.toml`, `pyproject_musa.toml`, `setup_metal.py`, or `csrc/cpu/CMakeLists.txt`. Each one usually produces a separately-named wheel (e.g. `sglang-kernel-cpu`).

For each alternate wheel:
- Note its `[build-system].requires` torch pin (it may differ from the main CUDA wheel's pin — e.g. CPU pinned to `torch==2.9.0` while CUDA uses `>=2.8.0`).
- Identify the **headers and source files it shares** with the in-scope wheel. Look for:
  - `target_include_directories(... ../../include)` or equivalent — the alt-wheel sees the same shared header files.
  - Headers like `sgl_kernel_ops.h` that declare function signatures with `at::Tensor`/`TORCH_CHECK`/`c10::optional` and are `#include`d by both the in-scope binder (`common_extension.cc`) and the alt-arch binders (`common_extension_musa.cc`, `common_extension_rocm.cc`).
  - Shared utility headers (`utils.h`, `scalar_type.hpp`).
- Surface the sharing as a **`shared-header-cross-backend` blocker** for the in-scope wheel if a naive migration would force the alt-wheels to move too. Migrating a shared header's `at::Tensor` to `torch::stable::Tensor` cascades into every `#include`r.

Remediation options to suggest with each `shared-header-cross-backend` blocker:
- **Plan A — migrate all wheels together.** Includes alt-arch in the same PR. Safest semantics, biggest scope.
- **Plan B (usually preferred) — split the shared header per-backend.** Create `..._unstable.h` (used by alt-arch backends staying on libtorch) and `..._stable.h` (used by the migrated wheel). Lets the in-scope wheel ship ABI-stable while alt-arches catch up.
- **Plan C — conditionalize the shared header on a macro** like `#ifdef PROJECT_USE_STABLE_ABI`. Lightest touch but increases header complexity.

Record the cross-backend findings in a top-level `cross_backend_analysis` section of the plan JSON: for each alt-wheel, list its torch pin, the headers it shares with the in-scope wheel, whether it's in scope for *this* run, and a one-line note about its migration prospects (e.g. "MUSA depends on torch_musa fork's stable-ABI support — gates this wheel"). This lets the user decide whether to migrate the wheels in one shot or as a sequence.`

### 3. Detect migration concerns per file

For each source file, record which of the following apply (these drive sub-skill dispatch):

| Concern | Sub-skill | Detection signal |
|---|---|---|
| Pybind bindings | `pybind-to-torch-library` | `PYBIND11_MODULE`, `m.def("name", ...)` outside `TORCH_LIBRARY`, `<pybind11/...>` includes |
| C++ meta functions | `migrate-meta-fns-to-python` | `TORCH_LIBRARY_IMPL(..., Meta, m)`, registrations with dispatch key `Meta` |
| C++ autograd impls | `migrate-autograd-fns-to-python` | `TORCH_LIBRARY_IMPL(..., Autograd, m)` / `..., AutogradCUDA, m)`, C++ `torch::autograd::Function` subclasses registered for the op |
| Unstable API usage | `migrate-to-stable-apis` | Any `at::Tensor`, `TORCH_CHECK`, `c10::cuda::CUDAGuard`, `at::empty`, `TORCH_LIBRARY_IMPL`, etc. |

The `scaffold-stable-target` subskill is dispatched only for **incremental** migrations. See Step 7 for the cadence decision; one-shot migrations skip it and `migrate-to-stable-apis` converts the existing extension in place.

### 4. Detect stable-ABI coverage gaps

Coverage gaps are APIs the project uses that have **no equivalent in the installed PyTorch's stable surface**. The stable surface changes between PyTorch releases, so this check must be against the actually-installed `torch`, not a hardcoded list.

**Resolve the stable headers directory:**

```bash
TORCH_STABLE_DIR=$(python3 -c "
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), 'include', 'torch', 'csrc', 'stable'))
")
ls "$TORCH_STABLE_DIR"
# Expected: ops.h, tensor.h, library.h, device.h, accelerator.h, c/, ...
```

Also locate `torch/headeronly/`:

```bash
TORCH_HO_DIR=$(python3 -c "
import os, torch
print(os.path.join(os.path.dirname(torch.__file__), 'include', 'torch', 'headeronly'))
")
```

**For each candidate gap**, grep the stable headers for the expected replacement symbol. Examples:

| Unstable API in user code | Expected stable symbol | Verification |
|---|---|---|
| `at::empty_strided(...)` | `new_empty_strided` or `empty_strided` | `grep -E 'new_empty_strided\|^.*empty_strided\b' "$TORCH_STABLE_DIR"/*.h "$TORCH_STABLE_DIR"/c/*.h` |
| `tensor.layout() == at::Layout::Strided` | `Layout` enum or a `layout()` method on `torch::stable::Tensor` | `grep -E 'enum.*Layout\|::Layout\b\|::layout\(' "$TORCH_HO_DIR"/**/*.h "$TORCH_STABLE_DIR"/*.h` |
| `at::native::<op>` | `<op>` in `ops.h` | `grep -E '\b<op>\b' "$TORCH_STABLE_DIR"/ops.h` |
| `at::parallel_for(...)` | `torch::stable::parallel_for` in `ops.h` (backed by `torch_parallel_for` C shim) | `grep -n 'parallel_for' "$TORCH_STABLE_DIR"/ops.h "$TORCH_STABLE_DIR"/c/shim.h` |
| `at::TensorIterator` | n/a — no current stable equivalent | confirm absence: `grep -i tensoriterator "$TORCH_STABLE_DIR"/*.h "$TORCH_HO_DIR"/**/*.h` |
| `torch::autograd::Function` | n/a in C++ — move to Python | not a header check; route to `migrate-autograd-fns-to-python` instead |

> **Important:** This table is illustrative and reflects the stable surface at the time of writing. The stable ABI is actively growing — never trust this table over the actual `grep` against the installed headers. Symbols that were "absent" in one PyTorch release often appear in the next.

A symbol is **available at build** if it's declared in any header under `$TORCH_STABLE_DIR` or `$TORCH_HO_DIR`. The C shims under `$TORCH_STABLE_DIR/c/` and `<torch_include>/torch/csrc/inductor/aoti_torch/c/` also count.

### Resolving the minimum target version for each symbol

Two cases, each handled differently:

**Case 1: C shim functions** (anything declared in `torch/csrc/stable/c/shim.h` or `torch/csrc/inductor/aoti_torch/c/*.h`). Use the authoritative version map:

```
torch/csrc/stable/c/shim_function_versions.txt
```

Format: one `function_name: TORCH_VERSION_X_Y_Z` line per shim. Per the file's own header: **"If a function is not in this file, it was available before 2.10.0."** Since our minimum supported target is 2.9, that means unlisted = available at every target we care about.

Lookup logic for a C shim symbol `<sym>`:
- Listed as `<sym>: TORCH_VERSION_X_Y_Z` → minimum version is `X.Y.Z`.
- Not listed → available at all supported targets (treat as no version constraint).

The file does **not** ship in the installed torch's `include/` tree, so to access the file:
- If `<pytorch_src>/torch/csrc/stable/c/shim_function_versions.txt` is available locally, read it directly.
- Otherwise fetch from GitHub pinned to the build torch's release tag:
  `https://raw.githubusercontent.com/pytorch/pytorch/v<build_torch_version>/torch/csrc/stable/c/shim_function_versions.txt`

**Case 2: C++ wrappers** (anything in `torch/csrc/stable/*.h` outside `c/`, or `torch/headeronly/`). No authoritative version map; use grep around the symbol declaration as an estimate:

```bash
grep -B 5 -A 2 "<symbol>" "$TORCH_STABLE_DIR"/*.h
# Look for the enclosing `#if TORCH_FEATURE_VERSION >= TORCH_VERSION_X_Y_Z`
```

Three outcomes:

- **Clear guard found** with `X.Y.Z > target_torch_version` → coverage gap; `min_target_to_close: "X.Y.Z"`.
- **Clear guard found** with `X.Y.Z ≤ target_torch_version` → available at target; not a gap.
- **No guard found, or guards are nested/combined and grep is ambiguous** → cannot determine the minimum version with certainty. Default to:
  - Available at `build_torch_version` (it's in the build headers, so the build works).
  - "Maybe supported" at older targets. Record `min_target_to_close: "<build_torch_version>"` with `reason: "version guard inconclusive; available at build, may not be at older runtimes"`. Surface in chat summary so the user can spot-check or accept the conservative answer.

The grep approach is rough — it can miss nested `#if`s, `||`/`&&` combinations, and macro indirection. For C shims that's not a problem (the txt file is authoritative). For C++ wrappers, accept some false-positive gaps as the price of a fast assess; the user can override via the `reason` field or run `migrate-to-stable-apis` and let the build confirm.

#### For each gap, evaluate whether bumping the target would close it

Three categories:

1. **Gap closeable by raising target.** Symbol exists in build headers under a `TORCH_FEATURE_VERSION` guard with `X.Y.Z > target_torch_version`. Bumping target to `X.Y.Z` closes the gap. Record `min_target_to_close: "X.Y.Z"`.

2. **Gap not closeable with current offering.** Symbol is absent from build headers for the installed torch. Options to try a documented workaround, rerunning the skill on a newer torch version, or filing an issue for the API to be added. Record `min_target_to_close: null` with a `reason` field, e.g., `"absent from torch::stable through build version <build_torch_version>"`.

3. **Gap structural, no stable equivalent expected.** E.g., `at::TensorIterator`, `torch::autograd::Function` — these are libtorch-internal designs without a planned stable surface (autograd specifically moves to Python via `migrate-autograd-fns-to-python`). Record `min_target_to_close: "n/a"` with `reason: "no planned stable equivalent"`. Be conservative classifying things here — if you're not sure whether an API is structurally absent vs. just not added yet, prefer category 2 (`null` with a `reason`).

Aggregate the closeable gaps for the chat summary. If raising the target by one minor version would close N of M gaps, surface that as actionable guidance: the user can decide whether the narrower runtime support window is worth shedding those workarounds.

**Heuristic for naming.** Most legacy ATen ops have a same-named stable counterpart in `torch::stable::`. So for `at::<op>(...)`, the first thing to check is `grep -E '\b<op>\b' "$TORCH_STABLE_DIR/ops.h"`. If that fails, also try variants prefixed with `new_` (PyTorch's convention for factory ops that take a reference tensor — `new_empty`, `new_zeros`).

**Record the target PyTorch version with each gap.** Use `target_torch_version` from the skill's inputs (not the installed version, if they differ). Future readers need to know "this gap was true at the target the assessment was scoped to" — the same API may be added in a later release that the user might be willing to raise the target to.

For each gap found, record in the plan: file, line, old API, the expected stable symbol that was checked for, the torch version checked against, suggested workaround if one is documented (see [api-mapping.md § Strided Tensors](../migrate-to-stable-apis/references/api-mapping.md) and [common-issues.md](../migrate-to-stable-apis/references/common-issues.md)), or `BLOCKED` if no workaround.

The plan's `coverage_gaps` entry should look like:

```json
{
  "path": "src/bar.cpp",
  "line": 142,
  "old_api": "at::empty_strided",
  "checked_symbol": "torch::stable::new_empty_strided",
  "checked_against_target_version": "2.11",
  "min_target_to_close": "2.13",   // or null / "n/a" — see Step 4 categories above
  "reason": null,                  // populated when min_target_to_close is null or "n/a"
  "workaround": "new_empty + transpose",
  "blocked": false
}
```

### 5. Capture build flags

Note whether the project already uses:
- `-DUSE_CUDA` (needed for `aoti_torch_get_current_cuda_stream`).
- `-DTORCH_TARGET_VERSION=...` (pins the target PyTorch version and bans unstable headers).
- `-DPy_LIMITED_API=0x...` (enables single-wheel-across-Python-versions). Also check `py_limited_api=True` in `cpp_extension.*Extension(...)` and `options={"bdist_wheel": {"py_limited_api": "..."}}` in `setup()`.

### 6. Detect existing Python fake/meta registrations

Grep for `@torch.library.register_fake(` and `torch.library.impl(..., "Meta")`. Record any ops that **already** have a Python fake — these don't need `migrate-meta-fns-to-python` even if a corresponding C++ Meta impl exists (the C++ one becomes redundant and gets deleted).

### 7. Decide migration cadence → whether to include `scaffold-stable-target`

Two valid shapes:

- **Incremental** (run `scaffold-stable-target`). A parallel stable `ext_modules` entry sits alongside the legacy one and grows file-by-file. The legacy target keeps building and serving un-migrated ops throughout. Best when the migration will land over multiple commits/PRs, or when CI gating requires a green tree between every file change.
- **One-shot** (skip `scaffold-stable-target`). `migrate-to-stable-apis` converts the single existing extension in place: adds `TORCH_TARGET_VERSION` to its flags, rewrites every in-scope file's APIs + registration macros, and the project ships the stable wheel directly. Best when the migration is small enough to validate as a single change — typically only a handful of translation units.

The threshold isn't strict. Use the count from Step 2 as a starting heuristic — roughly < ~5 TUs leans one-shot, more leans incremental — but the real question is "will the user land this as one change or many?"

**Ask the user before recording the decision:**

> "Found `<N>` translation units to migrate. Do you want to land this as: (a) one-shot — convert the existing extension in place, single PR; or (b) incremental — set up a parallel stable target and migrate file-by-file across multiple PRs, with the legacy target staying green throughout? Recommendation for `<N>` files: `<one-shot | incremental>`."

Record the choice. If **incremental**, include `scaffold-stable-target` in `dispatch_order` (before `migrate-to-stable-apis`). If **one-shot**, omit it. Save the user's choice and reasoning so re-runs don't re-prompt.

### 8. Detect test setup

Look for `pytest.ini`, `tests/`, `test_*.py`, `pyproject.toml`'s `[tool.pytest.ini_options]`, or a `Makefile` test target. Record the `test_cmd`. Sometimes testing the whole library is too expansive, in that case identify a few relevant smoke tests. If no tests exist at all, flag that as a `blocker` — a migration without a baseline test signal is too risky to auto-execute.

### Background: versioned shims (for explaining coverage gaps)

When this skill flags a "coverage gap," the underlying mechanic is usually shim versioning. The C shims in `torch/csrc/inductor/aoti_torch/c/shim.h` (and the wrappers in `torch/csrc/stable/`) are guarded:

```cpp
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_11_0
inline torch::stable::Tensor from_blob(...) { ... }
#endif
```

So an API can simultaneously "exist in PyTorch" and "not be available at this `TORCH_TARGET_VERSION`." When reporting a gap, mention which torch version first exposed the missing API if you can determine it — that lets the user weigh raising `TORCH_TARGET_VERSION` against keeping wider compatibility.

The C-shim APIs (under `torch/csrc/inductor/aoti_torch/c/` and `torch/csrc/stable/c/`) carry a **minimum 2-year backward-compatibility window** from the release that introduced them. The higher-level wrappers in `torch/csrc/stable/` and `torch/headeronly/` follow PyTorch's standard FC/BC policy. Cite this when users ask "how long will this binary keep working?"

## Outputs

### File: `<repo_root>/.abi-migration-plan.json`

Source of truth for the migration. Schema:

```json
{
  "version": 1,
  "generated_at": "<ISO 8601>",
  "repo_root": "<absolute path>",
  "build_torch_version": "<x.y.z>",      // version compiled against; matches installed torch
  "target_torch_version": "<x.y.z>",       // TORCH_TARGET_VERSION; ≤ build_torch_version
  "build_system": "setup.py | cmake | pyproject",
  "files": [
    {
      "path": "src/foo.cu",
      "concerns": ["unstable-api", "pybind"],
      "old_apis": ["at::Tensor", "TORCH_CHECK", "PYBIND11_MODULE"]
    }
  ],
  "build_flags": {
    "use_cuda": false,
    "torch_target_version": null,
    "py_limited_api": null
  },
  "existing_python_fakes": ["mylib::op_with_fake"],
  "coverage_gaps": [
    {
      "path": "src/bar.cpp",
      "line": 142,
      "old_api": "at::empty_strided",
      "checked_symbol": "torch::stable::new_empty_strided",
      "checked_against_target_version": "2.11",
      "min_target_to_close": "2.13",
      "reason": null,
      "workaround": "new_empty + transpose",
      "blocked": false
    }
  ],
  "gap_closure_summary": {
    "total": 5,
    "closeable_by_bumping_target": {
      "2.12": 1,
      "2.13": 3
    },
    "structural": 1
  },
  "dispatch_order": [
    "pybind-to-torch-library",
    "migrate-meta-fns-to-python",
    "migrate-autograd-fns-to-python",
    "scaffold-stable-target",   // omitted for one-shot migrations (Step 7)
    "migrate-to-stable-apis"
  ],
  "migration_cadence": "incremental",   // or "one-shot" — recorded from Step 7
  "migration_cadence_reason": "<user's stated reason or assess's recommendation>",
  "cross_backend_analysis": {
    "wheels": {
      "<wheel name> (e.g., sglang-kernel-cpu)": {
        "torch_pin": "<from pyproject_*.toml>",
        "shares_with_in_scope_wheel": ["include/foo.h", "..."],
        "in_scope_for_this_assessment": false,
        "note": "<one-line summary of migration prospects / gating factors>"
      }
    },
    "recommendation": "<which plan from the shared-header blocker; rationale>"
  },
  "blockers": [
    {
      "type": "vendored-source-unfetched",
      "scope": "<which built target / wheel>",
      "files": ["${repo-foo}/path/to/file.cu", "..."],
      "reason": "<what we know without grepping; why it's a blocker>",
      "remediation_options": ["fetch + re-run", "upstream PR", "fork-and-patch"]
    },
    {
      "type": "shared-header-cross-backend",
      "scope": "include/foo.h",
      "files": ["include/foo.h", "..."],
      "reason": "<which alt-arch binders #include this header; what cascades>",
      "remediation_options": ["Plan A — migrate all together", "Plan B — split per-backend", "Plan C — conditional macro"]
    }
  ]
}
```

Include only the sub-skills in `dispatch_order` whose concerns appear in the file list. Drop ones that don't apply.

### Chat summary

Print a short markdown summary to the user:

- Build torch version: `<build_torch_version>` (must match installed).
- Target torch version (`TORCH_TARGET_VERSION`): `<target_torch_version>`. Resulting wheel will require this version or newer at runtime.
- Migrateable: yes / yes-with-workarounds / no (with reason).
- Files in scope: count (broken down: first-party / vendored-fetched / vendored-unfetched).
- Cross-backend wheel summary (if alt-arch wheels exist): one bullet per alt-wheel with torch pin + shared headers + the chosen Plan A/B/C.
- Sub-skills that are recommended (in order).
- Coverage gaps requiring workarounds, with closure analysis:
  - "Bumping target to 2.12 would close 1 of 5 gaps."
  - "Bumping target to 2.13 would close 4 of 5 gaps."
  - "1 gap is structural (`at::TensorIterator`) and won't close at any runtime version."
  This lets the user weigh "narrower runtime support" against "fewer workarounds to write."
- Hard blockers, if any.

## Handoff

If the user wants to proceed, invoke `orchestrate-stable-abi-migration` and pass it the path to the plan JSON.
