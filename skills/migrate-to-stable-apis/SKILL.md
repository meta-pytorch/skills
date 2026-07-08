---
name: migrate-to-stable-apis
description: >-
  Rewrite unstable PyTorch C++/CUDA API calls to their `torch::stable` /
  `torch::headeronly` / `aoti_torch` equivalents and swap registration macros
  to `STABLE_TORCH_LIBRARY_*` with `TORCH_BOX`. Runs in two modes:
  **incremental** (file-by-file, after `scaffold-stable-target` set up a
  parallel target — each file's path flips from legacy `sources=` to stable
  `sources=` as it migrates) or **one-shot** (no parallel target — adds
  `TORCH_TARGET_VERSION` to the existing `ext_modules` entry and rewrites
  every in-scope file in one pass). Use after `pybind-to-torch-library`,
  `migrate-meta-fns-to-python`, and `migrate-autograd-fns-to-python` have run.
allowed-tools: Read, Bash, Edit, Write, Glob, Grep
license: BSD-3-Clause
---

# Migrate to Stable APIs

This is the bulk-rewrite step. By the time this skill runs, the project should:

- Use `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` for op registration (pybind removed).
- Have any C++ Meta impls moved to Python `register_fake`.
- Have any C++ Autograd impls moved to Python `register_autograd`.

The plan's `dispatch_order` (set by `assess-abi-migration` Step 7) determines which mode this skill runs in.

## Two modes

**Incremental** (when `scaffold-stable-target` ran). A second `ext_modules` entry (stable, with `TORCH_TARGET_VERSION` flags) already exists alongside the legacy one. Both share the same `csrc/`. Files stay where they are on disk; each file is in exactly one entry's `sources=` at any time. Migrating a file = rewrite it in place + remove its path from the legacy `sources=` and add it to the stable `sources=`. Both `.so`'s load at import time and register into the same `torch.ops.<lib_namespace>` namespace (the dispatcher is fine with this because each op is compiled into only one `.so`). The legacy target keeps serving un-migrated ops throughout.

**One-shot** (when `scaffold-stable-target` was skipped). There is no parallel target. This skill adds `TORCH_TARGET_VERSION` (and optionally `Py_LIMITED_API`) to the existing `ext_modules` entry, then rewrites every in-scope file's APIs + registration macros, then builds + tests the now-stable extension. No `sources=` flipping (there's nothing to flip between). The wheel built after this skill is the final stable wheel.

The rewrite work (steps 1–3 below) is identical in both modes. Step 4 (build target wiring) differs.

## Prerequisites

- PyTorch 2.10+, the same version as the desired build version (verify: `python3 -c "import torch; print(torch.__version__)"`).
- The full reference docs in this skill's `references/`:
  - [api-mapping.md](references/api-mapping.md) — old API → stable replacement table.
  - [migration-example.md](references/migration-example.md) — concrete before/after.
  - [common-issues.md](references/common-issues.md) — known build/runtime pitfalls.
  - [stack-and-boxing.md](references/stack-and-boxing.md) — `TORCH_BOX` internals, `StableIValue` conversions, `torch_call_dispatcher`. Only needed for advanced cases.

## Inputs

- `plan_path`: path to the migration plan JSON, OR ask the user for the following:
- Mode: derived from whether `scaffold-stable-target` appears in the plan's `dispatch_order`.
- File list: the plan's `unstable-api` concern entries (incremental mode can also cross-check against the legacy `ext_modules` entry's `sources=`).
- `lib_namespace`: the existing registration namespace (e.g., `extension_cpp`). Stays the same throughout the migration — do **not** suffix it with `_stable`.

## Workflow

**Incremental mode**: plan chunk order first (step 0 below), then process one chunk at a time. After each chunk is fully migrated and both targets build + tests pass, commit and advance. Within a chunk, work file-by-file so failures stay localized; the chunk boundary is what gets committed.

**One-shot mode**: rewrite every in-scope file in one pass (steps 1–3 below), then do step 4 once for the whole extension, then build + test once. The whole migration lands as a single change. Skip step 0.

### 0. (Incremental mode) Plan chunk order before any rewrites

Before touching any files, plan how the migration will land. The wrong order forces follow-up chunks to wait on dependencies; the right order unblocks parallel work.

**Strategy**

1. **Identify shared dependencies.** Grep the in-scope files for:
   - Shared helper headers (`utils.h`, `dispatch_utils.h`, `<project>_ops.h`, etc.) that multiple `.cpp`/`.cu` files include.
   - Common helper functions / inline templates defined in those headers.
   - `TORCH_LIBRARY` schema definitions (the `m.def(...)` blocks, which usually live in one `torch_bindings.cpp`-style file) — `TORCH_LIBRARY_IMPL` impls depend on the schemas being registered first.

   These are the **foundation files**: anything that other in-scope files transitively depend on.

2. **Group files into chunks.** Sensible grouping criteria, in priority order:
   - **Foundation first — bundled with at least one real consumer.** The first chunk contains the shared headers + the schema definition file **plus at least one migrated kernel that actually uses them**. Don't land foundation as a standalone chunk: a PR that adds stable helpers nobody calls reads as dead code to reviewers and doesn't prove the helpers actually work end-to-end. Pick a small, representative kernel from the same family as the helpers (e.g., the simplest op in `csrc/attention/` if you're migrating attention helpers) and migrate it in the same chunk. Once this lands, downstream chunks can be developed in parallel by different contributors.
   - **Same directory / same kernel family.** Files in `csrc/quantization/`, `csrc/attention/`, etc. usually share helpers and review context. Keeping a chunk within one subdirectory makes the diff easier to review.
   - **Same coverage gaps.** Files that all need the same workaround (e.g., the `empty_strided` rewrite from [common-issues.md](references/common-issues.md)) chunk well together — the reviewer sees the workaround once.

3. **Cap chunk size at ~1000 lines of diff.** This is a soft ceiling. Reviewers stall on bigger diffs; CI cycles get longer; bisecting a regression gets harder. Estimate per file roughly: each migrated file ≈ original LOC × 1.5 (includes + API swaps add lines). If a single file blows past 1k by itself, that's fine — it's one file, the diff is unavoidable — but don't combine it with others.

4. **Order the chunks so downstream work isn't blocked.** Foundation chunk #1. Then chunks that touch independent areas can be ordered by complexity (simple first to build confidence) or by who's working on them (if multiple contributors). A chunk that depends on another chunk's helper-migration must come after.

**Output of this step**

A short ordered list of chunks, each with:
- A 1-line title (e.g., `chunk-01: shared headers + torch_bindings + first attention kernel`).
- The list of files it touches. **Chunk 1 must include at least one real kernel migration** that exercises the foundation, not just the foundation files alone.
- Estimated diff size (rough).
- Which upstream chunks (by number) it depends on, if any.

Surface this list to the user before starting any rewrites:

> "Planned `<N>` chunks for this migration. Chunk 1 (`<title>`) lands the shared helpers — review and merge it first so chunks 2–`<N>` can be worked in parallel. Estimated diff sizes: …. OK to proceed?"

After the user approves, save the chunk plan to the migration plan JSON under a `chunks` field so the orchestrator can resume mid-migration without re-planning.

The per-chunk loop is steps 1–7 below — apply them to every file in the chunk, then build + test + commit + advance to the next chunk.

### 1. Update includes

Remove legacy includes and replace with the stable subset.

```cpp
// REMOVE:
// #include <torch/extension.h>
// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <torch/library.h>

// ADD (include only what you need):
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>

#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

// For CUDA stream access:
#include <torch/csrc/inductor/aoti_torch/c/shim.h>  // or torch/csrc/stable/c/shim.h on newer versions
#include <cuda_runtime.h>
```

Allowed header roots (anything outside these will fail to compile under `TORCH_TARGET_VERSION`):

- `torch/csrc/stable/`
- `torch/headeronly/`
- `torch/csrc/inductor/aoti_torch/`

### 2. Apply API replacements

Walk each file and apply the substitutions from [references/api-mapping.md](references/api-mapping.md). The most frequent ones:

| Old | Stable |
|---|---|
| `at::Tensor` / `torch::Tensor` | `torch::stable::Tensor` |
| `TORCH_CHECK(...)` | `STD_TORCH_CHECK(...)` (arguments unchanged) |
| `at::empty({...}, options)` | `torch::stable::new_empty(ref, {sizes}, dtype)` |
| `at::zeros({...}, options)` | `torch::stable::new_zeros(ref, {sizes}, dtype)` |
| `at::empty_like(t)` | `torch::stable::empty_like(t)` |
| `t.contiguous()` | `torch::stable::contiguous(t)` |
| `t.data_ptr<T>()` (read) | `t.const_data_ptr<T>()` |
| `t.data_ptr<T>()` (write) | `t.mutable_data_ptr<T>()` |
| `t.sizes()` / `t.strides()` | `t.size(i)` / `t.stride(i)` (per-element) |
| `at::kFloat` | `torch::headeronly::ScalarType::Float` |
| `at::kCUDA` | `torch::headeronly::kCUDA` |
| `tensor.device() == kCUDA` | `tensor.device().type() == torch::headeronly::kCUDA` |
| `at::pad(x, ...)` / `x.pad(...)` | `torch::stable::pad(x, ...)` (always free-function form; Tensor methods → ATen variant) |
| `c10::cuda::CUDAGuard guard(t.device())` | `torch::stable::accelerator::DeviceGuard guard(t.get_device_index())` |
| `at::cuda::getCurrentCUDAStream()` | `aoti_torch_get_current_cuda_stream(t.get_device_index(), &stream_ptr)` (see common-issues) |
| `C10_CUDA_KERNEL_LAUNCH_CHECK()` | `STD_CUDA_KERNEL_LAUNCH_CHECK()` |

See [api-mapping.md](references/api-mapping.md) for the full table, including strided tensor creation, layout checks, and scalar type enums.

### 3. Swap registration macros to STABLE_*

After this step, op registrations move from `TORCH_LIBRARY` (set up by `pybind-to-torch-library`) to the stable form:

```cpp
// Before:
TORCH_LIBRARY(mylib, m) {
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
}
TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
    m.impl("mymuladd", &mymuladd_cuda);
}

// After:
STABLE_TORCH_LIBRARY(mylib, m) {
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
}
STABLE_TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
    m.impl("mymuladd", TORCH_BOX(&mymuladd_cuda));
}
```

Rules:

- `TORCH_LIBRARY` → `STABLE_TORCH_LIBRARY` (schema definitions).
- `TORCH_LIBRARY_FRAGMENT` → `STABLE_TORCH_LIBRARY_FRAGMENT`.
- `TORCH_LIBRARY_IMPL` → `STABLE_TORCH_LIBRARY_IMPL`.
- Wrap each kernel function pointer in `TORCH_BOX(&fn)` inside `m.impl(...)`.
- Do **not** wrap the op name in `TORCH_SELECTIVE_NAME`.
- Do **not** wrap the function pointer in `TORCH_FN`.
- Do **not** manually write boxed kernels — `TORCH_BOX` generates the boxed wrapper that adapts `(stack, num_args, num_outputs)` to your typed function signature using `to<T>` / `from<T>` over `StableIValue`.

**Keep the library namespace the same.** `STABLE_TORCH_LIBRARY(<lib_namespace>, m)` uses exactly the namespace the legacy `TORCH_LIBRARY(<lib_namespace>, m)` used — do NOT add a `_stable` suffix. Both `.so`'s register into the same `torch.ops.<lib_namespace>` namespace; the dispatcher accepts this because each file (and therefore each op) is compiled into only one `.so` at a time. Python callers continue to use `torch.ops.<lib_namespace>.<op>(...)` unchanged throughout the migration.

### 4. Wire up the stable build flags

**Incremental mode.** In `setup.py`:
- Remove the file's path from the legacy `ext_modules` entry's `sources=`.
- Add the same path string to the stable `ext_modules` entry's `sources=`.

The file does not move on disk. Both entries reference the same `csrc/` directory; this edit just changes which target compiles it. Build flags are already set on the stable target by `scaffold-stable-target` (`TORCH_TARGET_VERSION`, plus `Py_LIMITED_API` if enabled there). Do not touch the legacy target's flags.

**One-shot mode.** Once on the first file (or in a single setup.py edit covering the whole migration), add `TORCH_TARGET_VERSION` to the existing `ext_modules` entry's `extra_compile_args`:

```python
extra_compile_args={
    "cxx": [
        "-DTORCH_TARGET_VERSION=0x020b000000000000",
        # Optional — uncomment to also collapse the per-Python-version build matrix.
        # "-DPy_LIMITED_API=0x03090000",
    ],
    "nvcc": [
        "-DTORCH_TARGET_VERSION=0x020b000000000000",
        "-DUSE_CUDA",
    ],
},
# py_limited_api=True,   # Pair with -DPy_LIMITED_API if enabling.
```

Pick the hex from the plan's `target_torch_version` (`0xMMmm_0000_0000_0000` — e.g., `0x020b000000000000` for 2.11). Offer the `Py_LIMITED_API` opt-in here the same way `scaffold-stable-target` does in the incremental case — it's not required for the LibTorch ABI migration but is cheap to enable now.

`TORCH_TARGET_VERSION` serves two purposes:

1. **Bans unstable headers.** Any `#include <ATen/...>`, `<c10/...>`, `<torch/extension.h>`, etc. in a translation unit compiled with this flag fails with:
   > "This file should not be included when either `TORCH_STABLE_ONLY` or `TORCH_TARGET_VERSION` is defined."
   This is your compile-time guarantee that the binary is ABI-stable.
2. **Selects the minimum runtime PyTorch version.** Shims in `shim.h` are guarded by `#if TORCH_FEATURE_VERSION >= TORCH_VERSION_X_Y_Z`. APIs added after the target version are unavailable. The compiled binary will load against any PyTorch ≥ the target version.

Constraint: `2.9 ≤ TORCH_TARGET_VERSION ≤ <libtorch version at build time>`. Choose the lowest version the project is willing to support; lower = wider compatibility, higher = more APIs.

### 5. Build and iterate

Run `build_cmd`. Both targets (incremental) or the single converted target (one-shot) must build successfully. Most failures fall into two buckets:

- **"Header not allowed" errors** — a legacy include slipped through the rewrite. Find and replace it. If the file genuinely needs an API with no stable equivalent, see [references/common-issues.md](references/common-issues.md) for workarounds, or stop and flag a blocker.
- **Substitution misses** — an unstable API call or `TORCH_LIBRARY_IMPL` macro wasn't replaced. Apply the mapping.

Common pitfalls are documented in [references/common-issues.md](references/common-issues.md): missing `-DUSE_CUDA`, undeclared `aoti_torch_get_current_cuda_stream`, layout check rewrites, and the `empty_strided` workaround.

### 6. Test

Run the project's existing test suite (`test_cmd` from the plan, or if that is too broad, look at the possible tests and only run those affecting the kernels we've migrated this chunk). Python-side registrations (fakes from `migrate-meta-fns-to-python`, autograd from `migrate-autograd-fns-to-python`) reference `torch.ops.<lib_namespace>.<op>` — that string is unchanged, so they keep working in both modes. When tests pass, prompt the user to make a commit.

In **incremental mode**, both `.so`'s load at import time; the migrated op now lives in `_C_stable.so` while everything else still resolves through `_C.so`. In **one-shot mode**, the single `.so` is now the stable one.

### 7. Advance

**Incremental mode.** Pick the next chunk of files from the legacy `ext_modules` entry's `sources=` in `setup.py` and repeat from Step 1. Stop when the legacy entry's `sources=` is empty, then hand off to [scaffold-stable-target's Completion section](../scaffold-stable-target/SKILL.md#completion-run-by-migrate-to-stable-apis-after-the-last-file-migrates) for final cleanup (removing the legacy entry, optionally renaming the stable extension back to `_C`).

**One-shot mode.** When all in-scope files are rewritten, the build is green, and tests pass — the migration is done. No further cleanup; the single existing `ext_modules` entry is now the stable wheel.

## Critical rules

These are mistakes that AI-assisted migrations commonly make — follow strictly:

1. **Keep the `torch.ops` namespace identical to the legacy one.** `STABLE_TORCH_LIBRARY(<lib_namespace>, m)` uses the same `<lib_namespace>` the legacy `TORCH_LIBRARY(<lib_namespace>, m)` used — never suffix it with `_stable`. Python callers (`torch.ops.<lib_namespace>.<op>`) and Python-side registrations (`@torch.library.register_fake("<lib_namespace>::<op>")`) reference this name and must keep working unchanged.
2. **Do NOT use `torch::stable::new_empty_strided`** — it does not exist. To create a strided tensor, create a contiguous tensor with transposed dimensions and call `torch::stable::transpose()`. See [common-issues.md § Strided Tensors](references/common-issues.md).
3. **Do NOT define a dummy `_C` module** accessible from Python. Operators reach Python through `torch.ops.<libname>.<op>`.
4. **Do NOT forward-declare `aoti_torch_get_current_cuda_stream`** — include it from `torch/csrc/inductor/aoti_torch/c/shim.h`.
5. **Do NOT hand-write boxed kernels** — use `TORCH_BOX(&fn)` in `m.impl()`.
6. **Do NOT change switch statements into if/else** as part of the migration. The stable ABI works fine with switch statements; rewriting just adds review noise.
7. **Preserve existing behavior and signatures — change ONLY what the migration strictly requires.** Swap the types the ABI needs (`at::Tensor`→`torch::stable::Tensor`, the macros, etc.) and nothing else. Do NOT add or drop `const`, switch a parameter between by-value and by-reference, reorder arguments, or change defaults. Gratuitous signature churn adds review noise, can subtly change behavior, and makes the migration commit hard to audit against the previous one.
8. **When replacing `TORCH_CHECK` with `STD_TORCH_CHECK`, change only the macro name** — leave the condition and message arguments untouched.
9. **For `DeviceGuard`**, pass `t.get_device_index()` (an integer), **not** `t.device()`. Do not cast to `char` or narrow type. DeviceGuard is RAII — don't move its declaration into an inner scope.
10. **For CUDA streams**, derive the device index from a tensor (`t.get_device_index()`), **not** from `torch::stable::accelerator::getCurrentDeviceIndex()`. Multi-GPU scenarios silently produce the wrong stream otherwise.
11. **Scalar type enums use the headeronly form** — `torch::headeronly::ScalarType::Float` (not `torch::kFloat32`), `::Half` (not `torch::kFloat16`), `::BFloat16`, `::Float8_e4m3fn`, etc.

## Verification

Build with `build_cmd`, then audit the built `.so` directly — this catches unstable symbols a passing test suite would miss:

```bash
torch-abi-audit <path/to/extension.so>   # or an import name; -v lists offending symbols
```

A clean report means the binary only depends on th e stable C-shim ABI. If it flags `c10::`/`at::` symbols, a legacy call slipped through — fix it before proceeding. (`pip install torch-abi-audit`.)

Then run `test_cmd`, and prove cross-version ABI stability empirically:

```bash
pip install torch==X.Y && <build_cmd>   # build against one minor
pip install torch==X.Z && <test_cmd>    # re-test against another, no rebuild
```

If the audit is clean and the second `test_cmd` passes, the migration is verified.

## Real-world references

- **Official tutorial:** [Custom C++ and CUDA Operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) — walks through the same migration end-to-end on a toy `mymuladd` op, with side-by-side ABI-stable vs non-stable tabs.
- **API-rewrite reference:** [`pytorch/extension-cpp`](https://github.com/pytorch/extension-cpp) — contains `extension_cpp/` (legacy) and `extension_cpp_stable/` (stable) implementations of the same op. Useful as a side-by-side diff for the API changes themselves. Note its parallel-directory layout is for didactic clarity; this skill's migration model keeps both targets in one `csrc/`.
- **Shared-namespace pattern:** [sglang's sgl-kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel) — Python callers use `torch.ops.sgl_kernel.<op>` regardless of which `.so` implements the op. The `.so` filename is implementation detail.
- **Real-world migration:** FlashAttention 3's [`flash_api.cpp`](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_api.cpp) (legacy) vs [`flash_api_stable.cpp`](https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_api_stable.cpp) (stable) — a production-scale example of the **API rewrite** (`at::` → `torch::stable::`, schema strings, `Tensor(out!)?` annotations).

  ⚠️ **Don't copy FA3's op registration.** It hand-writes boxed kernels (`boxed_mha_fwd(StableIValue* stack, …)`) — you don't need to. Register with `TORCH_BOX(&fn)` (Step 3 / critical rule 5).
- **Other adoptions:** xformers, torchaudio, torchao, vLLM (in progress).

## Handoff

This is typically the last step in the migration. After the last file is migrated:

- The plan's `dispatch_order` is fully `completed`.
- **Incremental mode:** the legacy `ext_modules` entry's `sources=` is empty. Return to [scaffold-stable-target § Completion](../scaffold-stable-target/SKILL.md#completion-run-by-migrate-to-stable-apis-after-the-last-file-migrates) to remove the legacy entry from `setup.py` and (optionally) rename the stable extension back to `_C`.
- **One-shot mode:** the single existing `ext_modules` entry now compiles under stable flags. No further cleanup needed.
- The orchestrator's final summary should note: target PyTorch version pinned, single-wheel goal achieved.
