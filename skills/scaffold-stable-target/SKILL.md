---
name: scaffold-stable-target
description: Set up a parallel ABI-stable build target alongside the existing legacy extension so files can be migrated into it incrementally by `migrate-to-stable-apis`. Both targets share the same `csrc/` directory — no files physically move; each source file's path simply switches `ext_modules` entries as it's migrated. Both `.so`'s load at import time and register into the same `torch.ops.<libname>` namespace, so Python callers never change. The end state is one stable wheel.
allowed-tools: Read, Bash, Edit, Write, Glob, Grep
license: BSD-3-Clause
---

# Set Up Parallel Stable Build Target

The goal of this skill is to **stand up a new ABI-stable wheel target** that will eventually replace the legacy extension entirely. By the end of the full migration:

- The new stable target compiles every C++/CUDA source with `-DTORCH_TARGET_VERSION` (and, if the user opted in, `-DPy_LIMITED_API`).
- The legacy target's `sources=` list is empty and the entry is removed.
- The project ships **one** wheel — the stable one.

`TORCH_TARGET_VERSION` is mandatory — it's the flag that makes the binary work across torch versions. `Py_LIMITED_API` is optional and additive: turning it on collapses the per-Python-version dimension of the build matrix too, but the migration delivers most of its value without it. Default to suggesting it; defer to the user if pybind or other CPython-API code makes it inconvenient right now.

This skill sets up the scaffolding. The actual per-file rewrite + `sources=` flip happens in `migrate-to-stable-apis`.

## The model: shared `csrc/`, shared torch.ops namespace

Files **do not move directories**. The same `csrc/` directory is referenced by both `ext_modules` entries; each file's path lives in exactly one `sources=` list at a time. Migration means: rewrite the file in place + move its path string from the legacy `sources=` to the stable `sources=`.

Both `.so`'s are loaded at import time and both register into the **same** `torch.ops.<libname>` namespace. Python callers keep calling `torch.ops.<libname>.<op>(...)` throughout — they never learn there are now two `.so` files behind it. The dispatcher is fine with this because each op is registered exactly once (it's compiled into exactly one `.so` at any time).

This is the pattern used by [sglang's sgl-kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel) — `torch.ops.sgl_kernel.<op>` is the contract; the `.so` filename is implementation detail.

## When to use

Run this skill for **incremental** migrations — when files will be rewritten over multiple commits/PRs and the legacy target needs to keep serving un-migrated ops throughout. The parallel target's whole purpose is to keep the build green between file moves.

**Skip this skill for one-shot migrations.** If the project is small enough that every in-scope file can be rewritten in a single PR (typically only a handful of translation units), there is no half-migrated state to protect against — `migrate-to-stable-apis` just adds `TORCH_TARGET_VERSION` directly to the existing `ext_modules` entry and rewrites all the files in one pass. Setting up a parallel target for a one-shot migration is pure overhead.

The cadence decision is made during `assess-abi-migration` Step 7 and recorded in the plan's `dispatch_order`. The orchestrator dispatches this skill iff it's listed there.

## Why a parallel target

The migration is multi-step (API substitutions, build flag changes, registration macro swaps). Without a parallel target:

- The repo is in a half-migrated, non-building state for the duration.
- Reverting a problematic file means undoing several intertwined changes.
- The legacy build can't be exercised for comparison during migration.

A parallel target sidesteps all three: the legacy extension keeps building and serving every not-yet-migrated op, and progress is visible as files accumulate in the stable target's `sources=`.

## Inputs

- `plan_path`: path to the migration plan JSON.
- `legacy_ext_name`: the existing extension module name (e.g., `extension_cpp._C`). Discovered from `setup.py`.
- `stable_ext_name` (optional): the new extension module name. Default: `<legacy_ext_name>_stable` (e.g., `extension_cpp._C_stable`).
- `lib_namespace`: the existing `torch.ops` namespace used by registrations (e.g., `extension_cpp`). Discovered from the existing `TORCH_LIBRARY(<name>, m)` block. **This stays the same** for the stable target — both `.so`'s register into it.

## Setup steps

### 1. Add the stable extension entry to `setup.py`

Keep the legacy `ext_modules` entry untouched. Add a second entry pointing at the same `csrc/` directory with empty `sources=` and the stable flags:

```python
from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="extension_cpp",
    ext_modules=[
        # Legacy — unchanged at setup time. Shrinks as migrate-to-stable-apis
        # moves paths into the stable entry below.
        cpp_extension.CUDAExtension(
            "extension_cpp._C",
            sources=[
                "extension_cpp/csrc/muladd.cu",
                "extension_cpp/csrc/other.cu",
            ],
        ),
        # New stable target — starts with no sources and grows.
        # Note: same csrc/ paths will appear here as files migrate; legacy entry's
        # sources= shrinks at the same time. A file is in exactly one list.
        cpp_extension.CUDAExtension(
            "extension_cpp._C_stable",
            sources=[],
            extra_compile_args={
                "cxx": [
                    "-DTORCH_TARGET_VERSION=0x020b000000000000",
                    # Optional — uncomment to also collapse the per-Python-version
                    # build matrix. See "Optional: Py_LIMITED_API" below.
                    # "-DPy_LIMITED_API=0x03090000",
                ],
                "nvcc": [
                    "-DTORCH_TARGET_VERSION=0x020b000000000000",
                    "-DUSE_CUDA",
                ],
            },
            # py_limited_api=True,   # Pair with -DPy_LIMITED_API if enabling.
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    # options={"bdist_wheel": {"py_limited_api": "cp39"}},   # Same — pair with Py_LIMITED_API.
)
```

Pick `TORCH_TARGET_VERSION` based on the plan's `target_torch_version`. See [migrate-to-stable-apis](../migrate-to-stable-apis/SKILL.md) for the flag's two purposes.

#### Optional: `Py_LIMITED_API`

Adding `Py_LIMITED_API` (plus `py_limited_api=True` on the Extension and `bdist_wheel.py_limited_api` in `setup()`) collapses the per-Python-version dimension of the build matrix — one wheel works across every CPython ≥ the chosen floor. It's not required for the LibTorch ABI migration, but it's cheap to enable here because the legacy target is left untouched and the stable target is starting fresh.

Offer it to the user when running this skill:

> "Want me to also enable `Py_LIMITED_API` on the stable target? It produces a single wheel across CPython versions on top of the cross-torch-version wheel. Recommended unless the project still uses CPython APIs that aren't in the Limited API (rare once pybind is gone)."

If yes, uncomment the three lines in the example above. If no, leave them commented — the migration still delivers cross-torch-version compatibility, just not cross-Python-version. Can always be flipped on later by uncommenting and rebuilding.

### 2. Load the stable `.so` from `__init__.py`

Both `.so`'s load. Both register into the same `torch.ops.<lib_namespace>` namespace. The legacy loader stays as-is; just add a second `load_library` call for the stable `.so`:

```python
# extension_cpp/__init__.py
import torch
from pathlib import Path

_pkg_dir = Path(__file__).parent

# Legacy (existing) loader — unchanged.
_legacy_sos = list(_pkg_dir.glob("_C*.so"))
_stable_sos = [p for p in _legacy_sos if "_C_stable" in p.name]
_legacy_sos = [p for p in _legacy_sos if p not in _stable_sos]
assert len(_legacy_sos) == 1, f"Expected one legacy _C*.so, found {len(_legacy_sos)}"
torch.ops.load_library(_legacy_sos[0])

# Stable loader — new. Once migrate-to-stable-apis has moved at least one file
# into the stable entry, _C_stable*.so exists.
if _stable_sos:
    assert len(_stable_sos) == 1, f"Expected one _C_stable*.so, found {len(_stable_sos)}"
    torch.ops.load_library(_stable_sos[0])

# Python-side registrations (fake kernels, autograd) stay where they are —
# they register against torch.ops.<lib_namespace>.<op>, which is the same
# namespace regardless of which .so implements the op.
from . import ops  # noqa: F401
```

The guard around the stable loader matters: until the first file is migrated, the stable target produces no `.so` (or an empty one, depending on the build system) and the loader would crash trying to dlopen it. After the first migration the stable `.so` exists and the guard becomes redundant — that's fine.

### 3. Verify both targets configure

```bash
<build_cmd>
```

Expected:
- Legacy target builds and links exactly as before — unchanged behavior.
- Stable target either builds an empty `.so` (no sources yet) or is skipped by the build system because `sources=[]`. Both are acceptable; the empty case proves the flags are accepted.

If the stable target's build system rejects an empty `sources=`, temporarily add a single trivial `.cpp` containing only:

```cpp
#include <torch/csrc/stable/library.h>
STABLE_TORCH_LIBRARY(<lib_namespace>, m) {}
```

`migrate-to-stable-apis` will move real registrations into this file (or replace it) once the first source migrates. The empty `STABLE_TORCH_LIBRARY` block does not conflict with the legacy `TORCH_LIBRARY(<lib_namespace>, m)` block because schema fragments are additive.

### 4. Run the project's existing tests

The legacy target is untouched, so `test_cmd` should still pass. This establishes a green baseline before file migration begins.

## Critical rules

1. **Do not move files in this step.** Setup only. Every file currently in the legacy `sources=` stays exactly where it is on disk. File `sources=` movement is `migrate-to-stable-apis`' job, and even that is just an edit to `setup.py`, not a `mv` on disk.
2. **Use the SAME `torch.ops` namespace for the stable target.** The library name passed to `STABLE_TORCH_LIBRARY(<libname>, m)` must match the existing `TORCH_LIBRARY(<libname>, m)` namespace — typically the project's own name (e.g., `extension_cpp`). This is what lets Python callers keep working unchanged. The stable target's `.so` filename being different (`_C_stable.so`) is fine; only the registration namespace must match.
3. **Do NOT use a `_stable`-suffixed `torch.ops` namespace** (e.g., `extension_cpp_stable`). That would force every Python caller in the ecosystem to dual-path their call sites. The whole point of the shared-namespace pattern is that they don't.
4. **Do not delete the legacy entry yet.** Decommissioning happens after all files have moved and tests pass — see the Completion section below.
5. **Keep `TORCH_TARGET_VERSION` (and `Py_LIMITED_API`, if enabled) only on the stable target.** The legacy target must not have either flag; the whole reason it exists during migration is that it can still use the unstable APIs the un-migrated files depend on.

## Completion (run by `migrate-to-stable-apis` after the last file migrates)

When the legacy `sources=` is empty:

1. Remove the legacy `ext_modules` entry from `setup.py`.
2. Optional: rename the stable extension from `_C_stable` back to `_C` (and remove the suffix-handling logic in `__init__.py`) for cleaner packaging. This is cosmetic — the `torch.ops.<libname>` namespace is what users actually depend on and that hasn't changed.
3. Final build + test against the now-single `_C_stable` (or renamed `_C`) target.

## Handoff

The orchestrator next dispatches `migrate-to-stable-apis`, which performs the file-by-file rewrite + `sources=` flip loop into the stable target this skill just set up.
