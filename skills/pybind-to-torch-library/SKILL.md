---
name: pybind-to-torch-library
description: Convert a PyTorch extension's pybind11 bindings to TORCH_LIBRARY registrations. Use when the user wants to migrate `PYBIND11_MODULE` / pybind `m.def` style bindings to `TORCH_LIBRARY` + `TORCH_LIBRARY_IMPL`, which is a prerequisite for then migrating to the stable ABI's `STABLE_TORCH_LIBRARY_*` macros.
allowed-tools: Read, Bash, Edit, Glob, Grep
license: BSD-3-Clause
---

# Pybind → TORCH_LIBRARY

`STABLE_TORCH_LIBRARY_IMPL` (the stable-ABI form) requires that operators are registered through `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL`, not through pybind. This step migrates pybind-style bindings to the dispatcher-aware `TORCH_LIBRARY` form so that `migrate-to-stable-apis` can later swap them to the stable macros.

This is **not** the stable-ABI migration itself — it's a structural prerequisite.

## When to use

The migration plan flagged at least one file with the `pybind` concern, i.e., contains:
- `#include <pybind11/...>` (other than indirect inclusion via `torch/extension.h`)
- `PYBIND11_MODULE(...)`
- `m.def("name", &fn, "docstring")` where `m` is a `pybind11::module_`

## Inputs

- `plan_path`: path to the migration plan JSON, OR
- A list of files with the `pybind` concern.

## Migration steps

### 1. Define a schema for each op

`TORCH_LIBRARY` requires a typed schema string. Write each op's schema optimizing for **honesty and minimality**. When they conflict, honesty wins.

1. **Honest — the schema must accurately describe the op.** Declare the real argument/return types, in-place mutation (`Tensor(a!)`), and input↔output aliasing (`Tensor(a)`, no `!`, when the op can return one of its inputs). An under-declared schema — a plain `Tensor`/`Tensor?` that hides a mutation or an aliased return — is a correctness bug, not a simplification. The full schema format and alias/mutation grammar is specified in [`aten/src/ATen/native/README.md`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md); consult it for anything non-trivial.

2. **Minimal — subject to honesty, change a C++ signature only as much as the dispatcher requires.** `c10`/`infer_schema` accepts only certain types, so some changes are unavoidable: `int`→`int64_t`, `float`→`double` (per `infer_schema.h`: *"Only int8_t/int64_t/bool…"*, *"float is not supported, use double instead"*), and non-const-ref params (e.g. `std::optional<at::Tensor>&`) → by-value. Make exactly those. A read-only `const at::Tensor&` stays `const at::Tensor&` — don't add/drop `const`, switch by-value/by-reference unnecessarily, reorder args, or change defaults. Never trade an accurate schema for a smaller diff.

For each pybind-exported function, write a schema:

```cpp
// Pybind (old):
m.def("my_quantize", &my_quantize, "Quantize input");

// TORCH_LIBRARY schema (new):
// In a TORCH_LIBRARY block:
m.def("my_quantize(Tensor input, int block_size, str format) -> (Tensor, Tensor)");
```

Schema rules (subset — see the [C++ custom-ops tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) and [Custom Operators Manual](https://docs.pytorch.org/docs/main/notes/custom_operators.html) for the full grammar):

- Tensor inputs: `Tensor` (or `Tensor?` for optional, `Tensor[]` for list).
- Scalars: `int`, `float`, `bool`, `str`, `Scalar`, `SymInt`.
- Returns: a single type, or a parenthesized tuple `(Tensor, Tensor)`.
- **`float` in the schema corresponds to the C++ `double` type and the Python `float` type** — don't try to use a 32-bit float here.

### Mutable arguments

If the pybind function writes into one of its tensor arguments in place (e.g., `out`-style functions, `result_ptr[i] = ...` against a non-locally-allocated tensor), the schema must declare it mutable with `Tensor(a!)`:

```cpp
// Schema for an out-style op:
m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
```

Rules for mutable args:

- One alias annotation per mutable tensor — use different letters when multiple are mutated: `Tensor(a!)`, `Tensor(b!)`, `Tensor(c!)`.
- **Do NOT return a mutated tensor as the op's output.** Returning it breaks `torch.compile` and other subsystems that rely on mutation being declared distinctly from the return path. Out-style ops typically return `()` (void) and let the caller use the `out` argument it passed in.
- If you can't tell whether a pybind function mutates its inputs (e.g., the C++ implementation is complex), stop and ask the user. Guessing here causes silent correctness bugs downstream.

### 2. Register the schema once per library namespace

Schemas go in `TORCH_LIBRARY(libname, m)`. Pick a stable library name (often the Python package name). Typical layout in a single C++ file or a dedicated `registration.cpp`:

```cpp
#include <torch/library.h>

TORCH_LIBRARY(mylib, m) {
    m.def("my_quantize(Tensor input, int block_size, str format) -> (Tensor, Tensor)");
    m.def("other_op(Tensor a, Tensor b) -> Tensor");
}
```

`TORCH_LIBRARY` must appear **exactly once per (library, namespace)** in the build. Duplicate registrations cause runtime errors.

### 3. Register kernels per dispatch key

Replace `m.def(name, &fn)` (pybind) with `TORCH_LIBRARY_IMPL(libname, DispatchKey, m)` + `m.impl(name, &fn)`:

```cpp
TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
    m.impl("my_quantize", &my_quantize);
}

TORCH_LIBRARY_IMPL(mylib, CPU, m) {
    m.impl("my_quantize_cpu", &my_quantize_cpu);
}
```

Common dispatch keys:
- `CPU`, `CUDA` — device-specific kernels.
- `Autograd` — custom backward (rare in extensions; prefer composing with autograd via `torch.library.custom_op`).
- `CompositeImplicitAutograd` — pure tensor compositions.
- `Meta` — shape/dtype inference (see [migrate-meta-fns-to-python](../migrate-meta-fns-to-python/SKILL.md); prefer Python registration).

### 4. Expose to Python

Remove the `PYBIND11_MODULE` block entirely once kernels are registered through the dispatcher. Operators are then available from Python as `torch.ops.<libname>.<op>(...)`.

The shared library still needs to be loaded for the static registration to run. In the package's `__init__.py`:

```python
# In <package>/__init__.py
import torch
from pathlib import Path

so_files = list(Path(__file__).parent.glob("_C*.so"))
assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"
torch.ops.load_library(so_files[0])
```

Notes:
- The `.so` filename pattern depends on whether `py_limited_api` is enabled (see Step 7). With the limited API, the file is named like `_C.abi3.so` rather than `_C.cpython-311-x86_64-linux-gnu.so` — the glob `_C*.so` matches both.
- **Do not** define a dummy `_C` module from C++ purely so Python can import it. The `_C` name is just the binary's filename; operators reach Python only through `torch.ops`.

### 5. Update Python call sites

Most libraries already wrap their C++ ops in a Python module (e.g., `_custom_ops.py`, `ops.py`, `<lib>_interface.py`) that defines a typed Python function per op. Public users import from there, not from `_C` directly. In that case the migration is mechanical:

- Find the wrapper module(s) that import from the pybind module.
- In each wrapper function, replace the pybind call with `torch.ops.<libname>.<op>(...)`.
- **Do not** change the wrapper's name, signature, docstring, or import path — public callers must continue to work unchanged.

```python
# OLD (inside e.g. mylib/_custom_ops.py)
from mylib._C import my_quantize as _my_quantize_cpp

def my_quantize(input, block_size, format):
    return _my_quantize_cpp(input, block_size, format)

# NEW
def my_quantize(input, block_size, format):
    return torch.ops.mylib.my_quantize(input, block_size, format)
```

**Remove side-effect `_C` imports.** Pre-migration code often contains lines like:

```python
import mylib._C  # noqa: F401  — triggers static registration
```

After this skill runs, `_C` is no longer an importable Python module — it's just the shared library's filename, loaded via `torch.ops.load_library(...)` in `__init__.py` (see Step 4). The side-effect import will raise `ModuleNotFoundError` at runtime.

It's also no longer needed: any `from mylib.<submodule>` import transitively runs `mylib/__init__.py` first, which calls `load_library` and triggers all the same static registrations. Delete the side-effect lines.

```bash
# Find them:
grep -rn 'import .*\._C\b' <package>/ | grep -E '# noqa|F401'
```

**If the library exposes `_C` directly to users** — i.e., public code paths do `from mylib._C import some_op` rather than going through a wrapper — STOP and ask the user before proceeding. Common in small or older extensions; rare in mature ones (vllm, flash-attention, etc., all route through wrapper modules). The user has to decide whether it's acceptable to introduce a wrapper module (mild breakage for downstream callers) or to keep the old import path alive as a Python shim that re-exports from `torch.ops`.

To check, grep the public surface for direct `_C` symbol usage outside known wrapper modules:

```bash
grep -rn 'from .*\._C import\|\._C\.\w' <package>/ \
    | grep -v '# noqa\|F401' \
    | grep -vE '(_custom_ops|ops|interface|kernels)\.py'
```

Empty result → safe to proceed mechanically. Hits → review with the user.

### 6. Note on `STABLE_TORCH_LIBRARY`

This skill targets the non-stable `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL` macros intentionally. The conversion to `STABLE_TORCH_LIBRARY` / `STABLE_TORCH_LIBRARY_IMPL` (plus `TORCH_BOX`) happens later in `migrate-to-stable-apis`, alongside the rest of the API rewrite. Splitting it into two hops keeps each step's diff focused.

### 7. [OPTIONAL] Enable `Py_LIMITED_API` (build one wheel across Python versions)

Now that pybind11 should be gone, check if the extension uses any other CPython APIs. If not, the extension can be compiled against CPython's Stable Limited API. This means one binary works across all Python 3.x versions ≥ the chosen minimum, which (combined with `TORCH_TARGET_VERSION` added later in `migrate-to-stable-apis`) collapses the build matrix to a single wheel.

Update `setup.py`:

```python
from torch.utils import cpp_extension
from setuptools import setup

setup(
    name="<package>",
    ext_modules=[
        cpp_extension.CUDAExtension(
            "<package>._C",
            sources=[...],
            extra_compile_args={
                "cxx": [
                    "-DPy_LIMITED_API=0x030a0000",  # 3.10 minimum
                ],
                "nvcc": [
                    "-DUSE_CUDA",
                ],
            },
            py_limited_api=True,
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
```

What each piece does:

- `-DPy_LIMITED_API=0x030a0000` — verifies at compile time that only the CPython Limited API is used. Pick the lowest Python version the project is willing to support (3.10 is a common floor).
- `py_limited_api=True` — tells `cpp_extension` to build with the limited-API ABI tag.
- `bdist_wheel py_limited_api=cp310` — produces a wheel named `*-cp310-abi3-...` instead of one tied to a single Python ABI.

Build and confirm the resulting `.so` is named `_C.abi3.so` (the `abi3` tag is the marker that the limited API is in use). Run the project's existing tests against it.

If the project still has any direct CPython API usage (rare, since pybind was the source of most of it), the compile will fail with `Py_LIMITED_API` errors pointing at the unstable calls. Refactor those out or, if there's a hard dependency, defer this step and report the blockers. This step is not necessary for LibTorch ABI stability, so it is only nice to have.

## Critical rules

1. **Schema must match function signature exactly.** Mismatched arg count or types cause cryptic dispatcher errors at call time.
2. **Pick one library name per project and stick to it.** Splitting an op across two `TORCH_LIBRARY` blocks with the same name in the same shared lib is an error. Use `TORCH_LIBRARY_FRAGMENT` to register ops to the same library across multiple files.
3. **Do not keep both pybind and `TORCH_LIBRARY` registrations for the same function** — this produces two Python entry points and confuses users.
4. **Mutable args (`Tensor(a!)`)** — if any pybind function mutates inputs in place, that needs to be reflected in the schema with alias annotations. Don't guess; flag for the user. Remember, honesty and minimality!
5. **Variadic / `*args` Python signatures** are not expressible in schemas. Refactor to fixed argument lists.

## Verification

After conversion:
- `python -c "import torch, <package>; print(torch.ops.<libname>.<op>)"` should succeed.
- The project's existing test suite (`test_cmd`) should pass unchanged.
- If tests import directly from `_C`, update those imports to use `torch.ops.<libname>`.
- If `Py_LIMITED_API` was enabled (Step 7), the built `.so` should be named `*.abi3.so`.

## Handoff

Marks `pybind` files as ready for downstream steps. The orchestrator will next dispatch `migrate-meta-fns-to-python` (if applicable) and then `scaffold-stable-target`. `STABLE_TORCH_LIBRARY` conversion and `TORCH_TARGET_VERSION` build flag come in `migrate-to-stable-apis`.
