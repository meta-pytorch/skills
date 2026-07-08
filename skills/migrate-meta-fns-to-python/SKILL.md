---
name: migrate-meta-fns-to-python
description: Move C++ Meta-dispatch implementations (shape/dtype inference kernels registered under the `Meta` dispatch key) into Python via `@torch.library.register_fake`. Use when the user wants to remove C++ Meta impls from a PyTorch extension as part of an ABI-stable migration, or to add torch.compile / fake-tensor support that didn't exist before.
allowed-tools: Read, Bash, Edit, Write, Grep
license: BSD-3-Clause
---

# Migrate C++ Meta Functions to Python

PyTorch's stable ABI does not expose enough of the dispatcher and SymInt machinery to implement Meta kernels cleanly in C++. The recommended pattern for modern extensions per the [official C++ custom-ops tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) is to register Meta (fake) kernels in **Python** using `torch.library.register_fake`, while leaving the device kernels (CPU, CUDA, etc.) in C++.

C++ Autograd impls are a parallel case handled by `migrate-autograd-fns-to-python`. The two skills are independent — order between them doesn't matter.

This step is run after `pybind-to-torch-library` (so ops are registered through `TORCH_LIBRARY`) and before `scaffold-stable-target` (so the C++ Meta translation units are gone before deciding which remaining files can be made stable).

## When to use

The migration plan flagged at least one file with the `meta-fns` concern, typically:

- `TORCH_LIBRARY_IMPL(libname, Meta, m)` blocks.
- `m.impl(...)` registrations whose target function performs shape/dtype inference only (no real compute, no real data access).
- Use of `Tensor::sym_sizes()`, `Tensor::sym_strides()`, `SymInt` arithmetic in C++.

## Inputs

- `plan_path`: path to the migration plan JSON, OR
- List of files with the `meta-fns` concern.

## Migration steps

### 1. Locate each C++ Meta impl

For each `TORCH_LIBRARY_IMPL(libname, Meta, m)` block, list the registered functions and their C++ signatures. Typical shape:

```cpp
// extension_cpp_stable/csrc/meta.cpp  (or similar)
Tensor mymuladd_meta(const Tensor& a, const Tensor& b, double c) {
    TORCH_CHECK(a.sym_sizes() == b.sym_sizes(), "shape mismatch");
    return at::empty_like(a);
}

TORCH_LIBRARY_IMPL(extension_cpp_stable, Meta, m) {
    m.impl("mymuladd", &mymuladd_meta);
}
```

### 2. Translate each Meta impl to a Python fake

Mirror the C++ logic in Python. Use `torch._check(...)` for the equivalent of `TORCH_CHECK`. Return an empty tensor with the correct shape, dtype, and device.

```python
# extension_cpp_stable/_meta.py  (or in __init__.py)
import torch

@torch.library.register_fake("extension_cpp_stable::mymuladd")
def _(a, b, c):
    torch._check(a.shape == b.shape)
    torch._check(a.dtype == torch.float)
    torch._check(b.dtype == torch.float)
    torch._check(a.device == b.device)
    return torch.empty_like(a)
```

Key translation patterns:

| C++ Meta | Python fake |
|---|---|
| `TORCH_CHECK(cond, msg)` | `torch._check(cond, lambda: msg)` |
| `at::empty_like(a)` | `torch.empty_like(a)` |
| `at::empty({M, N}, a.options())` | `a.new_empty([M, N])` |
| `a.sym_sizes()[i]` | `a.shape[i]` (works with SymInt automatically) |
| `a.scalar_type()` | `a.dtype` |
| `a.device()` | `a.device` |
| Custom shape arithmetic with `SymInt` | Plain Python operators — fake tensors carry `SymInt` transparently |

### 3. Register the fake at import time

The decorator must run before the op is called from fake/Meta contexts. Put the registration in a module that loads when the package is imported — usually `__init__.py` or a `_meta.py` imported from it:

```python
# __init__.py
import torch
from . import _meta  # noqa: F401  — registers fake kernels
torch.ops.load_library(...)  # see pybind-to-torch-library
```

### 4. Delete the C++ Meta impl

Once the Python fake is in place:

- Remove the `TORCH_LIBRARY_IMPL(..., Meta, m)` block.
- Remove the C++ Meta function definition (e.g., `mymuladd_meta`).
- Remove the `meta.cpp` translation unit from `setup.py` / `CMakeLists.txt` if no other code lives in it.

### 5. Update Python registrations for parameter names

Schema parameter names matter — they appear in error messages and downstream tooling. If the C++ schema used positional names like `a`, `b`, `c`, match those in the Python fake's signature.

## Critical rules

1. **Do not call the C++ op from inside the fake.** Fake kernels run in tracing contexts where the C++ kernel cannot execute. The fake must compute shape/dtype/device purely from input metadata.
2. **Never allocate real data in a fake.** Use `torch.empty_like`, `a.new_empty(...)`, or `torch.empty(...)` — these return fake tensors when called inside a fake context.
3. **Don't catch SymInt and convert to int.** Operations on `a.shape` work directly with `SymInt`. Calling `int(a.shape[0])` will break dynamic shapes.
4. **Register exactly once.** If the package can be imported multiple times (rare but possible in dev environments), guard with a module-level flag or rely on `register_fake`'s own idempotency.
5. **Keep the Python fake in sync with the C++ kernel's actual output shape.** If the C++ kernel ever reshapes or pads, the fake must mirror it.

## Verification

Run the project's existing test suite (`test_cmd` from the migration plan). If the project tests already cover the op through `torch.compile` or fake-tensor paths, that exercises the fake kernel.

A quick smoke test if you want to confirm the fake registered correctly without running the full suite:

```bash
python -c "
import torch, extension_cpp_stable
a = torch.empty(8, 8, device='meta')
b = torch.empty(8, 8, device='meta')
out = torch.ops.extension_cpp_stable.mymuladd(a, b, 0.5)
print(out.shape, out.dtype, out.device)
"
```

Expected: prints the correct shape on the meta device without crashing or running the real kernel.

## Handoff

The orchestrator will next dispatch `scaffold-stable-target`. With C++ Meta impls gone, the remaining C++ translation units no longer need access to the dispatcher's Meta machinery, which expands the set of files that can be made ABI-stable.
