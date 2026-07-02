---
name: migrate-autograd-fns-to-python
description: Move C++ Autograd-dispatch implementations (custom forward/backward kernels registered under the `Autograd` / `AutogradCUDA` dispatch key, or `torch::autograd::Function` subclasses) into Python via `torch.library.register_autograd`. Use when the user wants to remove C++ autograd impls from a PyTorch extension as part of an ABI-stable migration, or to add training support that didn't exist before.
allowed-tools: Read, Bash, Edit, Write, Grep
license: BSD-3-Clause
---

# Migrate C++ Autograd Functions to Python

PyTorch's stable ABI does not expose the C++ autograd machinery — `torch::autograd::Function`, `torch::autograd::AutogradContext`, `Variable`, and the associated `IValue` save/restore plumbing all live in libtorch-internal headers that fail to compile under `-DTORCH_TARGET_VERSION`. There is no `torch::stable::autograd::Function`. Writing autograd functions in Python is also the recommended path for modern C++ extensions today. Per the [official C++ custom-ops tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) — is to register the autograd kernel in **Python** using `torch.library.register_autograd`, while leaving the device forward kernels (CPU, CUDA, etc.) in C++.

This step is run after `pybind-to-torch-library` (so ops are registered through `TORCH_LIBRARY`) and before `scaffold-stable-target` (so the C++ Autograd translation units are gone before deciding which remaining files can be made stable). It is independent of `migrate-meta-fns-to-python` — order between the two doesn't matter.

## When to use

The migration plan flagged at least one file with the `autograd-fns` concern, typically:

- `TORCH_LIBRARY_IMPL(libname, Autograd, m)` or `..., AutogradCUDA, m)` blocks.
- C++ subclasses of `torch::autograd::Function` (with `forward` / `backward` static methods) registered for the op.
- Use of `torch::autograd::AutogradContext*`, `ctx->save_for_backward(...)`, `ctx->get_saved_variables()`, `ctx->needs_input_grad` in C++.

## Inputs

- `plan_path`: path to the migration plan JSON, OR
- A list of files with the `autograd-fns` concern.

## Migration steps

### 1. Locate each C++ Autograd impl

For each `TORCH_LIBRARY_IMPL(libname, Autograd, m)` block (or `AutogradCUDA`, etc.), find the `torch::autograd::Function` subclass it registers and list its `forward` / `backward` static methods. Typical shape:

```cpp
// extension_cpp/csrc/autograd.cpp  (or similar)
#include <torch/autograd.h>
#include <torch/library.h>

class MyMulAddFunction : public torch::autograd::Function<MyMulAddFunction> {
 public:
  static at::Tensor forward(
      torch::autograd::AutogradContext* ctx,
      const at::Tensor& a,
      const at::Tensor& b,
      double c) {
    ctx->save_for_backward({a, b});
    return a * b + c;
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::variable_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto a = saved[0];
    auto b = saved[1];
    auto grad = grad_outputs[0];
    return {grad * b, grad * a, at::Tensor()};  // at::Tensor() for non-differentiable scalar c
  }
};

at::Tensor mymuladd_autograd(const at::Tensor& a, const at::Tensor& b, double c) {
  return MyMulAddFunction::apply(a, b, c);
}

TORCH_LIBRARY_IMPL(extension_cpp, Autograd, m) {
    m.impl("mymuladd", &mymuladd_autograd);
}
```

### 2. Translate each Autograd impl to Python

Split the C++ `Function` into two Python callables: `_setup_context` (mirrors what `forward` saved into `ctx`) and `_backward` (mirrors the C++ `backward`). Register them via `torch.library.register_autograd`:

```python
# extension_cpp/_autograd.py  (or in __init__.py)
import torch

def _setup_context(ctx, inputs, output):
    a, b, c = inputs  # positional, matches schema order
    saved_a, saved_b = None, None
    if ctx.needs_input_grad[0]:
        saved_b = b
    if ctx.needs_input_grad[1]:
        saved_a = a
    ctx.save_for_backward(saved_a, saved_b)

def _backward(ctx, grad):
    a, b = ctx.saved_tensors
    grad_a, grad_b = None, None
    if ctx.needs_input_grad[0]:
        grad_a = grad * b
    if ctx.needs_input_grad[1]:
        grad_b = grad * a
    return grad_a, grad_b, None  # one value per forward input; None for non-tensor / non-differentiable

torch.library.register_autograd(
    "extension_cpp::mymuladd", _backward, setup_context=_setup_context,
)
```

Key translation patterns:

| C++ Autograd | Python equivalent |
|---|---|
| `class X : public torch::autograd::Function<X>` | Two free functions: `_setup_context(ctx, inputs, output)` and `_backward(ctx, *grads)` |
| `static at::Tensor forward(ctx, args...)` | The forward stays in C++ as the device kernel. `_setup_context` just records what `backward` will need. |
| `ctx->save_for_backward({a, b})` | `ctx.save_for_backward(a, b)` |
| `ctx->get_saved_variables()` | `ctx.saved_tensors` |
| `ctx->needs_input_grad[i]` | `ctx.needs_input_grad[i]` |
| `return {grad_a, grad_b, at::Tensor()}` | `return grad_a, grad_b, None` |
| `MyFn::apply(a, b, c)` from C++ wrapper | Not needed — the dispatcher routes Autograd-key calls to your Python callbacks automatically once registered |

### 3. Register the autograd at import time

The `register_autograd` call must run before the op is invoked from any autograd-tracing context. Put it in a module that loads when the package is imported:

```python
# __init__.py
import torch
torch.ops.load_library(...)  # see pybind-to-torch-library

from . import _autograd  # noqa: F401  — registers autograd
```

Order matters: load the C++ library first (so the op is defined in the dispatcher), then import the Python module that calls `register_autograd`.

### 4. Delete the C++ Autograd impl

Once the Python autograd is in place:

- Remove the `TORCH_LIBRARY_IMPL(..., Autograd, m)` block.
- Remove the `torch::autograd::Function` subclass and any C++ wrapper that called `::apply(...)`.
- Remove the autograd `.cpp` translation unit from `setup.py` / `CMakeLists.txt` if no other code lives in it.
- Remove `#include <torch/autograd.h>` from any remaining files.

 Run the project's existing tests for the ops that were migrated.

### 5. Update Python parameter names

`_setup_context` receives `inputs` as a tuple matching the schema's argument order. Use names that match the schema (`a, b, c` if that's how `m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor")` is declared) so the code reads cleanly.

## Critical rules

The Python version should match the C++ behavior. Optimization is a separate pass — don't conflate it with the migration.

1. **Save the same tensors the C++ version saved.** If the C++ `forward` did `ctx->save_for_backward({a, b})` unconditionally, the Python `_setup_context` should also save `a, b` unconditionally — even if `needs_input_grad` checks would let you skip some. Mirror the C++ first; tighten in a follow-up if desired.
2. **Return one gradient per forward input, in order, matching the C++ shape.** If C++ returned `{grad_a, grad_b, at::Tensor()}`, Python returns `grad_a, grad_b, None`. Same count, same positions, `None` wherever C++ returned an empty tensor.
3. **If the C++ backward called another custom op, the Python backward calls the same op via `torch.ops.<libname>.<op>(...)`.** Don't reimplement the math in Python — call the existing kernel.
4. **`_setup_context`'s `inputs` tuple matches the schema's argument order.** If the schema is `mymuladd(Tensor a, Tensor b, float c) -> Tensor`, unpack as `a, b, c = inputs`. Mismatched order silently corrupts gradients.
5. **Don't try to keep C++ autograd "for performance."** The forward kernel — where the real compute happens — stays in C++ under `STABLE_TORCH_LIBRARY_IMPL`. Only the autograd wiring (save tensors, compute grads from saved tensors) moves to Python, and that wiring is not a hot path.

## Verification

Run the project's existing test suite (`test_cmd` from the migration plan). Any test that exercises backward (typically `loss.backward()` or `torch.autograd.grad(...)`) covers the new registration.

If the project has no autograd tests at all, warn the user, but do run a quick smoke test:

```python
import torch
a = torch.randn(8, requires_grad=True)
b = torch.randn(8, requires_grad=True)
out = torch.ops.extension_cpp.mymuladd(a, b, 0.5).sum()
out.backward()
assert a.grad is not None and b.grad is not None
```

## Handoff

The orchestrator will next dispatch `scaffold-stable-target` (or `migrate-meta-fns-to-python` if it hasn't run yet — the two are independent). With C++ Autograd impls gone, the remaining C++ translation units no longer need `<torch/autograd.h>`, which expands the set of files that can be made ABI-stable.
