---
name: custom-ops
description: Author and register custom PyTorch operators with torch.library.  Use when wrapping a third-party library, a hand-written kernel (C++/CUDA/Triton), or any non-traceable Python function so it works with torch.compile / torch.export / autograd, or when you need to stop torch.compile from tracing into a function.
---

# custom-ops

## Overview

A custom op wraps an arbitrary function into a first-class PyTorch operator with a
stable name (`"namespace::op_name"`), a schema, and a mutation/aliasing contract.
Once registered, `torch.compile`, `torch.export`, and FX treat it as a single
**opaque, well-typed node**: they do not trace into the body, so the internals can
be anything (numpy, a library call, a compiled kernel), while the surrounding model
still compiles. With a fake kernel the op participates in shape inference, and with
an autograd registration it participates in training.

**When NOT to use one:** if the operation can be written as a composition of
existing PyTorch ops, do that instead — it will be faster (Inductor can fuse it),
differentiable for free, and less code. Reach for a custom op only when you need
opacity or are wrapping something Inductor cannot generate.

## When to apply a custom op

A custom op does one thing: it creates an **opaque, atomic operator boundary with an
implementation you supply** — compile treats the body as a single node it cannot see
into. There are two reasons to want that.

### 1. Supply an implementation compile can't generate (native / non-traceable code)
The body is something Dynamo can't trace or Inductor can't codegen, so without the op
you'd hit a graph break. The custom op makes it one opaque node backed by your code.
- **Third-party / no-torch-equivalent calls** — `scipy.special`, a C/Fortran library.
  (Common numpy already works: `torch.compile` rewrites supported numpy ops to torch via
  its numpy tracing, so wrap only what has no torch equivalent.)
- **Hand-written kernels** — custom CUDA/C++, a Triton kernel, an fbgemm-style op.
  Inductor calls the kernel at runtime instead of trying to regenerate it.

### 2. Suppress a compile transformation on otherwise-traceable code
The body *is* traceable torch ops, but you want the boundary to stop compile from
decomposing / fusing / reordering / regenerating it. Useful when that transformation is
unwanted because of:
- **Numerics** — a hand-stabilized `softmax`/`logsumexp`, or code whose dtype or reduction
  order must be preserved; fusion can change rounding.
- **An Inductor miscompile / bug** — the subgraph compiles to wrong results or crashes; the
  op forces an eager fallback (often temporary, until it's fixed upstream). It can likewise
  force a boundary when Inductor's fusion/decomposition is slower than the intact op.
- **Compile time / graph size** — a huge or pathological subgraph that Inductor is slow to
  lower/fuse/autotune; as an opaque op it's skipped by codegen and runs eager.

### Relationship to the Dynamo graph-break decorators
If you only need to make a function opaque — no backward formula, no need to survive
`torch.export` — the lighter-weight options from the graph-break workflow may be enough:
`@torch._dynamo.decorators.leaf_function` (opaque to Dynamo *and* AOTAutograd, needs
a `register_fake`), `@torch._dynamo.decorators.nonstrict_trace` (opaque to Dynamo but
AOTAutograd still traces the body), or `torch.compiler.disable()` (runs eager, causes
a graph break). Prefer a **custom op** when the op needs a real autograd formula —
i.e. its forward isn't differentiable by PyTorch (native code, a custom kernel), so
training requires a hand-written backward via `register_autograd` (a custom op trained
without one errors) — or when it must survive `torch.export`/AOTInductor: a custom op
stays a resolvable `namespace::name` node in the exported program, whereas the decorators
lower to a non-portable `invoke_leaf_function` wrapper that embeds a Python callable.

## Basic recipe (functional op)

The function **must have type hints** — they are used to infer the schema. `mutates_args`
is required.

```python
import torch
from torch import Tensor
import numpy as np

@torch.library.custom_op("mylib::numpy_sin", mutates_args=())
def numpy_sin(x: Tensor) -> Tensor:
    x_np = x.cpu().numpy()          # arbitrary non-traceable code
    y_np = np.sin(x_np)
    return torch.from_numpy(y_np).to(device=x.device)
```

- **Name:** `"{namespace}::{name}"`. Use your project as the namespace to avoid
  collisions (e.g. `"myproject::..."`).
- **`mutates_args`:** the names of arguments the op mutates in place — `()` for a pure
  function. Must be accurate; `"unknown"` pessimistically assumes everything is mutated.
- **Supported input types:** scalars (`int/float/bool/str/Number`), `Tensor`,
  `dtype`/`device`, flat `list[...]` of one type, and `Optional[...]` of the above.
  **Returns:** `Tensor`, `list[Tensor]`, or a scalar.
- **Schema:** inferred from the type hints. Only pass `schema=` if you have a specific
  reason; hand-written schemas are error-prone.

## Make it compile: register a fake kernel

`torch.compile`/`export` run the op on fake (data-less) tensors during tracing, so you
must describe the **output metadata** (shape, dtype, device, layout, strides) given the
input metadata. Allocate with `new_empty` / `empty_like` — never touch real data.

```python
@numpy_sin.register_fake
def _(x):
    return torch.empty_like(x)
```

**Data-dependent output shape** (the output size depends on values, as in `nonzero`):
allocate a fresh symbolic size with the ctx object.

```python
@torch.library.custom_op("mylib::nonzero", mutates_args=())
def nonzero(x: Tensor) -> Tensor:
    res = np.stack(np.nonzero(x.cpu().numpy()), axis=1)
    return torch.tensor(res, device=x.device)

@nonzero.register_fake
def _(x):
    ctx = torch.library.get_ctx()
    nnz = ctx.new_dynamic_size()        # unbacked symint for the data-dependent dim
    return x.new_empty([nnz, x.dim()], dtype=torch.int64)
```

The fake metadata **must match** what the real kernel produces, or you get silent
downstream errors.

## Add training support: register an autograd formula

**A custom op does not make non-differentiable internals differentiable** —
`register_autograd` requires you to supply the backward yourself.

Provide a `backward` and, if backward needs forward values, a `setup_context` to save
them. `setup_context(ctx, inputs, output)` runs in the forward; `backward(ctx, *grads)`
returns one gradient per input (`None` for non-differentiable inputs). `grads` has one
entry per op output.

```python
def _setup_context(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(x)

def _backward(ctx, grad):
    (x,) = ctx.saved_tensors
    return grad * x.cos()

numpy_sin.register_autograd(_backward, setup_context=_setup_context)
```

Both functions **must be traceable**: no `.data_ptr()` access, no dependence on or
mutation of global state. If the backward itself is non-traceable, wrap *it* in another
custom op and call that from `_backward`.

## Device-specific and in-place variants

- **Per-device kernels:** register the default with `custom_op(...)`, then add others
  with `@op.register_kernel("cuda")`. Or restrict a single definition with
  `custom_op(..., device_types="cpu")`.
- **Mutating op:** list every mutated arg in `mutates_args` and (by convention) return
  `None`.
  ```python
  @torch.library.custom_op("mylib::sin_", mutates_args={"x"}, device_types="cpu")
  def sin_(x: Tensor) -> None:
      np.sin(x.numpy(), out=x.numpy())
  ```
- **Factory functions** (no tensor inputs) that are device-specific must take a
  `device: torch.device` argument so the dispatcher knows where to run.

## Rules and common gotchas

- **No aliasing.** An op must never return an input, a view of an input, or two outputs
  that alias each other. Return fresh tensors.
- **Mutation must be declared.** Mutating an argument not listed in `mutates_args` is
  undefined behavior. Don't build "maybe writes to `out=`" ops — split into a functional
  op (allocates) and a separate mutating op.
- **Fake kernel is metadata-only.** It must not read tensor data; it only computes
  output shapes/dtypes/devices/strides.
- **No hidden side effects.** The compiler assumes an op is side-effect-free unless it
  declares mutation, so it may reorder or DCE the call. Keep IO/logging out of the body
  (or declare mutation for genuine in-place writes).
- **Type hints are mandatory** and drive the schema.
- **C++/CUDA kernels:** you can register directly from C++ with the `TORCH_LIBRARY`
  API, or wrap the compiled function in a Python `custom_op` as above. The fake-kernel
  and autograd rules are identical.

## Validate with opcheck

`torch.library.opcheck` checks the registration contract — schema vs. implementation,
mutation/aliasing, fake-kernel presence/correctness, and autograd registration — across
the sample inputs you give it. Include several cases: different devices and dtypes, edge
shapes (empty, size-1), and `requires_grad=True` inputs.

```python
x = torch.randn(3, requires_grad=True)
torch.library.opcheck(numpy_sin, (x,))
```

`opcheck` is **not** a numerical-correctness test. Separately verify values with
`torch.testing.assert_close` (vs. an eager reference) and gradients with
`torch.autograd.gradcheck`.

## Reference docs
- Python custom ops tutorial: https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html
- Custom ops landing page (choosing an approach): https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html
- `torch.library` API: https://docs.pytorch.org/docs/stable/library.html
