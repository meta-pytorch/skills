---
name: higher-order-ops
description: Use PyTorch higher-order ops (HOPs) to work around torch.compile problems — cond, switch, while_loop, map, scan, associative_scan, nested_compile_region, and the compile-safe print. Reach for these when data-dependent control flow or a loop graph-breaks / unrolls / recompiles under torch.compile or torch.export, or when repeated blocks make compilation slow.
---

# higher-order-ops

## Overview

A higher-order op (HOP) is an operator that takes a **function** (captured as a traced
subgraph) as an argument. HOPs let `torch.compile` / `torch.export` capture a *pattern* —
a branch, a loop, a repeated block — as a single structured node **with the body still
traced and compiled**, instead of graph-breaking on it, unrolling it into a giant graph,
or specializing (recompiling) on every shape/value. That makes them a targeted tool for
several common `torch.compile` problems.

Contrast with a **custom op** (see the custom-ops skill): a custom op makes a region
*opaque* — it runs eager, nothing inside is traced or compiled. A HOP is the opposite:
the region stays **traced, compiled, and (often) differentiable**, but its *structure*
(branches / iteration) is preserved. Reach for a HOP when the thing you're wrapping is
ordinary torch ops arranged as control flow, and you want compile to keep and optimize
that structure rather than break on it or unroll it.

Caveats that shape when you can use them: branch/body/combine functions must be **pure**
(no side effects, no in-place mutation of inputs or captured tensors); most are
**prototype**, and several (`map`, `associative_scan`, `while_loop`) have **limited or no
autograd** — check before putting one in a training path.

## Structured control flow

**Problem they fix:** Python control flow that depends on **tensor data** graph-breaks
(or raises `GuardOnDataDependentSymNode`), and Python loops get **unrolled** — baking the
trip count into the graph (recompiles when it changes) and blowing up graph size and
compile time. These HOPs capture the structure as one compact, shape-general node.

| HOP | Replaces | Reach for it when |
|---|---|---|
| `torch.cond(pred, true_fn, false_fn, operands)` | `if/else` on tensor data | The branch condition is data-dependent (`if x.sum() > 0:`) and graph-breaks or raises a data-dependent error. Both branches are traced, so both must be valid and return matching structure. |
| `torch._higher_order_ops.switch(index, branches, operands)` (newer PyTorch) | `if/elif` chain / index dispatch | N-way branch selected by an integer/tensor index, without writing (and tracing) an exponential `cond` tree. |
| `torch.while_loop(cond_fn, body_fn, carried)` | `while` with a data-dependent condition | The number of iterations depends on data (converge-until, variable steps) and you don't want the loop unrolled or specialized to a fixed count. |
| `torch._higher_order_ops.map(fn, xs, *args)` | `for` over the leading dim | Apply the same fn to each slice of a batch independently, without unrolling into N copies in the graph. |
| `torch._higher_order_ops.scan(combine_fn, init, xs)` | `for` with **carried state** | A sequential recurrence (RNN cell, running state, cumulative transform) where each step feeds the next. Keeps the graph small vs. unrolling the timesteps. |
| `torch._higher_order_ops.associative_scan(combine_fn, xs)` | cumulative op with an **associative** combine | Prefix-sum-shaped computations (cumsum/cumprod-like) where the combine is associative — parallelizable, so faster than `scan` as well as avoiding the unroll. |

### Illustrative example

A data-dependent `if` graph-breaks under `fullgraph=True`:

```python
def forward(self, x):
    if x.sum() > 0:          # data-dependent -> graph break
        return x * 2
    return x + 1
```

`torch.cond` captures both branches as one node — the branch logic goes in functions,
the operands in a tuple:

```python
def forward(self, x):
    return torch.cond(x.sum() > 0, lambda x: x * 2, lambda x: x + 1, (x,))
```

The other control-flow HOPs take the same shape (body/combine as functions, operands as a
tuple), and their bodies must be pure — e.g. return `x.clone()` rather than a carried input,
since HOPs reject input/output aliasing.

## Reducing compile time for repeated blocks

**Problem it fixes:** a block repeated many times with identical structure (the
transformer layers of an LLM) is compiled from scratch at each occurrence, so compile
time scales with depth.

| HOP | Reach for it when |
|---|---|
| `torch.compiler.nested_compile_region(fn)` (a.k.a. `invoke_subgraph`) | You want the region compiled **once** and the compiled artifact stamped out on every reuse ("hierarchical compilation"), substantially cutting compile time for deep, structurally-identical stacks. Safe: it transparently recompiles if inputs change; a no-op in eager. |

## Compile-safe print

**Problem it fixes:** a plain `print()` inside a compiled region graph-breaks.

- `torch._higher_order_ops.print(fmt, *args)` — a `print` that stays in the graph (Python
  format-string syntax). See the debug-graph-breaks skill for the full set of logging fixes.

## Choosing between HOP / custom op / graph break

- **Graph break (or `torch.compiler.disable`)** — give up compiling the region; it runs
  eager and splits the graph. Simplest, but you lose fusion across the boundary.
- **Custom op** — make a region *opaque*: native/non-traceable code runs eager as one
  node, no autograd unless you register it (custom-ops skill).
- **HOP** — keep the region *compiled and structured*. Use when the body is traceable
  torch ops shaped as control flow / a loop, and you want compile to preserve and optimize
  that structure (no unroll, no break, one shape-general graph).

## Other HOPs (for completeness)

These are real HOPs you may see, but they are specialized features or compiler internals
rather than general-purpose fixes for a `torch.compile` problem:

- **`torch.nn.attention.flex_attention.flex_attention(q, k, v, score_mod=..., block_mask=...)`**
  — express an attention variant (causal, ALiBi/relative bias, sliding-window, soft-capping,
  custom masking) as a `score_mod` / sparse `block_mask` and get a fused, memory-efficient
  kernel with no hand-written CUDA. A powerful feature (and can rescue custom-attention code
  that OOMs or breaks under compile), but primarily a specialized kernel, not a workaround.
- **`torch._higher_order_ops.foreach_map(fn, *lists)`** — horizontally fuse an elementwise op
  across a list of tensors (optimizer / `_foreach_` style). An optimization; it also avoids
  unrolling a per-tensor Python loop.
- **`torch._higher_order_ops.out_dtype(op, out_dtype, *args)`** — run an op at a specific
  output/accumulation dtype (e.g. integer matmul, wider accumulate). A mixed-precision
  representation primitive.
- **`torch.distributed._tensor.experimental.local_map(fn, out_placements, in_placements)`** —
  run a region on the **local shards** of `DTensor`s with explicit placements. A distributed
  tool, not a compile fix.
- **`torch._higher_order_ops.hints_wrap`** — attach backend-compiler hints to a traced region.
- **`run_const_graph`** — internal: carries a constant (weights-only) subgraph so it's folded
  and computed once (inference freezing / constant folding). Produced by compiler passes; you
  never call it directly.

## Reference docs
- Control-flow HOP index (`cond`, `while_loop`, `scan`, `associative_scan`, `map`, `switch`): https://docs.pytorch.org/docs/main/higher_order_ops/index.html
- FlexAttention — blog: https://pytorch.org/blog/flexattention/ ; paper: https://arxiv.org/abs/2412.05496
- `nested_compile_region`: `torch.compiler` docs (hierarchical compilation).
