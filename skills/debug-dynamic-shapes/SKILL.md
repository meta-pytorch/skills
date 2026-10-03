---
name: debug-dynamic-shapes
description: Debug issues related to dynamic shapes in torch.compile.  Useful for errors related to dynamic shapes and symbolic values (int, float, bool).  Other symptoms of dynamic shape problems can include recompiles, poor runtime performance, and excessive compilation overhead.

---

# debug-dynamic-shapes

## Overview
Collect information using the torch-compile-logs skill.  Look at `TORCH_LOGS=+dynamic` output or the specialization section of `compilation_metrics*.html` in tlparse (`put_local_code_state_*.txt` also lists which dims are dynamic, but only exists when a job id is set; see torch-compile-logs).  Then follow the workflow described in the following sections: determine whether there is actually a problem related to dynamic shapes, and apply the corresponding fix

## Step 1: determine whether dynamic shapes are to blame

Logs often contain information about dynamic shapes even when the shapes are incidental to the real problem.  Classify by the guard that failed, not by whether symbolic shapes appear in the logs.  Find the failing guard in `recompile_reasons_*.json` (tlparse) or `TORCH_LOGS=recompiles`.

Shape-related guard failures:
- Size or stride mismatches, e.g. `tensor 'x' size mismatch at index 0. expected 3, actual 4` or `tensor 'x' stride mismatch at index 0. expected 4, actual 1`
- Symbolic-shape guards, which are reported as expressions over sizes, e.g. `2 <= x.size()[0]` or `(x.size()[0] % 8) == 0`

NOT shape-related, even if the inputs also changed size or the logs mention symbolic shapes:
- Changes in tensor dtype, rank, device, or `requires_grad`
- Guard failures on type, object id, or attribute/constant values (e.g. `n == 2`)

Errors or graph breaks that mention data-dependent expressions or unbacked symbols (`u0`) are shape-related too, but they happen during tracing rather than as recompiles.  See Fix 3.

Only the first failing guard is reported, even with `TORCH_LOGS=recompiles_verbose`.  If several inputs changed at once, the reported guard may not be the only cause: after fixing it, check whether a different guard failure shows up.

## Step 2: pick the appropriate fix

Which fix to use depends on whether the shapes are static or dynamic.  Check `dynamo_output_graph`: a dynamic dimension appears as a symbol in the tensor type (`f32[4, s77, 64]`) with a matching `Sym(s77)` placeholder; a static dimension is a literal (`f32[4, 16, 64]`).

### Fix 1: making static shapes dynamic

If changing shapes are causing excessive recompilations, try to mark the relevant dimensions as dynamic.  By default Dynamo will treat a dimension as static initially, then recompile and mark dynamic if it sees a different size.  See the below table for different options to make shapes dynamic.  Prefer `mark_dynamic()` if possible

| API | Behavior |
|---|---|
| Default | Initially specializes, then may automatically generalize varying dimensions after a mismatch. Dimensions of size 0 and 1 remain specialized. |
| `mark_dynamic` | Creates a backed symbolic dimension, optionally with bounds. Full specialization of the marked dimension raises a constraint violation, although guarded subregions such as modulo branches are allowed. |
| `maybe_mark_dynamic` | A weak backed-symbolic hint. Dynamo may specialize it without raising the strict `mark_dynamic` constraint error. |
| `mark_unbacked` | Creates a hintless symbol. Data-dependent relationships become runtime assertions or data-dependent-expression errors; unlike backed symbols, it can cover sizes 0 and 1 when the operations permit them. |
| `dynamic=True` | Attempts to symbolize eligible inputs immediately, but operations, control flow, and forced-static sources can still specialize dimensions. Sizes 0 and 1 remain specialized. |

The following flags can force shapes to be static - check them if a shape you expect to be dynamic isn't
- `torch._dynamo.config.force_parameter_static_shapes`
- `torch._dynamo.config.force_nn_module_property_static_shapes`

### Fix 2: dynamic but specialized
Find the specialization reason: see the specialization section of `compilation_metrics*.html`, or `TORCH_LOGS=+dynamic` lines like `eval Eq(s0, 33) [guard added]`.

Some common ways that otherwise dynamic shapes can become specialized:
- Control flow on size: rewrite if possible, but make sure to preserve behavior.  If the size is unbacked (`u0`), see Fix 3
- Triton `tl.constexpr` kernel args (which are always specialized): may be possible to rewrite the kernel
- PyTorch bugs

### Fix 3: data-dependent errors (unbacked symbols)
Unbacked symbols (`u0`, `u1`, ...) are sizes or values that depend on tensor data, so Dynamo has no example value to guard on and can't recompile its way around them.  They come from:
- `.item()` / `.tolist()` (when `torch._dynamo.config.capture_scalar_outputs = True`)
- ops whose output shape depends on data: `nonzero`, `masked_select`, boolean indexing `x[mask]`, `unique` (when `torch._dynamo.config.capture_dynamic_output_shape_ops = True`, or always under `fullgraph=True`)
- `torch._dynamo.decorators.mark_unbacked(t, dim)`

Branching or guarding on an unbacked value fails during tracing:
- With `fullgraph=True`: `torch._dynamo.exc.UserError: ... Could not guard on data-dependent expression u0 > 0 (unhinted: u0 > 0)`, followed by `Caused by: <line that guarded>`
- Without `fullgraph`: a graph break with reason `Data-dependent branching` (gb0170).  If the capture flag above is off, Dynamo instead graph-breaks at the op itself (`Dynamic shape operator`, gb0036)

To find where a symbol was created, run with `TORCH_LOGS=+dynamic` and look for `create_unbacked_symint u0 ...`, which is followed by the source line.

Fixes, in order of preference.  The compiled code must match eager for every input, including edge cases like empty masks or zero counts:
1. Rewrite to avoid the branch, e.g. replace `idx = torch.nonzero(mask)` + `if idx.shape[0] > 0: x[idx].mean()` with `torch.where(mask, x, 0).sum() / mask.sum().clamp(min=1)`
2. `torch.cond(pred, true_fn, false_fn, operands)` when both branches are genuinely needed
3. `torch._check(expr)` when `expr` always holds for valid inputs, e.g. `torch._check(idx.shape[0] > 0)`.  This inserts a runtime assertion: if the fact is ever false, the compiled code raises `RuntimeError: Runtime assertion failed for expression u0 >= 1`.  Confirm with the user that the assumption is valid.

Do not use `guard_or_false` / `guard_or_true` in user code, even though the error message suggests them.  When the expression can't be decided they silently pick one branch, so `if guard_or_false(idx.shape[0] > 0)` always takes the false branch and returns the wrong result.  They are meant for framework code where both branches are correct (e.g. a fast path vs a general path).
