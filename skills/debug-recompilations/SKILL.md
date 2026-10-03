---
name: debug-recompilations
description: Debug excessive recompilations when using torch.compile.  Use when the user complains about recompilations or long runtimes (which may be caused by recompilations)

---

# debug-recompilations

## Overview

When this skill is invoked, first determine whether recompilations are in fact the problem.  In some cases recompilations may be explicit (either the user has already done some debugging, or torch logs mention recompilations explicitly).  In other cases recompilations may silently cause performance issues.  Only proceed to phase 2 once recompilations have been established as an important part of the problem.

## Phase 1: Detect recompilations

Gather logs using the torch-compile-logs skill.

## Phase 2: Diagnose

The following subsections list common patterns and how to fix them (if possible).  If none are applicable, follow the general advice in the last subsection.

### Genuine recompilations
These recompilations are genuine in the sense that they represent real changes in the model.  They generally cannot be fixed by changing user code, but may have other workarounds.  Look for relevant config options to report back to the user if possible.

| Pattern                                                           | Why                                                                        |
|-------------------------------------------------------------------|----------------------------------------------------------------------------|
| `self._modules['X'].attr == N` where the int changes graph structure (loop count, branch, indexing, module selection) | each value produces a different graph |
| `KeyError on self._modules['X']` / `self.X`                       | different module structure across instances                                |
| `len(self._modules['X']._modules) == N` (ModuleList length)       | structural difference                                                      |
| `tensor 'X' rank mismatch`                                        | 2D vs 3D etc. genuinely need separate graphs                               |
| `tensor 'X' requires_grad mismatch`                               | the input tensor's `requires_grad` metadata changed.  Separate variants are expected if that change is intentional; otherwise keep the metadata consistent |
| `self.bool_attr == True/False`                                    | different boolean config across instances                                  |
| `not self.X.attr` (mutable-state truthiness)                      | one-time state transition; single unavoidable recompile                    |
| `X is None` / `is not None`, `len(kwargs) == N` / `not kwargs`    | genuine control-flow / signature differences                               |

### Fixable recompilations

These patterns indicate recompilations that can be fixed by modifying user code.  *Rule: a fix must not change model output or control flow.*  Use `torch.compiler.is_compiling()` to guard purely observational code (i.e. profiling/logging) if it is causing recompilations.

| Pattern | Description | Possible fix |
| ------- | ------------ | ----------- |
| `record_function(f"...{self.attr}")` | guarding on a non-const f-string | wrap in `if not torch.compiler.is_compiling():` or delete.  Confirm it's profiling/logging only |
| 0/1 specialization despite `mark_dynamic` | `mark_dynamic` adds a `size >= 2` guard; sizes 0/1 still specialize. | Use `mark_unbacked` (beware DDEs) or run rare sizes eager.
| `self._modules['X'].attr == N` where the int only feeds arithmetic or view/reshape sizes (e.g. `head_dim = d // self.n_heads`) | int attributes on nn.Modules are specialized by default, even when one graph could handle every value | list the attribute in `torch.compiler.config.dynamic_sources` (PyTorch 2.7+), using the guard's source name with the `L['...']` prefix restored, e.g. guard `block._modules['attn'].n_heads` → `L['block']._modules['attn'].n_heads` (regexes like `.*n_heads` also work; the attribute-style `L['block'].attn.n_heads` does not match).  Alternatively, `torch._dynamo.config.allow_unspec_int_on_nn_module = True` applies to all int module attributes; add `dynamic=True` to also avoid the first recompile.  Check the source line printed with the guard to see how the int is used |
| `DTensor.redistribute` signature mismatch | | align all call sites |
| `tensor 'X' size mismatch at index N` (e.g. 89318 vs 91504) | | see dynamic shapes (next section) |

### Dynamic-shape recompilations (size-mismatch reasons)
Use the debug-dynamic-shapes skill for recompilations related to changing shapes.

### General advice
#### Change the recompile limit
There is a limit to how many times a function can be recompiled, determined by `torch._dynamo.config.recompile_limit` and `torch._dynamo.config.accumulated_recompile_limit`.  Increase these limits if the number of recompilations is reasonably expected to be above the defaults, but do not increase them arbitrarily

#### Varying scalar arguments
Dynamo will usually mark Python ints/floats as dynamic if they change values on subsequent calls.  `torch.compiler.config.dynamic_sources` can be used to avoid the initial recompilation, if necessary.  Note that this does not work for values that are used in control flow, or that otherwise change graph topology.

#### Graph Breaking to Reduce Recompilation Costs
If a large graph is recompiling and causing high compile time, you can intentionally introduce a graph break using `torch._dynamo.graph_break()`.  This results in smaller graphs, reducing compiler overhead, but at the cost of a performance hit.  Inform the user.
