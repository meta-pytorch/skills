---
name: pytorch-general-qa
description: >-
  Answer general PyTorch usage questions covering autograd, numerical behavior,
  distributed training, memory management, and API semantics. Use when a user
  asks why PyTorch behaves unexpectedly, reports a suspected bug, or needs help
  diagnosing training issues.
license: BSD-3-Clause
metadata:
  author: test
  version: "0.1"
---

# PyTorch General Q&A

You are a PyTorch expert. When answering questions about unexpected PyTorch behavior, use the reference knowledge below to provide accurate, specific answers. Always identify the root cause before suggesting fixes.

## Numerical Precision & Floating-Point Behavior

### Extremal float32 values cause NaN in LayerNorm on CPU
When input values are at or near `torch.finfo(torch.float32).max` (~3.4e38), intermediate computations in LayerNorm (mean, variance) overflow in float32, producing NaN. This may happen on CPU but not CUDA because they use different internal implementations with different overflow characteristics. This is expected behavior per PyTorch's numerical accuracy documentation on extremal values. Fix: upcast to float64 or scale inputs to a normal range.

### Conv2d uses different CPU algorithms for different batch sizes
On CPU, PyTorch dispatches to different convolution implementations depending on batch size (e.g., oneDNN/MKLDNN for batch size > 1, native implementation for batch size 1). Different algorithms use different floating-point reduction orders, producing ~1e-6 differences. This is expected, not a bug. Use `torch.allclose` with appropriate tolerance for comparisons.

### Different memory layouts dispatch to different GPU kernels
When using `torch.channels_last` vs contiguous memory format, PyTorch dispatches to different GPU kernels (different cuBLAS/cuDNN algorithms). Because floating-point arithmetic is not associative, different reduction orders produce slightly different results. These differences compound over training iterations. PyTorch makes no guarantees about numerical consistency across memory layouts.

### Changing thread count changes Linear results
`torch.set_num_threads` changes how MKL/BLAS parallelizes matrix multiplication. Different thread counts mean different partition boundaries and different floating-point reduction orders. Because FP addition is not associative, `(a + b) + c ≠ a + (b + c)` in general, so different thread counts produce slightly different results (~1e-8). This is expected behavior. Bit-exact reproducibility requires identical hardware, CUDA flags, library versions, thread counts, etc.

### CPU denormal floats cause progressive slowdown with weight decay
When using Adam (or other optimizers) with weight_decay on CPU, weights are continuously shrunk toward zero. When they enter the denormal (subnormal) float range (~1e-38 and below), CPU arithmetic becomes 10-100x slower because most hardware handles denormals via slow microcode emulation. This looks like a memory leak but is actually a CPU hardware performance issue. Fix: call `torch.set_flush_denormal(True)` to flush denormals to zero.

## Autograd & Computation Graph

### Accumulating loss tensors retains the computation graph
Adding a loss tensor (which has `grad_fn`) to a running sum (e.g., `epoch_loss += loss`) keeps the entire computation graph alive in memory, because Python holds a reference to the tensor with its graph. Memory grows ~3MB per step. Fix: use `epoch_loss += loss.item()` to extract the scalar value before accumulating.

### Multiplying a requires_grad tensor makes it non-leaf
`torch.randn(..., requires_grad=True) * 0.01` creates a non-leaf tensor — the result of a computation. Only leaf tensors (not results of operations) can be optimizer parameters. Fix: create the tensor first, then multiply separately: `t = torch.randn(..., requires_grad=True); t.data *= 0.01`.

### torch.no_grad() blocks gradient flow for distributions
If you construct a distribution and compute `log_prob` inside `torch.no_grad()`, the result will have `requires_grad=False`. This affects all distributions, not just Categorical. The Normal example in common comparisons appears to work because it's constructed outside `no_grad()`. Fix: move distribution construction and `log_prob` outside the `no_grad()` block.

### register_full_backward_hook conflicts with inplace ReLU
`register_full_backward_hook` internally attaches a custom autograd Function that returns a **view** of the module's output. When the next module uses `ReLU(inplace=True)`, the in-place operation on this view is forbidden by autograd (it would override the custom backward). The error fires during **forward**, not backward, which is confusing. The legacy `register_backward_hook` works because it doesn't create views, but may give incorrect `grad_input`. Fix: use a forward hook + tensor hook pattern:
```python
def tensor_hook(grad): print(grad.size())
def fw_hook(mod, input, output):
    output.register_hook(tensor_hook)
mod[2].register_forward_hook(fw_hook)
```

### Activation checkpointing with use_reentrant=True fires hooks twice for shared parameters
When using `torch.utils.checkpoint.checkpoint` with the default `use_reentrant=True`, the backward pass uses a custom autograd Function whose backward triggers a **nested** `.backward()` call. For shared parameters, the outer backward doesn't see the dependency through the checkpoint boundary, so the gradient accumulation hook fires once in the outer backward and once in the inner (reentrant) backward. Fix: use `checkpoint(..., use_reentrant=False)`, which uses `saved_tensors_hooks` instead of a nested backward, keeping the full dependency graph visible in a single backward pass.

### Embedding padding_idx: forward-mode vs reverse-mode Jacobians differ
`padding_idx` marks a row of the embedding weight as constant. Reverse-mode backward explicitly zeros the gradient at that row. Forward-mode AD computes the raw JVP without this masking. The Jacobians will therefore disagree at the padding_idx row. This is expected behavior and does not affect training — the padding row is conceptually constant. Using `nn.Embedding` (which initializes the padding row to zero) avoids confusion.

## Memory & Serialization

### torch.save saves the entire underlying storage for tensor views
When tensors are views of a larger storage (common after slicing or in state_dicts from certain frameworks), `torch.save` serializes the full underlying storage, not just the viewed portion. This causes files to be much larger than expected. `.cpu()` or `.clone()` creates an independent contiguous copy, breaking the view relationship and producing correctly-sized files. There is no information loss.

### Batched matmul with broadcasting can allocate huge temporary memory
When doing `a @ b.mT` with broadcasting on 4D+ tensors, PyTorch's matmul internally lowers to `bmm`. It `expand`s the broadcasted operand (creating stride-0 virtual dimensions), then `reshape`s to 3D. The reshape cannot be done as a view on stride-0 dimensions, so it **materializes** the full expanded tensor. For large batch dims, this can allocate gigabytes of temporary memory. Fix: use `torch.einsum('...ik,...jk', a, b)` instead, which uses `sumproduct_pair` internally and avoids materialization by passing stride-0 dimensions directly to cuBLAS.

## nn.Module Behavior

### Bound method assignment creates reference cycles preventing GC
Assigning a bound method to an instance attribute (e.g., `self.which_foo = self.foo`) creates a reference cycle: the method object holds a reference to `self`, and `self` holds a reference to the method. Python's garbage collector can break reference cycles, but only when it runs. If the module has a `register_buffer` with a CUDA tensor, the tensor won't be freed until GC runs, making the leak visible in `torch.cuda.memory_allocated()`. Fix: call `gc.collect()`, avoid storing bound methods, or use `weakref`.

### BatchNorm with track_running_stats=False always computes batch statistics
When `track_running_stats=False`, `running_mean` and `running_var` are `None`. In eval mode, BatchNorm normally uses running statistics, but when they don't exist, it falls back to computing batch statistics — which requires batch size > 1. This is expected behavior, not a bug. Fix: use `track_running_stats=True` (default), or use InstanceNorm/GroupNorm for batch-size-1 scenarios.

### CrossEntropyLoss returns NaN when all targets are ignore_index
With `reduction='mean'`, the loss divides the sum (0) by the count of non-ignored elements (0), producing 0/0 = NaN. With `reduction='sum'`, the result is 0. Fix: check for all-ignored batches before computing loss, or use `reduction='sum'` with manual averaging.

## Random Number Generation

### Sampling one-by-one gives different results than batch sampling
PyTorch's RNG generates random numbers in blocks (e.g., blocks of 4 via Philox). When sampling fewer elements than the block size, unused values are discarded. So sampling 1 element 32 times consumes different random numbers than sampling 32 at once. This is expected. Fix: generate all needed random numbers in a single call.

### torch.multinomial CUDA RNG advancement depends on GPU architecture
The CUDA Philox RNG advances its counter based on the GPU's grid configuration (number of SMs/thread blocks). Different GPUs have different SM counts, so `torch.multinomial` on CUDA advances the RNG by different amounts for the same tensor. This means identical seeds produce different samples across GPU models. CPU is unaffected because it uses a single sequential RNG. Multinomial without replacement uses a different code path (`exponential_` for Gumbel-max sampling) that is particularly sensitive to this. Fix: perform sampling on CPU for cross-GPU reproducibility.

## Distributions

### Beta distribution entropy can be negative
Differential entropy (for continuous distributions) CAN be negative, unlike discrete Shannon entropy which is always ≥ 0. For Beta(α, β) with α, β close to 1, the distribution is more concentrated than the uniform distribution, giving negative entropy. PyTorch's `Beta.entropy()` is correct.

## Distributed Training

### DDP + GradScaler: different scale factors cause weight divergence
DDP's gradient all-reduce happens during `backward()`, **before** `GradScaler.step()` unscales the gradients. If different ranks have different scale factors, the scaled gradients being all-reduced are inconsistent — each rank contributed gradients scaled by a different factor. Fix: ensure all ranks use the same `GradScaler` scale factor (set the same `init_scale`, or synchronize scales across ranks).

### FSDP with mixed frozen/trainable parameters (LoRA/PEFT)
FSDP's flat parameter design concatenates all parameters in a wrapped module into a single `FlatParameter`. A single parameter can only be `requires_grad=True` or `False` — not mixed. `transformer_auto_wrap_policy` wraps entire modules (e.g., Attention, FFN) that may contain both frozen and trainable parameters. Fix: use `lambda_auto_wrap_policy` to wrap trainable and frozen submodules in separate FSDP units. Note: `use_orig_params=True` support for non-uniform `requires_grad` was added in PyTorch ≥ 2.2. Even with this, gradient memory is allocated for the full flat buffer — separating frozen/trainable into different FSDP units is needed to optimize memory.

## API Semantics

### torch.unique sorted=False does not preserve insertion order
`torch.unique` internally sorts for algorithmic efficiency regardless of the `sorted` parameter. `sorted=False` means the output order is **unspecified**, not that insertion order is preserved. This is similar to NumPy's behavior. For order-preserving uniqueness, use `torch.unique_consecutive` on a pre-sorted input.

## DataLoader & Multiprocessing

### DataLoader freezes on Windows without __main__ guard
Windows uses the `spawn` multiprocessing start method (not `fork`), which re-imports the main module to start each worker process. Without `if __name__ == '__main__':`, the DataLoader creation runs again in each worker, causing recursive spawning and hangs. Fix: wrap all DataLoader usage in `if __name__ == '__main__':`.

### torch.median returns different indices on CPU vs GPU for tied values
When multiple elements have the same value, the index returned by `torch.median` is implementation-defined and may differ between CPU and GPU backends. Both results are valid. The tie-breaking behavior is not guaranteed to be consistent across devices.
