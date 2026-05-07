---
name: abi-stable
description: >-
  Migrate C++/CUDA PyTorch extension code to use the PyTorch stable ABI.
  Use when the user wants to make their project ABI stable, migrate to
  stable ABI, or use torch::stable APIs.
license: BSD-3-Clause
---

# PyTorch Stable ABI Migration

Migrate C++/CUDA PyTorch extension code to the stable ABI so that extensions built
against one PyTorch version continue to work with future versions without recompilation.

## Prerequisites

**PyTorch 2.11+ is required.** Before making any changes, verify the installed version:

```bash
python3 -c "import torch; print(torch.__version__)"
```

If the version is below 2.11, stop and tell the user they need to upgrade PyTorch first.
The stable ABI headers (`torch/csrc/stable/`) were introduced in PyTorch 2.7 but the
API surface needed for most migrations was not complete until 2.11.

## Reference Documentation

Before starting a migration, read the project's reference documentation:

- **Official guide**: https://docs.pytorch.org/docs/stable/notes/libtorch_stable_abi.html
- **C++ API reference**: https://docs.pytorch.org/cppdocs/stable.html

For a concrete before/after example of a real migration, see
[references/migration-example.md](references/migration-example.md).

For the complete API mapping (old API -> stable ABI replacement), see
[references/api-mapping.md](references/api-mapping.md).

## Migration Workflow

### Step 1: Identify files to migrate

Find all `.cpp`, `.cu`, and `.cuh` files that include PyTorch headers like:
- `<torch/extension.h>`
- `<ATen/ATen.h>`
- `<ATen/cuda/CUDAContext.h>`
- `<c10/cuda/CUDAGuard.h>`
- `<torch/library.h>`

### Step 2: Update includes

Remove old includes and replace with stable ABI headers:

```cpp
// REMOVE these:
// #include <torch/extension.h>
// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <c10/cuda/CUDAGuard.h>
// #include <torch/library.h>

// ADD these (include only what you need):
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>

#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

// For CUDA stream access:
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <cuda_runtime.h>
```

### Step 3: Apply API replacements

Apply the replacements described in [references/api-mapping.md](references/api-mapping.md).
The most common ones are:

| Old API | Stable ABI Replacement |
|---------|----------------------|
| `at::Tensor` / `torch::Tensor` | `torch::stable::Tensor` |
| `TORCH_CHECK(...)` | `STD_TORCH_CHECK(...)` |
| `c10::cuda::CUDAGuard` | `torch::stable::accelerator::DeviceGuard` |
| `at::empty({...}, options)` | `torch::stable::new_empty(ref_tensor, {sizes}, dtype)` |
| `at::zeros({...}, options)` | `torch::stable::new_zeros(ref_tensor, {sizes}, dtype)` |
| `TORCH_LIBRARY_IMPL` | `STABLE_TORCH_LIBRARY_IMPL` |
| `m.impl("name", &func)` | `m.impl("name", TORCH_BOX(&func))` |

### Step 4: Update build configuration

In the project's `setup.py` (or CMakeLists.txt), add `-DUSE_CUDA` to **both** cxx
and nvcc compiler flags if using CUDA stream access via `aoti_torch_get_current_cuda_stream`.

### Step 5: Build and iterate

Build the project and fix compilation errors iteratively. Common issues are
documented in [references/common-issues.md](references/common-issues.md).

## Critical Rules

These are mistakes that AI models commonly make. Follow these strictly:

1. **Do NOT use `torch::stable::new_empty_strided`** - this API does not exist.
   To create a strided tensor, create a contiguous tensor with transposed dimensions
   and then call `torch::stable::transpose()`.

2. **Do NOT define a dummy `_C` module** that can be accessed from Python.

3. **Do NOT declare `aoti_torch_get_current_cuda_stream` yourself.** Include it from
   `torch/csrc/inductor/aoti_torch/c/shim.h`.

4. **Do NOT manually box kernels.** Use `TORCH_BOX(&func)` in `m.impl()` calls.

5. **Do NOT change switch statements into if/else blocks.** The stable ABI works
   fine with switch statements.

6. **When replacing `TORCH_CHECK` with `STD_TORCH_CHECK`, only replace the function
   name.** Do not change the content/arguments of the check.

7. **When replacing `tensor.data_ptr<T>()`**, use `tensor.const_data_ptr<T>()` for
   read-only access. Do not change the template type.

8. **For `DeviceGuard`**, pass `tensor.get_device_index()` not `tensor.device()`.

9. **For CUDA streams**, get the device index from a tensor
   (`tensor.get_device_index()`), not from
   `torch::stable::accelerator::getCurrentDeviceIndex()`.

10. **Scalar type enums** use `torch::headeronly::ScalarType::Float` (not
    `torch::kFloat32`), `torch::headeronly::ScalarType::Half` (not `torch::kFloat16`),
    etc.
