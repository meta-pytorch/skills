# Common Issues During Stable ABI Migration

## Build Errors

### `STD_CUDA_KERNEL_LAUNCH_CHECK` undefined

Missing include. Add the header that provides this macro. Check which header defines
it in your PyTorch version — it may be in `torch/headeronly/util/Exception.h` or a
CUDA-specific stable header.

### `aoti_torch_get_current_cuda_stream` undeclared

Do NOT forward-declare this function. Include it from:

```cpp
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
```

Also add `-DUSE_CUDA` to both `cxx` and `nvcc` compiler flags in your `setup.py`:

```python
extra_compile_args={
    "cxx": ["-DUSE_CUDA", ...],
    "nvcc": ["-DUSE_CUDA", ...],
},
```

### `cudaStream_t` not found

Include CUDA runtime:

```cpp
#include <cuda_runtime.h>
```

### `CUDA_VERSION` not defined

Include `cuda.h` before checking `#if defined(CUDA_VERSION)`:

```cpp
#include <cuda.h>
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12040
// ...
#endif
```

### Layout check fails to compile

The old pattern:

```cpp
TORCH_CHECK(tensor.layout() == at::Layout::Strided, ...);
```

Does not have a direct stable ABI replacement via `tensor.layout()`. Use:

```cpp
using torch::headeronly::Layout;
int32_t layout_val;
TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(tensor.get(), layout_val));
auto layout = torch::stable::detail::to<Layout>(
    torch::stable::detail::from(layout_val));
STD_TORCH_CHECK(layout == Layout::Strided, ...);
```

Save the converted layout to a variable to reduce duplication when checking
multiple tensors.

### `torch::stable::new_empty_strided` not found

This API does not exist. Create a contiguous tensor with transposed dimensions
and use `torch::stable::transpose()`:

```cpp
// Want shape {M, N} with strides {1, M} (column-major)
auto tmp = torch::stable::new_empty(ref, {N, M}, dtype);
auto result = torch::stable::transpose(tmp, 0, 1);
```

## Runtime Errors

### Tensor shape mismatch after migration

When replacing `at::empty_strided` with the create-and-transpose pattern, verify
that the transposed dimensions produce the correct final shape and strides.

For 3D column-major tensors like `{E, N, K}` with strides `{N*K, 1, N}`:

```cpp
// Create {E, K, N} then transpose dims 1 and 2
auto tmp = torch::stable::new_empty(ref, {E, K, N}, dtype);
auto result = torch::stable::transpose(tmp, 1, 2);
// result has shape {E, N, K} with strides {K*N, K, 1}...
// Wait — that gives {K*N, K, 1}, not {N*K, 1, N}.
```

Be careful: transposing only swaps two dimensions. For complex stride patterns,
you may need multiple transposes or a different dimension ordering. Always verify
the resulting strides match the original.

### Wrong CUDA stream device

When calling `aoti_torch_get_current_cuda_stream`, always get the device index
from the actual tensor being operated on:

```cpp
// CORRECT
aoti_torch_get_current_cuda_stream(input.get_device_index(), &stream_ptr);

// WRONG — may get stream for wrong device in multi-GPU scenarios
aoti_torch_get_current_cuda_stream(
    torch::stable::accelerator::getCurrentDeviceIndex(), &stream_ptr);
```

## setup.py Changes

### Adding `-DUSE_CUDA` for shim.h

If your extension uses `aoti_torch_get_current_cuda_stream`, the shim header
needs `-DUSE_CUDA` to expose the CUDA-specific functions:

```python
ext_modules.append(
    CUDAExtension(
        name="my_ext",
        sources=["my_ext.cpp", "my_kernel.cu"],
        extra_compile_args={
            "cxx": [
                "-std=c++17",
                "-DUSE_CUDA",
            ],
            "nvcc": [
                "-DUSE_CUDA",
            ],
        },
    )
)
```

### Adding `-DTORCH_TARGET_VERSION`

Some projects may need to specify the target PyTorch version for stable ABI
compatibility:

```python
extra_compile_args={
    "cxx": ["-DTORCH_TARGET_VERSION=0x020b000000000000", ...],
    "nvcc": ["-DTORCH_TARGET_VERSION=0x020b000000000000", ...],
},
```

The hex value encodes the version: `0x020b...` = PyTorch 2.11.
