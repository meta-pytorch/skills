# Migration Example

This shows a concrete before/after for a CUDA extension file migration to the stable ABI.
The example is based on real migrations of PyTorch extension projects.

## Before (unstable API)

```cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <string>

void check_cuda_tensor(const at::Tensor &t, const char *name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

std::tuple<at::Tensor, at::Tensor>
my_quantize(const at::Tensor& input, int64_t block_size,
            const std::string &format) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(input.scalar_type() == at::kFloat ||
                input.scalar_type() == at::kHalf ||
                input.scalar_type() == at::kBFloat16,
                "Input must be float32, float16, or bfloat16");

    const int64_t rows = input.size(0);
    const int64_t cols = input.size(1);

    c10::cuda::CUDAGuard device_guard(input.device());

    // Create output tensors
    const auto options_out = at::TensorOptions()
                                 .dtype(at::kFloat8_e4m3fn)
                                 .device(input.device());
    const auto options_scale = at::TensorOptions()
                                   .dtype(at::kFloat)
                                   .device(input.device());

    at::Tensor output = at::empty({rows, cols}, options_out);

    const int64_t num_blocks = (cols + block_size - 1) / block_size;
    at::Tensor scales = at::zeros({rows, num_blocks}, options_scale);

    // Column-major output
    at::Tensor output_colmajor = at::empty_strided(
        {rows, cols}, {1, rows}, options_out);

    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Get strides
    int64_t stride_0 = scales.strides()[0];
    int64_t stride_1 = scales.strides()[1];

    // Launch kernel...
    launch_kernel(
        input.data_ptr<float>(),
        output.data_ptr(),
        scales.data_ptr<float>(),
        rows, cols, block_size,
        stride_0, stride_1,
        stream);

    return std::make_tuple(output, scales);
}

TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
    m.impl("my_quantize", &my_quantize);
}
```

## After (stable ABI)

```cpp
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

using torch::stable::Tensor;
namespace tsa = torch::stable::accelerator;

void check_cuda_tensor(const Tensor &t, const char *name) {
    STD_TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    STD_TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

std::tuple<Tensor, Tensor>
my_quantize(const Tensor& input, int64_t block_size,
            const std::string &format) {

    STD_TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    STD_TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    STD_TORCH_CHECK(input.dim() == 2, "input must be 2D");
    STD_TORCH_CHECK(input.scalar_type() == torch::headeronly::ScalarType::Float ||
                    input.scalar_type() == torch::headeronly::ScalarType::Half ||
                    input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
                    "Input must be float32, float16, or bfloat16");

    const int64_t rows = input.size(0);
    const int64_t cols = input.size(1);

    // DeviceGuard takes device index, not device object
    tsa::DeviceGuard device_guard(input.get_device_index());

    // Create output tensors — derive device from reference tensor
    Tensor output = torch::stable::new_empty(input, {rows, cols},
        torch::headeronly::ScalarType::Float8_e4m3fn);

    const int64_t num_blocks = (cols + block_size - 1) / block_size;
    Tensor scales = torch::stable::new_zeros(input, {rows, num_blocks},
        torch::headeronly::ScalarType::Float);

    // Column-major output: new_empty_strided does NOT exist!
    // Create transposed shape and transpose instead.
    Tensor output_colmajor_tmp = torch::stable::new_empty(input, {cols, rows},
        torch::headeronly::ScalarType::Float8_e4m3fn);
    Tensor output_colmajor = torch::stable::transpose(output_colmajor_tmp, 0, 1);

    // Get CUDA stream — use device index from tensor, not global
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(input.get_device_index(), &stream_ptr));
    cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

    // Use .stride(i) instead of .strides()[i]
    int64_t stride_0 = scales.stride(0);
    int64_t stride_1 = scales.stride(1);

    // Launch kernel...
    launch_kernel(
        input.const_data_ptr<float>(),  // const_data_ptr for read-only
        output.data_ptr(),
        static_cast<float*>(scales.data_ptr()),
        rows, cols, block_size,
        stride_0, stride_1,
        stream);

    return std::make_tuple(output, scales);
}

// STABLE_TORCH_LIBRARY_IMPL + TORCH_BOX
STABLE_TORCH_LIBRARY_IMPL(mylib, CUDA, m) {
    m.impl("my_quantize", TORCH_BOX(&my_quantize));
}
```

## Key Differences Summary

1. **Includes**: `torch/extension.h` and ATen headers -> `torch/csrc/stable/*` and `torch/headeronly/*`
2. **Tensor type**: `at::Tensor` -> `torch::stable::Tensor`
3. **Error macro**: `TORCH_CHECK` -> `STD_TORCH_CHECK` (same arguments)
4. **Device guard**: `c10::cuda::CUDAGuard(tensor.device())` -> `tsa::DeviceGuard(tensor.get_device_index())`
5. **Tensor creation**: `at::empty({}, options)` -> `torch::stable::new_empty(ref, {}, dtype)`
6. **Strided tensors**: `at::empty_strided()` -> create + `torch::stable::transpose()`
7. **Zeros**: `at::zeros({}, options)` -> `torch::stable::new_zeros(ref, {}, dtype)`
8. **CUDA stream**: `at::cuda::getCurrentCUDAStream()` -> `aoti_torch_get_current_cuda_stream(idx, &ptr)`
9. **Scalar types**: `at::kFloat` -> `torch::headeronly::ScalarType::Float`
10. **Strides/sizes**: `.strides()` -> `.stride(i)`, `.sizes()` -> `.size(i)`
11. **Data pointers**: `.data_ptr<T>()` (read) -> `.const_data_ptr<T>()`
12. **Library reg**: `TORCH_LIBRARY_IMPL` -> `STABLE_TORCH_LIBRARY_IMPL` + `TORCH_BOX`
