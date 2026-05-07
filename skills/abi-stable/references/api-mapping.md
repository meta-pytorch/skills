# Stable ABI API Mapping

Complete mapping from legacy PyTorch C++ APIs to their stable ABI equivalents.

## Tensor Types

| Old | Stable ABI |
|-----|-----------|
| `at::Tensor` | `torch::stable::Tensor` |
| `torch::Tensor` | `torch::stable::Tensor` |
| `const at::Tensor&` | `const torch::stable::Tensor&` |

Add a using declaration for convenience:

```cpp
using torch::stable::Tensor;
```

Watch out for `using namespace at;` directives — plain `Tensor` types may actually
be `at::Tensor` and need replacement.

## Tensor Creation

| Old | Stable ABI |
|-----|-----------|
| `at::empty({sizes}, options)` | `torch::stable::new_empty(ref_tensor, {sizes}, dtype)` |
| `at::zeros({sizes}, options)` | `torch::stable::new_zeros(ref_tensor, {sizes}, dtype)` |
| `t.new_empty({sizes})` | `torch::stable::new_empty(t, {sizes}, dtype)` |
| `at::empty_strided({sizes}, {strides}, opts)` | See "Strided Tensors" below |

### Options to explicit arguments

The old `at::TensorOptions()` pattern:

```cpp
// OLD
auto opts = at::TensorOptions().dtype(at::kFloat).device(input.device());
auto out = at::empty({M, N}, opts);
```

Becomes:

```cpp
// NEW — derive device from a reference tensor
auto out = torch::stable::new_empty(input, {M, N},
    torch::headeronly::ScalarType::Float);
```

When possible, prefer `torch::stable::new_empty(ref_tensor, ...)` over
`torch::stable::empty(...)` since it inherits device from the reference tensor
automatically.

### Strided Tensors

`torch::stable::new_empty_strided` does **not exist**. To create a tensor with
non-contiguous strides, create a contiguous tensor with transposed dimensions and
then transpose:

```cpp
// OLD: column-major tensor {rows, cols} with strides {1, rows}
auto out = at::empty_strided({rows, cols}, {1, rows}, options);

// NEW: create transposed shape, then transpose
auto tmp = torch::stable::new_empty(input, {cols, rows},
    torch::headeronly::ScalarType::Float8_e4m3fn);
auto out = torch::stable::transpose(tmp, 0, 1);
```

## Tensor Operations

| Old | Stable ABI |
|-----|-----------|
| `t.view({...})` | `torch::stable::view(t, {...})` |
| `t.fill_(val)` | `torch::stable::fill_(t, val)` |
| `t.t()` | `torch::stable::transpose(t, 0, 1)` |
| `t.transpose(d0, d1)` | `torch::stable::transpose(t, d0, d1)` |

## Tensor Accessors

| Old | Stable ABI |
|-----|-----------|
| `t.data_ptr<T>()` (read-only) | `t.const_data_ptr<T>()` |
| `t.data_ptr<T>()` (read-write) | `t.data_ptr()` then `static_cast` or `reinterpret_cast` |
| `t.data_ptr<int32_t>()` | `static_cast<const int32_t*>(t.data_ptr())` |
| `t.sizes()` | `t.size(i)` (per-element) |
| `t.strides()` | `t.stride(i)` (per-element) |
| `t.scalar_type()` | `t.scalar_type()` (same, but compare against `torch::headeronly::ScalarType`) |
| `t.device()` | `t.device()` (for DeviceGuard, use `t.get_device_index()` instead) |

## Scalar Types

| Old | Stable ABI |
|-----|-----------|
| `at::kFloat` / `at::kFloat32` / `torch::kFloat32` | `torch::headeronly::ScalarType::Float` |
| `at::kHalf` / `at::kFloat16` / `torch::kFloat16` | `torch::headeronly::ScalarType::Half` |
| `at::kBFloat16` | `torch::headeronly::ScalarType::BFloat16` |
| `at::kFloat8_e4m3fn` | `torch::headeronly::ScalarType::Float8_e4m3fn` |
| `at::kFloat8_e8m0fnu` | `torch::headeronly::ScalarType::Float8_e8m0fnu` |
| `at::kByte` / `at::kUInt8` | `torch::headeronly::ScalarType::Byte` |
| `at::kInt` | `torch::headeronly::ScalarType::Int` |
| `at::kLong` | `torch::headeronly::ScalarType::Long` |
| `at::kBool` | `torch::headeronly::ScalarType::Bool` |
| `at::ScalarType::X` | `torch::headeronly::ScalarType::X` |

## Error Checking

| Old | Stable ABI |
|-----|-----------|
| `TORCH_CHECK(cond, msg...)` | `STD_TORCH_CHECK(cond, msg...)` |

**Important**: Only replace the macro name. Do NOT change the condition or message arguments.

## CUDA

### Device Guard

| Old | Stable ABI |
|-----|-----------|
| `c10::cuda::CUDAGuard guard(tensor.device())` | `torch::stable::accelerator::DeviceGuard guard(tensor.get_device_index())` |
| `at::cuda::CUDAGuard guard(device)` | `torch::stable::accelerator::DeviceGuard guard(tensor.get_device_index())` |

**Important**: Pass `tensor.get_device_index()` (an integer), not `tensor.device()`.
Do NOT cast to `char` or any narrow type. DeviceGuard is RAII — do not move its
declaration to an inner scope.

### CUDA Streams

| Old | Stable ABI |
|-----|-----------|
| `at::cuda::getCurrentCUDAStream()` | See below |

Replace:

```cpp
// OLD
cudaStream_t stream = at::cuda::getCurrentCUDAStream();

// NEW
void* stream_ptr = nullptr;
TORCH_ERROR_CODE_CHECK(
    aoti_torch_get_current_cuda_stream(tensor.get_device_index(), &stream_ptr));
cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);
```

**Important**: Get the device index from a tensor (`tensor.get_device_index()`), NOT
from `torch::stable::accelerator::getCurrentDeviceIndex()`.

**Important**: Do NOT forward-declare `aoti_torch_get_current_cuda_stream`. Include
it from `torch/csrc/inductor/aoti_torch/c/shim.h`. Also include `<cuda_runtime.h>`
for `cudaStream_t`.

### CUDA Kernel Launch Check

| Old | Stable ABI |
|-----|-----------|
| `C10_CUDA_KERNEL_LAUNCH_CHECK()` | `STD_CUDA_KERNEL_LAUNCH_CHECK()` |

Make sure the appropriate header is included for `STD_CUDA_KERNEL_LAUNCH_CHECK`.

## Library Registration

| Old | Stable ABI |
|-----|-----------|
| `TORCH_LIBRARY_IMPL(lib, dispatch, m)` | `STABLE_TORCH_LIBRARY_IMPL(lib, dispatch, m)` |
| `TORCH_LIBRARY_FRAGMENT(lib, m)` | `STABLE_TORCH_LIBRARY_FRAGMENT(lib, m)` |
| `m.impl("name", &func)` | `m.impl("name", TORCH_BOX(&func))` |
| `m.impl(TORCH_SELECTIVE_NAME("name"), TORCH_FN(&func))` | `m.impl("name", TORCH_BOX(&func))` |

**Important**: Inside `STABLE_TORCH_LIBRARY_FRAGMENT` / `STABLE_TORCH_LIBRARY_IMPL`:

- Do NOT wrap the first argument of `m.impl()` with `TORCH_SELECTIVE_NAME`
- Do NOT wrap the second argument with `TORCH_FN`
- DO wrap the function pointer with `TORCH_BOX`

## Layout Checks

For layout comparisons that previously used `at::Layout::Strided`:

```cpp
// OLD
TORCH_CHECK(tensor.layout() == at::Layout::Strided, "must be strided");

// NEW
using torch::headeronly::Layout;
int32_t layout_val;
TORCH_ERROR_CODE_CHECK(aoti_torch_get_layout(tensor.get(), layout_val));
auto layout = torch::stable::detail::to<Layout>(
    torch::stable::detail::from(layout_val));
STD_TORCH_CHECK(layout == Layout::Strided, "must be strided");
```

## Headers Reference

| Header | Provides |
|--------|---------|
| `torch/csrc/stable/tensor.h` | `torch::stable::Tensor`, tensor ops |
| `torch/csrc/stable/library.h` | `STABLE_TORCH_LIBRARY_IMPL`, `STABLE_TORCH_LIBRARY_FRAGMENT` |
| `torch/csrc/stable/ops.h` | `torch::stable::new_empty`, `new_zeros`, `view`, `fill_`, `transpose` |
| `torch/csrc/stable/accelerator.h` | `torch::stable::accelerator::DeviceGuard` |
| `torch/csrc/stable/device.h` | Device-related utilities |
| `torch/headeronly/core/ScalarType.h` | `torch::headeronly::ScalarType` enum |
| `torch/headeronly/util/Exception.h` | `STD_TORCH_CHECK` macro |
| `torch/headeronly/util/shim_utils.h` | Shim utilities |
| `torch/csrc/inductor/aoti_torch/c/shim.h` | `aoti_torch_get_current_cuda_stream`, `aoti_torch_get_layout` |
| `cuda_runtime.h` | `cudaStream_t` |

Only include headers from `torch/csrc/stable/`, `torch/headeronly/`, and
`torch/csrc/inductor/aoti_torch/`. Do NOT include `pt_stable_utils.h`.
