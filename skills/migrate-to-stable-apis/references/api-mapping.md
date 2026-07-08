# Stable ABI API Mapping

Complete mapping from legacy PyTorch C++ APIs to their stable ABI equivalents.

## The three components of the stable ABI

Everything in this mapping comes from one of three layers:

1. **Stable C headers** — `torch/csrc/inductor/aoti_torch/c/shim.h` and friends. Low-level C API implemented by libtorch. ABI-stable across releases with at least a 2-year compatibility window.
2. **Header-only C++** — `torch/headeronly/*`. Pure-header utilities (e.g., `torch::headeronly::ScalarType`, `STD_TORCH_CHECK`). No dependence on `libtorch.so`.
3. **Stable C++ wrappers** — `torch/csrc/stable/*`. High-level convenience wrappers (`torch::stable::Tensor`, `torch::stable::Device`, `STABLE_TORCH_LIBRARY`, etc.) that hide the rough edges of the C API.

Prefer (3) over (1) wherever possible. The C shim is the foundation, not the recommended user surface.

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
| `t.contiguous()` | `torch::stable::contiguous(t)` |
| `at::empty_like(t)` | `torch::stable::empty_like(t)` |

## Tensor Accessors

| Old | Stable ABI |
|-----|-----------|
| `t.data_ptr<T>()` (read-only) | `t.const_data_ptr<T>()` |
| `t.data_ptr<T>()` (read-write) | `t.mutable_data_ptr<T>()` |
| `t.data_ptr<int32_t>()` (read-write) | `t.mutable_data_ptr<int32_t>()` |
| `t.sizes()` | `t.size(i)` (per-element) |
| `t.strides()` | `t.stride(i)` (per-element) |
| `t.scalar_type()` | `t.scalar_type()` (same, but compare against `torch::headeronly::ScalarType`) |
| `t.device()` | `t.device()` (returns `torch::stable::Device`; compare with `t.device().type() == torch::headeronly::DeviceType::CUDA`) |
| `t.get_device_index()` | `t.get_device_index()` — use for `DeviceGuard` and CUDA stream lookup |
| `t.numel()` | `t.numel()` (unchanged) |

## Scalar Types

| Old | Stable ABI |
|-----|-----------|
| `at::kCUDA` / `torch::kCUDA` | `torch::headeronly::kCUDA` (alias for `DeviceType::CUDA`) |
| `at::kCPU` / `torch::kCPU` | `torch::headeronly::kCPU` |
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

## Native function calls (ATen ops)

When the legacy code calls an ATen native function — either free-function style or as a Tensor method — the stable replacement is the free-function form in `torch::stable`:

```cpp
// OLD (free function)
at::pad(input, padding);
// OLD (method)
input.pad(padding);
// NEW (always free function)
torch::stable::pad(input, padding);
```

`torch::stable::ops` provides the curated subset; the generated C shims (`c_shim_aten.h`) expose the lower-level form if you need an op that isn't yet wrapped.

## Headers Reference

| Header | Provides |
|--------|---------|
| `torch/csrc/stable/tensor.h` | `torch::stable::Tensor`, tensor ops |
| `torch/csrc/stable/library.h` | `STABLE_TORCH_LIBRARY`, `STABLE_TORCH_LIBRARY_IMPL`, `STABLE_TORCH_LIBRARY_FRAGMENT`, `TORCH_BOX` |
| `torch/csrc/stable/ops.h` | `torch::stable::new_empty`, `new_zeros`, `view`, `fill_`, `transpose`, `contiguous`, `empty_like`, `pad`, etc. |
| `torch/csrc/stable/accelerator.h` | `torch::stable::accelerator::DeviceGuard` |
| `torch/csrc/stable/device.h` | `torch::stable::Device` |
| `torch/csrc/stable/c/shim.h` | Newer general-purpose C shim location (preferred when available) |
| `torch/headeronly/core/ScalarType.h` | `torch::headeronly::ScalarType` enum, `kFloat`/`kHalf`/etc. aliases |
| `torch/headeronly/util/Exception.h` | `STD_TORCH_CHECK` macro |
| `torch/headeronly/util/shim_utils.h` | Shim utilities, `TORCH_ERROR_CODE_CHECK` |
| `torch/headeronly/version.h` | `TORCH_ABI_VERSION`, `TORCH_FEATURE_VERSION` macros |
| `torch/csrc/inductor/aoti_torch/c/shim.h` | `aoti_torch_get_current_cuda_stream`, `aoti_torch_get_layout`, etc. (general AOTI shims) |
| `torch/csrc/inductor/aoti_torch/generated/c_shim_aten.h` | C-style shims for ATen ops (e.g., `aoti_torch_aten_new_empty`) |
| `torch/csrc/inductor/aoti_torch/generated/c_shim_<backend>.h` | Backend-specific kernel shims (e.g., `aoti_torch_cuda_pad`) — use only inside that backend's kernels; opts out of the dispatcher |
| `cuda_runtime.h` | `cudaStream_t` |

Only include headers from `torch/csrc/stable/`, `torch/headeronly/`, and `torch/csrc/inductor/aoti_torch/`. Do NOT include `pt_stable_utils.h`.
