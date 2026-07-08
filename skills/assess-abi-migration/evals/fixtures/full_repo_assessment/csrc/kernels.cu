// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace mixedext {

__global__ void fused_scale_add_kernel(
    const float* x,
    const float* y,
    float* out,
    float scale,
    int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = x[i] * scale + y[i];
  }
}

__global__ void row_l2norm_kernel(
    const float* x,
    float* out,
    int64_t rows,
    int64_t cols) {
  int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < rows) {
    float acc = 0.0f;
    for (int64_t c = 0; c < cols; ++c) {
      float v = x[row * cols + c];
      acc += v * v;
    }
    out[row] = sqrtf(acc);
  }
}

at::Tensor fused_scale_add_cuda(
    const at::Tensor& x,
    const at::Tensor& y,
    double scale) {
  TORCH_CHECK(x.is_cuda(), "fused_scale_add: x must be a CUDA tensor");
  TORCH_CHECK(x.sizes() == y.sizes(), "fused_scale_add: shape mismatch");
  TORCH_CHECK(
      x.scalar_type() == at::kFloat, "fused_scale_add: only float32 supported");
  c10::cuda::CUDAGuard device_guard(x.device());
  auto x_c = x.contiguous();
  auto y_c = y.contiguous();
  auto out = at::empty_like(x_c);
  int64_t n = x_c.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t blocks = (n + 255) / 256;
  fused_scale_add_kernel<<<blocks, 256, 0, stream>>>(
      x_c.data_ptr<float>(),
      y_c.data_ptr<float>(),
      out.data_ptr<float>(),
      static_cast<float>(scale),
      n);
  return out;
}

at::Tensor already_faked_op_cuda(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda(), "already_faked_op: x must be a CUDA tensor");
  TORCH_CHECK(x.dim() == 2, "already_faked_op: expected a 2-D tensor");
  TORCH_CHECK(
      x.scalar_type() == at::kFloat, "already_faked_op: only float32 supported");
  c10::cuda::CUDAGuard device_guard(x.device());
  auto x_c = x.contiguous();
  auto out = at::empty({x_c.size(0)}, x_c.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t blocks = (x_c.size(0) + 255) / 256;
  row_l2norm_kernel<<<blocks, 256, 0, stream>>>(
      x_c.data_ptr<float>(), out.data_ptr<float>(), x_c.size(0), x_c.size(1));
  return out;
}

TORCH_LIBRARY(mixedext, m) {
  m.def("fused_scale_add(Tensor x, Tensor y, float scale) -> Tensor");
  m.def("already_faked_op(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(mixedext, CUDA, m) {
  m.impl("fused_scale_add", &fused_scale_add_cuda);
  m.impl("already_faked_op", &already_faked_op_cuda);
}

} // namespace mixedext
