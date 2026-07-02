// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace extension_cpp {

__global__ void muladd_kernel(
    int64_t numel,
    const float* a,
    const float* b,
    float c,
    float* out) {
  int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (idx < numel) {
    out[idx] = a[idx] * b[idx] + c;
  }
}

at::Tensor mymuladd_cuda(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  TORCH_CHECK(a.dtype() == at::kFloat, "a must be float32");
  TORCH_CHECK(b.dtype() == at::kFloat, "b must be float32");
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");

  const c10::cuda::CUDAGuard device_guard(a.device());
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor out = at::empty_like(a_contig);

  int64_t numel = a_contig.numel();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int threads = 256;
  int64_t blocks = (numel + threads - 1) / threads;
  muladd_kernel<<<blocks, threads, 0, stream>>>(
      numel,
      a_contig.const_data_ptr<float>(),
      b_contig.const_data_ptr<float>(),
      static_cast<float>(c),
      out.mutable_data_ptr<float>());
  return out;
}

TORCH_LIBRARY(extension_cpp, m) {
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
  m.def("attn_fwd(Tensor q, Tensor k, Tensor v, float softmax_scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("mymuladd", &mymuladd_cuda);
}

} // namespace extension_cpp
