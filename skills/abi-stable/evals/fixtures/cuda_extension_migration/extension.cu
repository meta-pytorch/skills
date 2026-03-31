// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void scale_kernel(float* data, float factor, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] *= factor;
  }
}

at::Tensor scale_cuda(const at::Tensor& input, double factor) {
  TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
  c10::cuda::CUDAGuard device_guard(input.device());

  auto output = at::empty(input.sizes(), input.options());
  auto stream = at::cuda::getCurrentCUDAStream();

  int n = input.numel();
  int threads = 256;
  int blocks = (n + threads - 1) / threads;

  scale_kernel<<<blocks, threads, 0, stream>>>(
      output.data_ptr<float>(), static_cast<float>(factor), n);

  return output;
}

TORCH_LIBRARY_IMPL(myops, CUDA, m) {
  m.impl("scale_cuda", &scale_cuda);
}
