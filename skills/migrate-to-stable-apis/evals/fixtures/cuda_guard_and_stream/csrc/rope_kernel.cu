// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Rotary position embedding, applied in place (vLLM-style out-mutating kernel).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cstdint>

namespace fastrope {

__global__ void rotary_embedding_kernel(
    const int64_t* __restrict__ positions,  // [num_tokens]
    float* __restrict__ query,              // [num_tokens, num_heads, head_dim]
    const float* __restrict__ cos_sin,      // [max_pos, head_dim]
    int num_heads,
    int head_dim) {
  const int token_idx = blockIdx.x;
  const int64_t pos = positions[token_idx];
  const float* cs = cos_sin + pos * head_dim;
  const int half = head_dim / 2;

  for (int i = threadIdx.x; i < num_heads * half; i += blockDim.x) {
    const int head = i / half;
    const int rot = i % half;
    float* q = query + (static_cast<int64_t>(token_idx) * num_heads + head) *
        head_dim;
    const float c = cs[rot];
    const float s = cs[half + rot];
    const float x = q[rot];
    const float y = q[half + rot];
    q[rot] = x * c - y * s;
    q[half + rot] = y * c + x * s;
  }
}

void rotary_embedding(
    const at::Tensor& positions,
    at::Tensor& query,
    const at::Tensor& cos_sin_cache) {
  TORCH_CHECK(positions.dim() == 1, "positions must be 1D [num_tokens]");
  TORCH_CHECK(query.dim() == 3, "query must be 3D [num_tokens, heads, dim]");
  TORCH_CHECK(cos_sin_cache.dim() == 2, "cos_sin_cache must be 2D");
  TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(cos_sin_cache.is_cuda(), "cos_sin_cache must be a CUDA tensor");
  TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
  TORCH_CHECK(cos_sin_cache.is_contiguous(), "cos_sin_cache must be contiguous");
  TORCH_CHECK(
      positions.size(0) == query.size(0),
      "positions and query must agree on num_tokens");
  TORCH_CHECK(
      query.size(2) == cos_sin_cache.size(1),
      "head_dim mismatch between query and cos_sin_cache");
  TORCH_CHECK(query.size(2) % 2 == 0, "head_dim must be even");

  const int64_t num_tokens = query.size(0);
  const int num_heads = static_cast<int>(query.size(1));
  const int head_dim = static_cast<int>(query.size(2));

  const c10::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const dim3 grid(static_cast<unsigned int>(num_tokens));
  const dim3 block(std::min(num_heads * head_dim / 2, 512));

  rotary_embedding_kernel<<<grid, block, 0, stream>>>(
      positions.data_ptr<int64_t>(),
      query.data_ptr<float>(),
      cos_sin_cache.data_ptr<float>(),
      num_heads,
      head_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace fastrope

TORCH_LIBRARY(fastrope, m) {
  m.def(
      "rotary_embedding(Tensor positions, Tensor(a!) query, "
      "Tensor cos_sin_cache) -> ()");
}

TORCH_LIBRARY_IMPL(fastrope, CUDA, m) {
  m.impl("rotary_embedding", &fastrope::rotary_embedding);
}
