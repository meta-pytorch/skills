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

// Reference forward attention: softmax(q @ k^T * softmax_scale) @ v.
at::Tensor attn_fwd_cuda(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double softmax_scale) {
  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(q.dim() == 4, "q must be [batch, heads, seq, dim]");
  TORCH_CHECK(k.sizes() == q.sizes(), "k must match q's shape");
  TORCH_CHECK(v.sizes() == q.sizes(), "v must match q's shape");
  TORCH_CHECK(
      q.scalar_type() == at::kHalf || q.scalar_type() == at::kFloat,
      "q must be float16 or float32");

  const c10::cuda::CUDAGuard device_guard(q.device());
  at::Tensor scores = at::matmul(q, k.transpose(-2, -1));
  scores = scores * softmax_scale;
  at::Tensor probs = at::softmax(scores, /*dim=*/-1);
  return at::matmul(probs, v);
}

TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("attn_fwd", &attn_fwd_cuda);
}

} // namespace extension_cpp
