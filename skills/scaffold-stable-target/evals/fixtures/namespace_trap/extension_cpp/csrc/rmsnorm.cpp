// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/library.h>

namespace extension_cpp {

at::Tensor rmsnorm_cpu(const at::Tensor& input, const at::Tensor& weight, double eps) {
  TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
  TORCH_CHECK(
      weight.numel() == input.size(-1),
      "weight must match input's last dimension");
  at::Tensor x = input.contiguous().to(at::kFloat);
  at::Tensor variance = x.pow(2).mean(/*dim=*/-1, /*keepdim=*/true);
  at::Tensor normed = x * at::rsqrt(variance + eps);
  return (normed * weight.to(at::kFloat)).to(input.scalar_type());
}

TORCH_LIBRARY(extension_cpp, m) {
  m.def("rmsnorm(Tensor input, Tensor weight, float eps) -> Tensor");
  m.def("quantize_per_tensor(Tensor input, float scale) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
  m.impl("rmsnorm", &rmsnorm_cpu);
}

} // namespace extension_cpp
