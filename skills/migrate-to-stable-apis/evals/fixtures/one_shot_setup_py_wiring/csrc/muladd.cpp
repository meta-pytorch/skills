// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Fused multiply-add: out = a * b + c, elementwise over float32 tensors.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <cstdint>

namespace tinyops {

at::Tensor muladd(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  TORCH_CHECK(a.scalar_type() == at::kFloat, "a must be float32");
  TORCH_CHECK(b.scalar_type() == at::kFloat, "b must be float32");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

  at::Tensor out = at::empty_like(a);

  const float* a_ptr = a.data_ptr<float>();
  const float* b_ptr = b.data_ptr<float>();
  float* out_ptr = out.data_ptr<float>();
  const float cf = static_cast<float>(c);

  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) {
    out_ptr[i] = a_ptr[i] * b_ptr[i] + cf;
  }

  return out;
}

}  // namespace tinyops

TORCH_LIBRARY(tinyops, m) {
  m.def("muladd(Tensor a, Tensor b, float c) -> Tensor");
  m.def("relu_out(Tensor input, Tensor(a!) out) -> ()");
}

TORCH_LIBRARY_IMPL(tinyops, CPU, m) {
  m.impl("muladd", &tinyops::muladd);
}
