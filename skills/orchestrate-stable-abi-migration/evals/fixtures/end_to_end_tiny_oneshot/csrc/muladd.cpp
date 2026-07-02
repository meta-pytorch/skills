// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/library.h>

at::Tensor muladd(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  TORCH_CHECK(a.dtype() == at::kFloat, "a must be float32");
  TORCH_CHECK(b.dtype() == at::kFloat, "b must be float32");
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty_like(a_contig);
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* res_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    res_ptr[i] = a_ptr[i] * b_ptr[i] + static_cast<float>(c);
  }
  return result;
}

TORCH_LIBRARY(tinyops, m) {
  m.def("muladd(Tensor a, Tensor b, float c) -> Tensor");
}

TORCH_LIBRARY_IMPL(tinyops, CPU, m) {
  m.impl("muladd", &muladd);
}
