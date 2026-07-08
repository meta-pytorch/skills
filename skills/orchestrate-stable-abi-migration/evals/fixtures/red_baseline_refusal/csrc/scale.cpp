// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/library.h>

at::Tensor scale_out(const at::Tensor& x, double factor) {
  TORCH_CHECK(x.dtype() == at::kFloat, "x must be float32");
  at::Tensor x_contig = x.contiguous();
  at::Tensor result = at::empty_like(x_contig);
  const float* x_ptr = x_contig.data_ptr<float>();
  float* res_ptr = result.data_ptr<float>();
  for (int64_t i = 0; i < result.numel(); i++) {
    res_ptr[i] = x_ptr[i] * static_cast<float>(factor);
  }
  return result;
}

TORCH_LIBRARY(scaleops, m) {
  m.def("scale_out(Tensor x, float factor) -> Tensor");
}

TORCH_LIBRARY_IMPL(scaleops, CPU, m) {
  m.impl("scale_out", &scale_out);
}
