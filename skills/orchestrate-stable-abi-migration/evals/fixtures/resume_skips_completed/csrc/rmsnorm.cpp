// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// NOTE: pybind bindings were already removed from this repo (see
// .abi-migration-plan.json — the pybind-to-torch-library step is completed).
// Registration goes through TORCH_LIBRARY; only the unstable-API rewrite
// remains.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <cmath>

at::Tensor rmsnorm(const at::Tensor& x, double eps) {
  TORCH_CHECK(x.dim() == 1, "x must be 1-D");
  TORCH_CHECK(x.dtype() == at::kFloat, "x must be float32");
  at::Tensor x_contig = x.contiguous();
  at::Tensor result = at::empty_like(x_contig);
  const float* x_ptr = x_contig.data_ptr<float>();
  float* res_ptr = result.data_ptr<float>();
  const int64_t n = x_contig.numel();
  float sum_sq = 0.0f;
  for (int64_t i = 0; i < n; i++) {
    sum_sq += x_ptr[i] * x_ptr[i];
  }
  const float rms = std::sqrt(sum_sq / static_cast<float>(n) +
                              static_cast<float>(eps));
  for (int64_t i = 0; i < n; i++) {
    res_ptr[i] = x_ptr[i] / rms;
  }
  return result;
}

TORCH_LIBRARY(normops, m) {
  m.def("rmsnorm(Tensor x, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(normops, CPU, m) {
  m.impl("rmsnorm", &rmsnorm);
}
