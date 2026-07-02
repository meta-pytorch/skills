// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/library.h>

#include <atomic>

namespace statops {

// Telemetry: counts real device-kernel launches.
std::atomic<int64_t> launch_count{0};

at::Tensor row_l2norm_cpu(const at::Tensor& x, double eps) {
  TORCH_CHECK(x.dim() == 2, "row_l2norm: expected a 2-D input");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "row_l2norm: expected float32");
  launch_count.fetch_add(1);
  return x / (x.norm(2, {-1}, true) + eps);
}

TORCH_LIBRARY(statops, m) {
  m.def("row_l2norm(Tensor x, float eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(statops, CPU, m) {
  m.impl("row_l2norm", &row_l2norm_cpu);
}

} // namespace statops
