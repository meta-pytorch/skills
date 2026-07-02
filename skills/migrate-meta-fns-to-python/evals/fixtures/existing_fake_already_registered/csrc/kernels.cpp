// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/library.h>

namespace gridops {

at::Tensor colwise_scale_cpu(const at::Tensor& x, const at::Tensor& scales) {
  TORCH_CHECK(x.dim() == 2, "colwise_scale: expected a 2-D input");
  TORCH_CHECK(scales.dim() == 1, "colwise_scale: scales must be 1-D");
  TORCH_CHECK(x.size(1) == scales.size(0), "colwise_scale: scales length must match x columns");
  return x * scales.unsqueeze(0);
}

at::Tensor tile_rows_cpu(const at::Tensor& x, int64_t repeats) {
  TORCH_CHECK(x.dim() == 2, "tile_rows: expected a 2-D input");
  TORCH_CHECK(repeats >= 1, "tile_rows: repeats must be >= 1");
  return x.repeat({repeats, 1});
}

TORCH_LIBRARY(gridops, m) {
  m.def("colwise_scale(Tensor x, Tensor scales) -> Tensor");
  m.def("tile_rows(Tensor x, int repeats) -> Tensor");
}

TORCH_LIBRARY_IMPL(gridops, CPU, m) {
  m.impl("colwise_scale", &colwise_scale_cpu);
  m.impl("tile_rows", &tile_rows_cpu);
}

} // namespace gridops
