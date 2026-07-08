// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/library.h>

namespace gridops {

// NOTE: colwise_scale also has a Python fake in gridops/_meta.py; this C++
// Meta impl predates it and is redundant.
at::Tensor colwise_scale_meta(const at::Tensor& x, const at::Tensor& scales) {
  TORCH_CHECK(x.dim() == 2, "colwise_scale: expected a 2-D input");
  TORCH_CHECK(scales.dim() == 1, "colwise_scale: scales must be 1-D");
  TORCH_CHECK(x.sym_sizes()[1] == scales.sym_sizes()[0],
              "colwise_scale: scales length must match x columns");
  return at::empty_like(x);
}

at::Tensor tile_rows_meta(const at::Tensor& x, int64_t repeats) {
  TORCH_CHECK(x.dim() == 2, "tile_rows: expected a 2-D input");
  TORCH_CHECK(repeats >= 1, "tile_rows: repeats must be >= 1");
  auto R = x.sym_sizes()[0];
  auto C = x.sym_sizes()[1];
  return at::empty_symint({R * repeats, C}, x.options());
}

TORCH_LIBRARY_IMPL(gridops, Meta, m) {
  m.impl("colwise_scale", &colwise_scale_meta);
  m.impl("tile_rows", &tile_rows_meta);
}

} // namespace gridops
