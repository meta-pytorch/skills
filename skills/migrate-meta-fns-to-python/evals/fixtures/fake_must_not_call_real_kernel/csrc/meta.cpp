// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/library.h>

namespace statops {

at::Tensor row_l2norm_meta(const at::Tensor& x, double eps) {
  TORCH_CHECK(x.dim() == 2, "row_l2norm: expected a 2-D input");
  TORCH_CHECK(x.scalar_type() == at::kFloat, "row_l2norm: expected float32");
  return at::empty_like(x);
}

TORCH_LIBRARY_IMPL(statops, Meta, m) {
  m.impl("row_l2norm", &row_l2norm_meta);
}

} // namespace statops
