// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/library.h>

namespace tinyops {

at::Tensor mean_center(const at::Tensor& x) {
  TORCH_CHECK(x.numel() > 0, "mean_center: input must be non-empty");
  auto out = at::empty_like(x);
  out.copy_(x - x.mean());
  return out;
}

TORCH_LIBRARY_FRAGMENT(tinyops, m) {
  m.def("mean_center(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(tinyops, CompositeExplicitAutograd, m) {
  m.impl("mean_center", &mean_center);
}

} // namespace tinyops
