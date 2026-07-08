// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/library.h>

namespace tinyops {

at::Tensor add_bias(const at::Tensor& x, const at::Tensor& bias) {
  TORCH_CHECK(
      bias.numel() == x.size(-1), "add_bias: bias must match last dim");
  return x + bias;
}

TORCH_LIBRARY(tinyops, m) {
  m.def("add_bias(Tensor x, Tensor bias) -> Tensor");
}

TORCH_LIBRARY_IMPL(tinyops, CompositeExplicitAutograd, m) {
  m.impl("add_bias", &add_bias);
}

} // namespace tinyops
