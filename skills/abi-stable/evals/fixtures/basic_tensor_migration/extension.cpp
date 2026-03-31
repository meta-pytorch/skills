// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>
#include <ATen/ATen.h>

at::Tensor my_add(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes(), "Tensors must have the same shape");
  return a + b;
}

TORCH_LIBRARY_IMPL(myops, CPU, m) {
  m.impl("my_add", &my_add);
}
