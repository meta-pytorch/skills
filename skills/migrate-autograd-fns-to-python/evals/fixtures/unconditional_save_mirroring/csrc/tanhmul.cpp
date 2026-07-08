// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/library.h>
#include <ATen/ATen.h>

// CPU device kernel for gradext::mytanhmul: tanh(a * b).
// This is the "real compute" and stays in C++.
at::Tensor mytanhmul_cpu(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  return at::tanh(a * b);
}

TORCH_LIBRARY(gradext, m) {
  m.def("mytanhmul(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(gradext, CPU, m) {
  m.impl("mytanhmul", &mytanhmul_cpu);
}
