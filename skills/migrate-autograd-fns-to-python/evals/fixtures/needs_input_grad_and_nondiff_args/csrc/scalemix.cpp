// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/library.h>
#include <ATen/ATen.h>

// CPU device kernel for gradext::scalemix: sin(a) + b^2 + shift.
// This is the "real compute" and stays in C++.
at::Tensor scalemix_cpu(const at::Tensor& a, const at::Tensor& b, int64_t shift) {
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  return at::sin(a) + b * b + shift;
}

TORCH_LIBRARY(gradext, m) {
  m.def("scalemix(Tensor a, Tensor b, int shift) -> Tensor");
}

TORCH_LIBRARY_IMPL(gradext, CPU, m) {
  m.impl("scalemix", &scalemix_cpu);
}
