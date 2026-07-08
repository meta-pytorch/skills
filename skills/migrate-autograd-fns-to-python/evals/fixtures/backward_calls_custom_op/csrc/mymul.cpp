// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/library.h>
#include <ATen/ATen.h>

// CPU device kernels for gradext. The backward kernel is a hand-tuned
// fused kernel in the real extension; both stay in C++.

at::Tensor mymul_cpu(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  return a * b;
}

at::Tensor mymul_backward_kernel_cpu(const at::Tensor& grad, const at::Tensor& other) {
  return grad * other;
}

TORCH_LIBRARY(gradext, m) {
  m.def("mymul(Tensor a, Tensor b) -> Tensor");
  m.def("mymul_backward_kernel(Tensor grad, Tensor other) -> Tensor");
}

TORCH_LIBRARY_IMPL(gradext, CPU, m) {
  m.impl("mymul", &mymul_cpu);
  m.impl("mymul_backward_kernel", &mymul_backward_kernel_cpu);
}
