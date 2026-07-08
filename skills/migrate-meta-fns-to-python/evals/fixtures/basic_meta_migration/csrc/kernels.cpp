// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/library.h>

namespace extpkg {

at::Tensor mymuladd_cpu(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes(), "mymuladd: a and b must have the same shape");
  TORCH_CHECK(a.scalar_type() == at::kFloat, "mymuladd: a must be float32");
  TORCH_CHECK(b.scalar_type() == at::kFloat, "mymuladd: b must be float32");
  return a * b + c;
}

TORCH_LIBRARY(extpkg, m) {
  m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
}

TORCH_LIBRARY_IMPL(extpkg, CPU, m) {
  m.impl("mymuladd", &mymuladd_cpu);
}

} // namespace extpkg
