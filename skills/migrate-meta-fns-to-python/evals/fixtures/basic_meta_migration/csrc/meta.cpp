// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <torch/library.h>

namespace extpkg {

at::Tensor mymuladd_meta(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sym_sizes() == b.sym_sizes(), "mymuladd: a and b must have the same shape");
  TORCH_CHECK(a.scalar_type() == at::kFloat, "mymuladd: a must be float32");
  TORCH_CHECK(b.scalar_type() == at::kFloat, "mymuladd: b must be float32");
  return at::empty_like(a);
}

TORCH_LIBRARY_IMPL(extpkg, Meta, m) {
  m.impl("mymuladd", &mymuladd_meta);
}

} // namespace extpkg
