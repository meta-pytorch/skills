// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>

#include <ATen/ATen.h>

// Returns a + alpha * b.
at::Tensor weighted_sum(
    const at::Tensor& a,
    const at::Tensor& b,
    float alpha) {
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  return a + alpha * b;
}

PYBIND11_MODULE(_C, m) {
  m.def("weighted_sum", &weighted_sum, "Returns a + alpha * b",
        py::arg("a"), py::arg("b"), py::arg("alpha"));
}
