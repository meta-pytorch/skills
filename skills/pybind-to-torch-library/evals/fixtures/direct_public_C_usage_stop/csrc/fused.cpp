// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>

#include <ATen/ATen.h>

// Fused multiply-add followed by ReLU: relu(a * b + c).
at::Tensor fused_op(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c) {
  TORCH_CHECK(a.sizes() == b.sizes() && a.sizes() == c.sizes(),
              "all inputs must have the same shape");
  return at::relu(a * b + c);
}

PYBIND11_MODULE(_C, m) {
  m.def("fused_op", &fused_op, "Fused multiply-add + ReLU",
        py::arg("a"), py::arg("b"), py::arg("c"));
}
