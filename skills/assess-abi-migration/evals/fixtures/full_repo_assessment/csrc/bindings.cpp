// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>

#include <pybind11/pybind11.h>

namespace mixedext {

// Legacy pybind-only entry points, exposed as mixedext._C.* functions
// instead of going through the dispatcher.
at::Tensor scale_only(const at::Tensor& x, double scale) {
  TORCH_CHECK(x.numel() > 0, "scale_only: input must be non-empty");
  return x * scale;
}

int64_t version_tag() {
  return 3;
}

} // namespace mixedext

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "scale_only",
      &mixedext::scale_only,
      "Multiply a tensor by a scalar (legacy pybind path)",
      pybind11::arg("x"),
      pybind11::arg("scale"));
  m.def("version_tag", &mixedext::version_tag, "Extension version tag");
}
