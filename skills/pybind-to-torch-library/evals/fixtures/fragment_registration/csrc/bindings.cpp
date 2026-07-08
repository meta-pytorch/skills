// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>

#include "ops.h"

PYBIND11_MODULE(_C, m) {
  m.def("rms_norm", &rms_norm, "RMS layer norm",
        py::arg("input"), py::arg("weight"), py::arg("eps"));
  m.def("silu_mul", &silu_mul, "SiLU(gate) * up",
        py::arg("gate"), py::arg("up"));
}
