# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This module SIMULATES the compiled C++ extension (csrc/kernels.cpp)
# so the package is usable without building. It registers the op schema and
# the CPU device kernel exactly as the compiled TORCH_LIBRARY block would.
# Do not migrate or delete this file; it stands in for the device kernels,
# which stay in C++.

import torch

torch.library.define("extpkg::mymuladd", "(Tensor a, Tensor b, float c) -> Tensor")


@torch.library.impl("extpkg::mymuladd", "CPU")
def _mymuladd_cpu(a, b, c):
    return a * b + c
