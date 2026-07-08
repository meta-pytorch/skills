# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This module SIMULATES the compiled C++ extension (csrc/kernels.cpp)
# so the package is usable without building. It registers the op schemas and
# the CPU device kernels exactly as the compiled TORCH_LIBRARY block would.
# Do not migrate or delete this file; it stands in for the device kernels,
# which stay in C++.

import torch

torch.library.define("gridops::colwise_scale", "(Tensor x, Tensor scales) -> Tensor")
torch.library.define("gridops::tile_rows", "(Tensor x, int repeats) -> Tensor")


@torch.library.impl("gridops::colwise_scale", "CPU")
def _colwise_scale_cpu(x, scales):
    return x * scales.unsqueeze(0)


@torch.library.impl("gridops::tile_rows", "CPU")
def _tile_rows_cpu(x, repeats):
    return x.repeat(repeats, 1)
