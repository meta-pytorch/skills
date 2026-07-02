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
import torch.nn.functional as F

torch.library.define(
    "windowops::window_avg", "(Tensor x, int kernel, int stride, int pad) -> Tensor"
)
torch.library.define("windowops::pack_pair", "(Tensor a, Tensor b) -> Tensor")


@torch.library.impl("windowops::window_avg", "CPU")
def _window_avg_cpu(x, kernel, stride, pad):
    return F.pad(x, (pad, pad)).unfold(-1, kernel, stride).mean(-1)


@torch.library.impl("windowops::pack_pair", "CPU")
def _pack_pair_cpu(a, b):
    sep = a.new_ones(a.shape[0], 1)
    return torch.cat([a, sep, b], dim=1)
