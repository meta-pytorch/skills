# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# NOTE: This module SIMULATES the compiled C++ extension (csrc/kernels.cpp)
# so the package is usable without building. Like the real C++ kernel, the
# CPU implementation records telemetry: it bumps a launch counter every time
# the device kernel actually executes. Do not migrate or delete this file;
# it stands in for the device kernels, which stay in C++.

import torch

_LAUNCHES = {"count": 0}


def kernel_launch_count():
    """How many times the (simulated) device kernel has actually run."""
    return _LAUNCHES["count"]


torch.library.define("statops::row_l2norm", "(Tensor x, float eps) -> Tensor")


@torch.library.impl("statops::row_l2norm", "CPU")
def _row_l2norm_cpu(x, eps):
    _LAUNCHES["count"] += 1
    return x / (x.norm(dim=-1, keepdim=True) + eps)
