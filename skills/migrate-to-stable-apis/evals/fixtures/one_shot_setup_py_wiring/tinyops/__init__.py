# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""tinyops: a tiny PyTorch C++ extension."""

import torch


def muladd(a: torch.Tensor, b: torch.Tensor, c: float) -> torch.Tensor:
    return torch.ops.tinyops.muladd(a, b, c)


def relu_out(input: torch.Tensor, out: torch.Tensor) -> None:
    torch.ops.tinyops.relu_out(input, out)
