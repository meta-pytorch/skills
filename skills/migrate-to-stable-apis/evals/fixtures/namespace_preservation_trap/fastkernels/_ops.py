# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python wrapper API for the fastkernels extension."""

import torch


@torch.library.register_fake("fastkernels::rmsnorm")
def _rmsnorm_fake(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    torch._check(input.dim() == 2)
    torch._check(weight.dim() == 1)
    torch._check(input.size(1) == weight.size(0))
    return torch.empty_like(input)


def rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMS normalization over the last dimension."""
    return torch.ops.fastkernels.rmsnorm(input, weight, eps)
