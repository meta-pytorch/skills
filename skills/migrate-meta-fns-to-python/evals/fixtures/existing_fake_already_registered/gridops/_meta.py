# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python fake (meta) kernels for gridops."""

import torch


@torch.library.register_fake("gridops::colwise_scale")
def _(x, scales):
    torch._check(x.dim() == 2, lambda: "colwise_scale: expected a 2-D input")
    torch._check(scales.dim() == 1, lambda: "colwise_scale: scales must be 1-D")
    torch._check(
        x.shape[1] == scales.shape[0],
        lambda: "colwise_scale: scales length must match x columns",
    )
    return torch.empty_like(x)
