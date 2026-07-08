# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from . import _meta  # noqa: F401


def fused_scale_add(x, y, scale=1.0):
    return torch.ops.mixedext.fused_scale_add(x, y, scale)


def already_faked_op(x):
    return torch.ops.mixedext.already_faked_op(x)
