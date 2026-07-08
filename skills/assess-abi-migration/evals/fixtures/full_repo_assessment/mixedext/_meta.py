# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python fake (meta) registrations for ops that already have one."""

import torch


@torch.library.register_fake("mixedext::already_faked_op")
def _already_faked_op_fake(x):
    torch._check(x.dim() == 2, lambda: "already_faked_op: expected 2-D input")
    return x.new_empty((x.shape[0],))
