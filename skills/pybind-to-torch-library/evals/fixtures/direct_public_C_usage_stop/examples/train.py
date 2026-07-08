# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Example training loop shipped with the docs. Users copy this file; it
imports the compiled op straight from mylib._C as the README instructs."""

import torch
from mylib._C import fused_op


def train_step(a, b, c):
    return fused_op(a, b, c).sum()


if __name__ == "__main__":
    a = torch.randn(4, 4, requires_grad=True)
    b = torch.randn(4, 4)
    c = torch.randn(4, 4)
    loss = train_step(a, b, c)
    loss.backward()
    print(loss.item())
