# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

tinyops = pytest.importorskip("tinyops")


def test_add_bias():
    x = torch.randn(4, 8)
    bias = torch.randn(8)
    torch.testing.assert_close(torch.ops.tinyops.add_bias(x, bias), x + bias)


def test_mean_center():
    x = torch.randn(16)
    out = torch.ops.tinyops.mean_center(x)
    torch.testing.assert_close(out.mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
