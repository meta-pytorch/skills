# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

gapext = pytest.importorskip("gapext")


def test_clamp_shift():
    x = torch.randn(32)
    out = torch.ops.gapext.clamp_shift(x, -0.5, 0.5)
    torch.testing.assert_close(out, x.clamp(-0.5, 0.5) + 1.0)


def test_make_colmajor_buffer():
    ref = torch.randn(4, 6)
    out = torch.ops.gapext.make_colmajor_buffer(ref)
    assert out.shape == (4, 6)
    assert out.stride() == (1, 4)
