# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

mixedext = pytest.importorskip("mixedext")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_scale_add_matches_eager():
    x = torch.randn(64, 32, device="cuda")
    y = torch.randn(64, 32, device="cuda")
    out = mixedext.fused_scale_add(x, y, 2.5)
    torch.testing.assert_close(out, x * 2.5 + y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_already_faked_op_shape():
    x = torch.randn(16, 8, device="cuda")
    out = mixedext.already_faked_op(x)
    assert out.shape == (16,)
