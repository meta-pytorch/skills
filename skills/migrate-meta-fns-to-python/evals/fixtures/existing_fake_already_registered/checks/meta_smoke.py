# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test: both gridops ops must work on the meta device via Python fakes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gridops  # noqa: F401
import torch

# CPU behavior unchanged.
x = torch.rand(3, 4)
s = torch.rand(4)
torch.testing.assert_close(torch.ops.gridops.colwise_scale(x, s), x * s.unsqueeze(0))
torch.testing.assert_close(torch.ops.gridops.tile_rows(x, 3), x.repeat(3, 1))

# colwise_scale on meta (pre-existing Python fake must still be in effect).
xm = torch.empty(5, 7, device="meta")
sm = torch.empty(7, device="meta")
out = torch.ops.gridops.colwise_scale(xm, sm)
assert out.shape == (5, 7), f"colwise_scale meta shape: {out.shape}"
assert out.device.type == "meta"

# tile_rows on meta (needs the newly migrated fake).
out = torch.ops.gridops.tile_rows(xm, 4)
assert out.shape == (20, 7), f"tile_rows meta shape: {out.shape}"
assert out.device.type == "meta"
assert out.dtype == xm.dtype

print("meta smoke OK")
