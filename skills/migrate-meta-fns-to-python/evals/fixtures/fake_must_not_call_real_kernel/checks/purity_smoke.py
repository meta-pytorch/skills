# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test: the fake kernel must never execute the real device kernel.

The simulated compiled kernel bumps a launch counter on every real run. Meta
and fake-tensor calls must produce correct metadata WITHOUT bumping it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import statops
import torch

# Real CPU call: counter goes up, result is correct.
x = torch.rand(4, 8)
before = statops.kernel_launch_count()
out = torch.ops.statops.row_l2norm(x, 1e-6)
assert statops.kernel_launch_count() == before + 1, "CPU kernel did not run"
torch.testing.assert_close(out, x / (x.norm(dim=-1, keepdim=True) + 1e-6))

# Meta call: correct metadata, NO real kernel launch, no real allocation.
before = statops.kernel_launch_count()
m = torch.empty(16, 32, device="meta")
out = torch.ops.statops.row_l2norm(m, 1e-6)
assert out.shape == (16, 32), f"bad meta shape: {out.shape}"
assert out.dtype == torch.float32, f"bad meta dtype: {out.dtype}"
assert out.device.type == "meta", f"bad meta device: {out.device}"
assert statops.kernel_launch_count() == before, (
    "fake kernel executed the real device kernel (launch counter moved)"
)

# FakeTensorMode call: same contract.
from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: E402

before = statops.kernel_launch_count()
with FakeTensorMode():
    f = torch.empty(6, 10)
    out = torch.ops.statops.row_l2norm(f, 1e-6)
    assert out.shape == (6, 10), f"bad fake shape: {out.shape}"
assert statops.kernel_launch_count() == before, (
    "fake kernel executed the real device kernel under FakeTensorMode"
)

print("purity smoke OK")
