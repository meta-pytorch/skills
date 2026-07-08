# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test: extpkg::mymuladd must have a Python fake (meta) kernel."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import extpkg  # noqa: F401
import torch

# The CPU kernel must still work and be numerically unchanged.
x = torch.rand(4, 4)
y = torch.rand(4, 4)
torch.testing.assert_close(torch.ops.extpkg.mymuladd(x, y, 0.5), x * y + 0.5)

# The op must run on meta tensors (requires a registered fake kernel).
a = torch.empty(8, 8, device="meta")
b = torch.empty(8, 8, device="meta")
out = torch.ops.extpkg.mymuladd(a, b, 0.5)
assert out.shape == (8, 8), f"bad shape: {out.shape}"
assert out.dtype == torch.float32, f"bad dtype: {out.dtype}"
assert out.device.type == "meta", f"bad device: {out.device}"

# TORCH_CHECK fidelity: the shape check must survive the migration as
# torch._check (RuntimeError), not be dropped and not become a bare assert.
bad = torch.empty(8, 4, device="meta")
try:
    torch.ops.extpkg.mymuladd(a, bad, 0.5)
except RuntimeError:
    pass
else:
    raise SystemExit("expected RuntimeError for mismatched shapes on meta device")

# Same for the dtype checks.
d = torch.empty(8, 8, device="meta", dtype=torch.float64)
try:
    torch.ops.extpkg.mymuladd(d, d, 0.5)
except RuntimeError:
    pass
else:
    raise SystemExit("expected RuntimeError for float64 inputs on meta device")

print("meta smoke OK")
