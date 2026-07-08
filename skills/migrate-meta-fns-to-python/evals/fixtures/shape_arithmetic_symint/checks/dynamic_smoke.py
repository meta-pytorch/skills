# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test: windowops fakes must be correct AND preserve dynamic shapes.

Compiles wrappers with fullgraph=True, dynamic=True, runs them at one size,
then enables error_on_recompile and runs at a different size. A fake that
calls int(...) on a SymInt (or hardcodes sizes) specializes the graph and
triggers a recompile, which fails this test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch._dynamo as dynamo
import windowops  # noqa: F401

# Eager correctness of the shape math (CPU kernel is ground truth).
out = torch.ops.windowops.window_avg(torch.randn(4, 16), 3, 2, 1)
assert out.shape == (4, 8), f"window_avg eager shape: {out.shape}"
out = torch.ops.windowops.pack_pair(torch.randn(3, 5), torch.randn(3, 5))
assert out.shape == (3, 11), f"pack_pair eager shape: {out.shape}"

# Fake-kernel shape correctness on the meta device.
m = torch.empty(4, 16, device="meta")
out = torch.ops.windowops.window_avg(m, 3, 2, 1)
assert out.shape == (4, 8), f"window_avg meta shape: {out.shape}"
assert out.device.type == "meta"
p = torch.empty(3, 5, device="meta")
out = torch.ops.windowops.pack_pair(p, p)
assert out.shape == (3, 11), f"pack_pair meta shape: {out.shape}"
assert out.device.type == "meta"


def run_window(x):
    return torch.ops.windowops.window_avg(x, 3, 2, 1)


def run_pack(a, b):
    return torch.ops.windowops.pack_pair(a, b)


cw = torch.compile(run_window, fullgraph=True, dynamic=True)
cp = torch.compile(run_pack, fullgraph=True, dynamic=True)

r = cw(torch.randn(4, 16))
assert r.shape == (4, 8), f"compiled window_avg shape: {r.shape}"
a = torch.randn(3, 5)
r = cp(a, a)
assert r.shape == (3, 11), f"compiled pack_pair shape: {r.shape}"

# Different sizes must NOT trigger a recompile (dynamic shapes preserved).
dynamo.config.error_on_recompile = True
r = cw(torch.randn(5, 33))
assert r.shape == (5, 17), f"compiled window_avg dynamic shape: {r.shape}"
b = torch.randn(7, 9)
r = cp(b, b)
assert r.shape == (7, 19), f"compiled pack_pair dynamic shape: {r.shape}"

# Compiled results must match eager (the real CPU kernel).
x = torch.randn(6, 20)
torch.testing.assert_close(cw(x), run_window(x))


print("dynamic smoke OK")
