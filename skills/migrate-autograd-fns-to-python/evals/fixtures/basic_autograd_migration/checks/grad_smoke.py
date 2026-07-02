# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test: gradext::mymuladd must produce correct gradients.

Run from the project root: python3 checks/grad_smoke.py
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import gradext  # noqa: F401
import torch


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


gen = torch.Generator().manual_seed(0)
a = torch.randn(16, generator=gen, requires_grad=True)
b = torch.randn(16, generator=gen, requires_grad=True)
grad_out = torch.randn(16, generator=gen)

out = torch.ops.gradext.mymuladd(a, b, 0.75)
if not torch.allclose(out, a.detach() * b.detach() + 0.75):
    fail("forward result is wrong")

out.backward(grad_out)
if a.grad is None or b.grad is None:
    fail(
        "backward produced no gradients for gradext::mymuladd - "
        "is an autograd implementation registered for the op?"
    )
if not torch.allclose(a.grad, grad_out * b.detach()):
    fail("a.grad does not match the analytic gradient grad * b")
if not torch.allclose(b.grad, grad_out * a.detach()):
    fail("b.grad does not match the analytic gradient grad * a")

a64 = torch.randn(8, dtype=torch.float64, requires_grad=True)
b64 = torch.randn(8, dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(lambda x, y: torch.ops.gradext.mymuladd(x, y, 0.75), (a64, b64))

print("OK: gradext::mymuladd gradients verified (analytic + gradcheck)")
