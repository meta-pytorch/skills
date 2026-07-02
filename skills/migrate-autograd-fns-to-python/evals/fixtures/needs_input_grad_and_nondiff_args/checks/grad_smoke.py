# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test: gradext::scalemix gradients under three requires_grad configs.

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


gen = torch.Generator().manual_seed(7)


def run_case(a_requires: bool, b_requires: bool) -> None:
    a = torch.randn(12, generator=gen, requires_grad=a_requires)
    b = torch.randn(12, generator=gen, requires_grad=b_requires)
    grad_out = torch.randn(12, generator=gen)
    label = f"(a_requires={a_requires}, b_requires={b_requires})"

    out = torch.ops.gradext.scalemix(a, b, 3)
    expected = torch.sin(a.detach()) + b.detach() * b.detach() + 3
    if not torch.allclose(out, expected):
        fail(f"forward result is wrong {label}")

    out.backward(grad_out)
    if a_requires:
        if a.grad is None:
            fail(
                f"a.grad is None {label} - is an autograd implementation "
                "registered for gradext::scalemix?"
            )
        if not torch.allclose(a.grad, grad_out * torch.cos(a.detach())):
            fail(f"a.grad does not match grad * cos(a) {label}")
    else:
        if a.grad is not None:
            fail(f"a.grad was populated even though a does not require grad {label}")
    if b_requires:
        if b.grad is None:
            fail(
                f"b.grad is None {label} - is an autograd implementation "
                "registered for gradext::scalemix?"
            )
        if not torch.allclose(b.grad, grad_out * 2 * b.detach()):
            fail(f"b.grad does not match grad * 2 * b {label}")
    else:
        if b.grad is not None:
            fail(f"b.grad was populated even though b does not require grad {label}")


run_case(True, False)
run_case(False, True)
run_case(True, True)

a64 = torch.randn(8, dtype=torch.float64, requires_grad=True)
b64 = torch.randn(8, dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(lambda x, y: torch.ops.gradext.scalemix(x, y, 3), (a64, b64))

print("OK: gradext::scalemix gradients verified for all requires_grad configs")
