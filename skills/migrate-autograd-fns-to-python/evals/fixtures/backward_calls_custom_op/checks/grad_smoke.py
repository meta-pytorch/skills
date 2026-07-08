# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test: gradext::mymul gradients, and the backward must go through
the extension's fused kernel gradext::mymul_backward_kernel (not
reimplemented tensor math).

Run from the project root: python3 checks/grad_smoke.py
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import gradext  # noqa: F401
import torch
from torch.utils._python_dispatch import TorchDispatchMode


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


class OpRecorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


gen = torch.Generator().manual_seed(3)
a = torch.randn(16, generator=gen, requires_grad=True)
b = torch.randn(16, generator=gen, requires_grad=True)
grad_out = torch.randn(16, generator=gen)

out = torch.ops.gradext.mymul(a, b)
if not torch.allclose(out, a.detach() * b.detach()):
    fail("forward result is wrong")

recorder = OpRecorder()
with recorder:
    out.backward(grad_out)

if a.grad is None or b.grad is None:
    fail(
        "backward produced no gradients for gradext::mymul - "
        "is an autograd implementation registered for the op?"
    )
if not torch.allclose(a.grad, grad_out * b.detach()):
    fail("a.grad does not match the analytic gradient grad * b")
if not torch.allclose(b.grad, grad_out * a.detach()):
    fail("b.grad does not match the analytic gradient grad * a")

kernel_calls = [op for op in recorder.ops if "mymul_backward_kernel" in op]
if len(kernel_calls) < 2:
    fail(
        "backward did not call gradext::mymul_backward_kernel for both "
        "gradients - the migrated backward must call the existing fused "
        f"kernel via torch.ops, not reimplement the math (saw {recorder.ops})"
    )

a64 = torch.randn(8, dtype=torch.float64, requires_grad=True)
b64 = torch.randn(8, dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(torch.ops.gradext.mymul, (a64, b64))

print("OK: gradext::mymul gradients verified and fused backward kernel was used")
