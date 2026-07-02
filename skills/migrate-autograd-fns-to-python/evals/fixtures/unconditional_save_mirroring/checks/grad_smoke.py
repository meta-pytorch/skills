# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke test: gradext::mytanhmul gradients, mirroring the C++ save set
(a, b, AND the forward output).

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


gen = torch.Generator().manual_seed(11)
a = torch.randn(16, generator=gen, requires_grad=True)
b = torch.randn(16, generator=gen, requires_grad=True)
grad_out = torch.randn(16, generator=gen)

out = torch.ops.gradext.mytanhmul(a, b)
expected = torch.tanh(a.detach() * b.detach())
if not torch.allclose(out, expected):
    fail("forward result is wrong")

if out.grad_fn is None:
    fail("output has no grad_fn - is an autograd implementation registered for gradext::mytanhmul?")

# The C++ Function saved {a, b, output}. The Python setup_context must
# mirror that save set exactly (fidelity first; optimize later).
try:
    saved = out.grad_fn.saved_tensors
except AttributeError:
    fail(
        "grad_fn has no saved_tensors - gradext::mytanhmul is not backed by "
        "a Python register_autograd registration"
    )
if len(saved) != 3:
    fail(
        f"expected 3 saved tensors (a, b, output) to mirror the C++ "
        f"save_for_backward, found {len(saved)}"
    )
if not any(torch.equal(t, out.detach()) for t in saved):
    fail(
        "the forward output was not saved for backward - the C++ version "
        "saved it to reuse in the tanh gradient"
    )

out.backward(grad_out)
if a.grad is None or b.grad is None:
    fail("backward produced no gradients for gradext::mytanhmul")
grad_u = grad_out * (1 - expected * expected)
if not torch.allclose(a.grad, grad_u * b.detach()):
    fail("a.grad does not match grad * (1 - out^2) * b")
if not torch.allclose(b.grad, grad_u * a.detach()):
    fail("b.grad does not match grad * (1 - out^2) * a")

a64 = torch.randn(8, dtype=torch.float64, requires_grad=True)
b64 = torch.randn(8, dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(torch.ops.gradext.mytanhmul, (a64, b64))

print("OK: gradext::mytanhmul gradients verified; saved set mirrors the C++ version")
