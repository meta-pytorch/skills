# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fastops._C import rms_norm as _rms_norm_cpp
from fastops._C import silu_mul as _silu_mul_cpp


def rms_norm(input, weight, eps):
    """Root-mean-square layer norm over the last dimension."""
    return _rms_norm_cpp(input, weight, eps)


def silu_mul(gate, up):
    """SiLU(gate) * up, the SwiGLU gated activation."""
    return _silu_mul_cpp(gate, up)
