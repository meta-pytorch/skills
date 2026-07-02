# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public API for mylib. Documented entry points; downstream projects import
`fused_op` from `mylib._C` directly, as shown in the project README."""

from mylib._C import fused_op


def fused_apply(a, b, c, *, iterations=1):
    """Apply fused_op `iterations` times."""
    out = fused_op(a, b, c)
    for _ in range(iterations - 1):
        out = fused_op(out, b, c)
    return out
