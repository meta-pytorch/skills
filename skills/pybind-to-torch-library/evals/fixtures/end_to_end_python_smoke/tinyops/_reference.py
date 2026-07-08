# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure-Python mirror of the C++ kernels, used by tests as a reference."""


def weighted_sum_ref(a, b, alpha):
    """Reference implementation of weighted_sum: a + alpha * b."""
    return a + alpha * b
