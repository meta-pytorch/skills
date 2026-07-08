# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tinyops._C import weighted_sum as _weighted_sum_cpp


def weighted_sum(a, b, alpha):
    """Return a + alpha * b."""
    return _weighted_sum_cpp(a, b, alpha)
