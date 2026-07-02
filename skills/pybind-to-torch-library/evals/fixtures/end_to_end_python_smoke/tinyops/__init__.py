# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tinyops._C  # noqa: F401  — triggers static registration
from tinyops._custom_ops import weighted_sum

__all__ = ["weighted_sum"]
