# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import fastops._C  # noqa: F401  — triggers static registration
from fastops._custom_ops import rms_norm, silu_mul

__all__ = ["rms_norm", "silu_mul"]
