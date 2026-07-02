# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import mylib._C  # noqa: F401  — triggers static registration
from mylib._custom_ops import dequantize_per_block, quantize_per_block

__all__ = ["dequantize_per_block", "quantize_per_block"]
