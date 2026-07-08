# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from mylib._C import dequantize_per_block as _dequantize_per_block_cpp
from mylib._C import quantize_per_block as _quantize_per_block_cpp


def quantize_per_block(input, block_size, eps):
    """Blockwise int8 quantization. Returns (values, per-block scales)."""
    return _quantize_per_block_cpp(input, block_size, eps)


def dequantize_per_block(qdata, scale, block_size, out_dtype):
    """Reverses quantize_per_block. out_dtype is "float32" or "float64"."""
    return _dequantize_per_block_cpp(qdata, scale, block_size, out_dtype)
