# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from vkernels._C import rotary_embedding as _rotary_embedding_cpp


def rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox):
    """Apply rotary positional embedding to query and key in place.

    query and key are mutated; nothing is returned.
    """
    _rotary_embedding_cpp(positions, query, key, head_size, cos_sin_cache, is_neox)
