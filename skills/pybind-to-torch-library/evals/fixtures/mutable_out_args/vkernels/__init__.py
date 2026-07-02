# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import vkernels._C  # noqa: F401  — triggers static registration
from vkernels._custom_ops import rotary_embedding

__all__ = ["rotary_embedding"]
