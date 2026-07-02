# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# In the real package this would be `torch.ops.load_library(...)` on the
# compiled extension built from csrc/. For local development and testing we
# import the Python simulation of the compiled kernels instead.
from . import _kernels  # noqa: F401
from ._kernels import kernel_launch_count  # noqa: F401
