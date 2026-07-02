# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""gradext: toy PyTorch C++ extension used for the autograd migration exercise."""

# The real package would call torch.ops.load_library on the compiled
# extension here. In this exercise the compiled kernels are simulated in
# Python by _kernels, which must be imported first so the ops are defined.
from . import _kernels  # noqa: F401
