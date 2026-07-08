# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python simulation of the compiled C++ kernels in csrc/.

In the real project the schemas and CPU kernels below are provided by
TORCH_LIBRARY / TORCH_LIBRARY_IMPL in csrc/mymul.cpp and loaded from the
compiled extension. The extension cannot be compiled in this environment,
so this module mirrors them with torch.library to keep the package
runnable. Treat torch.ops.gradext.* as the compiled extension: do not
change the schemas or kernel math in this file.
"""

import torch

torch.library.define("gradext::mymul", "(Tensor a, Tensor b) -> Tensor")
torch.library.define("gradext::mymul_backward_kernel", "(Tensor grad, Tensor other) -> Tensor")


@torch.library.impl("gradext::mymul", "CPU")
def _mymul_cpu(a, b):
    # A compiled kernel is opaque to autograd; simulate that with no_grad.
    with torch.no_grad():
        return a * b


@torch.library.impl("gradext::mymul_backward_kernel", "CPU")
def _mymul_backward_kernel_cpu(grad, other):
    # Hand-tuned fused backward kernel in the real extension.
    with torch.no_grad():
        return grad * other
