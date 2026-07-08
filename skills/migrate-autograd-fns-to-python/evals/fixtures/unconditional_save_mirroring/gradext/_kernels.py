# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python simulation of the compiled C++ kernels in csrc/.

In the real project the schema and CPU kernel below are provided by
TORCH_LIBRARY / TORCH_LIBRARY_IMPL in csrc/tanhmul.cpp and loaded from
the compiled extension. The extension cannot be compiled in this
environment, so this module mirrors them with torch.library to keep the
package runnable. Treat torch.ops.gradext.* as the compiled extension:
do not change the schemas or kernel math in this file.
"""

import torch

torch.library.define("gradext::mytanhmul", "(Tensor a, Tensor b) -> Tensor")


@torch.library.impl("gradext::mytanhmul", "CPU")
def _mytanhmul_cpu(a, b):
    # A compiled kernel is opaque to autograd; simulate that with no_grad.
    with torch.no_grad():
        return torch.tanh(a * b)
