# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="tinyops",
    version="1.0.0",
    ext_modules=[
        CppExtension(
            name="tinyops._C",
            sources=["csrc/ops.cpp", "csrc/extra.cpp"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
