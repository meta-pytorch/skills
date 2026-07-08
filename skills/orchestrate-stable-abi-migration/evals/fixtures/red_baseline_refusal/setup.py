# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="scaleops",
    version="0.1.0",
    ext_modules=[
        CppExtension(
            name="scaleops._C",
            sources=["csrc/scale.cpp"],
            extra_compile_args=["-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
