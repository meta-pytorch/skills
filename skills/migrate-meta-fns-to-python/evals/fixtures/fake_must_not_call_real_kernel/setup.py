# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="statops",
    version="0.1.0",
    packages=["statops"],
    ext_modules=[
        CppExtension(
            name="statops._C",
            sources=[
                "csrc/kernels.cpp",
                "csrc/meta.cpp",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
