# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="tinyops",
    version="0.3.0",
    packages=["tinyops"],
    ext_modules=[
        CppExtension(
            name="tinyops._C",
            sources=[
                "csrc/muladd.cpp",
                "csrc/relu_op.cpp",
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
