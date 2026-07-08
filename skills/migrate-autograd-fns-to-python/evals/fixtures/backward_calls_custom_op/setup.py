# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="gradext",
    version="0.1.0",
    packages=["gradext"],
    ext_modules=[
        CppExtension(
            "gradext._C",
            [
                "csrc/mymul.cpp",
                "csrc/autograd.cpp",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
