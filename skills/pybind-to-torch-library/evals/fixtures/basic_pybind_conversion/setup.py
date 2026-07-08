# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="mylib",
    packages=["mylib"],
    ext_modules=[
        cpp_extension.CppExtension(
            "mylib._C",
            sources=["csrc/quantize.cpp"],
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
