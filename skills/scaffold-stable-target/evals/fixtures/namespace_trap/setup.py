# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="extension_cpp",
    version="0.0.1",
    packages=["extension_cpp"],
    ext_modules=[
        cpp_extension.CppExtension(
            "extension_cpp._C",
            sources=[
                "extension_cpp/csrc/rmsnorm.cpp",
                "extension_cpp/csrc/quantize.cpp",
            ],
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
