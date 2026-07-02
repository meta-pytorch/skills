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
        cpp_extension.CUDAExtension(
            "extension_cpp._C",
            sources=[
                "extension_cpp/csrc/muladd.cu",
                "extension_cpp/csrc/attention.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
