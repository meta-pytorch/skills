# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="gapext",
    version="0.1.0",
    ext_modules=[
        CppExtension(
            name="gapext._C",
            sources=["csrc/gapext.cpp"],
            extra_compile_args=["-O2"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
