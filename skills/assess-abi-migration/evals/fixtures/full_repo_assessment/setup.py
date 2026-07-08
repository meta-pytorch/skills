# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mixedext",
    version="0.3.1",
    packages=["mixedext"],
    ext_modules=[
        CUDAExtension(
            name="mixedext._C",
            sources=[
                "csrc/bindings.cpp",
                "csrc/meta.cpp",
                "csrc/autograd.cpp",
                "csrc/kernels.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-DUSE_CUDA"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
