# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Success contract: the migrated CUDA file must compile with
# TORCH_TARGET_VERSION set to PyTorch 2.12 plus the stable-only guard.
# Any remaining legacy include (<ATen/...>, <c10/...>) is a hard compile
# error under these flags.
set -euo pipefail

TORCH_INC=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'include'))")

nvcc -std=c++17 --compiler-options -fsyntax-only \
  -DTORCH_TARGET_VERSION=0x020c000000000000 \
  -DTORCH_STABLE_ONLY \
  -DUSE_CUDA \
  -I"$TORCH_INC" \
  -c csrc/rope_kernel.cu -o /dev/null

echo "compile check passed"
