# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Success contract: the migrated file must compile with TORCH_TARGET_VERSION
# set to PyTorch 2.12 plus the stable-only guard. Any remaining legacy include
# (<ATen/...>, <c10/...>, <torch/extension.h>, <torch/library.h>) is a hard
# compile error under these flags.
set -euo pipefail

TORCH_INC=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'include'))")

g++ -fsyntax-only -std=c++17 \
  -DTORCH_TARGET_VERSION=0x020c000000000000 \
  -DTORCH_STABLE_ONLY \
  -I"$TORCH_INC" \
  csrc/attention.cpp

echo "compile check passed"
