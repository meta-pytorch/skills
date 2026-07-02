# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Success contract: every source in the extension must compile with
# TORCH_TARGET_VERSION set to PyTorch 2.12 plus the stable-only guard.
set -euo pipefail

TORCH_INC=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'include'))")

for src in csrc/muladd.cpp csrc/relu_op.cpp; do
  g++ -fsyntax-only -std=c++17 \
    -DTORCH_TARGET_VERSION=0x020c000000000000 \
    -DTORCH_STABLE_ONLY \
    -I"$TORCH_INC" \
    "$src"
done

echo "compile check passed"
