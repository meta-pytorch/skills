#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Migration success contract: every C++ source must compile against ONLY the
# stable ABI surface. TORCH_STABLE_ONLY makes any legacy header include
# (<ATen/...>, <c10/...>, <torch/extension.h>, <torch/library.h>) a hard
# compile error, and TORCH_TARGET_VERSION pins the stable surface to
# PyTorch 2.12. Passing proves the migration is complete and syntactically
# valid.
set -euo pipefail
cd "$(dirname "$0")/.."

TORCH_INC=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'include'))")

for f in csrc/*.cpp; do
    echo "stable-abi verifying $f"
    g++ -fsyntax-only -w -std=c++17 \
        -DTORCH_TARGET_VERSION=0x020c000000000000 \
        -DTORCH_STABLE_ONLY \
        -I"$TORCH_INC" "$f"
done
echo "STABLE ABI VERIFY OK"
