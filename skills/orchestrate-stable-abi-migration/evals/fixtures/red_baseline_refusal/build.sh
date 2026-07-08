#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Fast build check for this repo: syntax-check every C++ source against the
# installed torch headers. Used as the project's build_cmd.
set -euo pipefail
cd "$(dirname "$0")"

TORCH_INC=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'include'))")

for f in csrc/*.cpp; do
    echo "build-checking $f"
    g++ -fsyntax-only -w -std=c++17 -I"$TORCH_INC" "$f"
done
echo "BUILD OK"
