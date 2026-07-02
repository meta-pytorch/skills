#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Syntax-checks csrc/pos_encoding.cpp against regular (non-stable-ABI) torch
# headers. TORCH_LIBRARY / TORCH_LIBRARY_IMPL registrations must compile here.
set -euo pipefail

TORCH_INC=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'include'))")
PY_INC=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")

g++ -fsyntax-only -std=c++17 -w \
    -I"$TORCH_INC" \
    -I"$TORCH_INC/torch/csrc/api/include" \
    -I"$PY_INC" \
    csrc/pos_encoding.cpp

echo "compile check passed"
