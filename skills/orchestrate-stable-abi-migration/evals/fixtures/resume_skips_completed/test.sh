#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Project test suite: validates every op schema registered in csrc/ parses
# and registers cleanly via torch.library. Used as the project's test_cmd.
set -euo pipefail
cd "$(dirname "$0")"
python3 checks/schema_check.py
echo "TESTS OK"
