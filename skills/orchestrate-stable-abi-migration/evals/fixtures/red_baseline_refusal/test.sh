#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Project test suite. Used as the project's test_cmd.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m pytest tests/ -q
