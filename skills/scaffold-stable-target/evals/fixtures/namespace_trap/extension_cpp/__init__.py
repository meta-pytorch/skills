# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import torch

_so_files = list(Path(__file__).parent.glob("_C*.so"))
assert len(_so_files) == 1, f"Expected exactly one extension .so, found {len(_so_files)}"
torch.ops.load_library(str(_so_files[0]))
