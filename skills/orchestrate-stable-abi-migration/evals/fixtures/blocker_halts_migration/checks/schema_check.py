# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Extracts every m.def("...") op schema from csrc/*.cpp and validates that it
# parses and registers cleanly via torch.library.define. Works both before and
# after the stable-ABI migration because op schema strings are unchanged.
import pathlib
import re
import sys
import uuid

import torch

repo_root = pathlib.Path(__file__).resolve().parent.parent
schemas = []
for src in sorted((repo_root / "csrc").glob("*.cpp")):
    schemas += re.findall(r'm\.def\(\s*"([^"]+)"', src.read_text())

if not schemas:
    print("FAIL: no m.def op schemas found in csrc/*.cpp")
    sys.exit(1)

ns = "schemacheck_" + uuid.uuid4().hex[:8]
for schema in schemas:
    name, _, sig = schema.partition("(")
    torch.library.define(f"{ns}::{name.strip()}", "(" + sig)
    print(f"schema OK: {schema}")
print(f"validated {len(schemas)} op schema(s)")
