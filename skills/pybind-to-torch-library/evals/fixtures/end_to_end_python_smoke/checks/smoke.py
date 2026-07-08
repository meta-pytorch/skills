# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""End-to-end smoke test for the pybind -> TORCH_LIBRARY migration.

Extracts the schema strings registered in csrc/scale.cpp, defines them via
torch.library under the `tinyops` namespace, registers the pure-Python
reference kernel as the CPU impl (standing in for the compiled C++ kernel),
then imports the package the way a user would and calls the public wrapper.
"""

import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))

src = Path("csrc/scale.cpp").read_text()
assert "PYBIND11_MODULE" not in src, "PYBIND11_MODULE still present in csrc/scale.cpp"

# Collect every schema string registered via m.def("...").
schemas = []
for match in re.finditer(r'm\.def\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)', src):
    literal = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(1)))
    if "->" in literal:
        schemas.append(literal)
assert schemas, "no TORCH_LIBRARY schema strings found in csrc/scale.cpp"

names = []
for schema in schemas:
    name = schema.split("(", 1)[0].strip().split("::")[-1]
    torch.library.define(f"tinyops::{name}", schema[schema.index("(") :])
    names.append(name)
assert "weighted_sum" in names, f"weighted_sum schema not registered; found {names}"

# Fake the compiled extension BEFORE importing the package: the loader in
# tinyops/__init__.py should find a _C*.so via glob (abi3 naming included) and
# hand it to torch.ops.load_library. The schemas above already stand in for
# the registrations the real .so would perform.
Path("tinyops/_C.abi3.so").touch()
torch.ops.load_library = lambda *args, **kwargs: None

from tinyops._reference import weighted_sum_ref  # noqa: E402

torch.library.impl("tinyops::weighted_sum", "CPU", weighted_sum_ref)

import tinyops  # noqa: E402

a = torch.randn(8)
b = torch.randn(8)
out = tinyops.weighted_sum(a, b, 0.25)
expected = weighted_sum_ref(a, b, 0.25)
torch.testing.assert_close(out, expected)
print("smoke ok: torch.ops.tinyops.weighted_sum resolved and matches reference")
