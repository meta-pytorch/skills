# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom checkers for the migrate-meta-fns-to-python eval."""

import re
from pathlib import Path

_META_IMPL_RE = re.compile(r"TORCH_LIBRARY_IMPL\s*\(\s*\w+\s*,\s*Meta\s*,")


def _cpp_meta_impl_gone(workdir: Path) -> tuple[bool, str]:
    """No TORCH_LIBRARY_IMPL(..., Meta, ...) block may remain in any C++ file."""
    offenders = [
        str(path.relative_to(workdir))
        for path in workdir.rglob("*.cpp")
        if _META_IMPL_RE.search(path.read_text(errors="replace"))
    ]
    if offenders:
        return False, f"C++ Meta impl block(s) still present in: {', '.join(offenders)}"
    return True, "no TORCH_LIBRARY_IMPL(..., Meta, ...) blocks remain"


def _meta_cpp_unregistered(workdir: Path) -> tuple[bool, str]:
    """setup.py must no longer list csrc/meta.cpp as a source."""
    setup_py = workdir / "setup.py"
    if not setup_py.exists():
        return False, "setup.py is missing from the workdir"
    if "meta.cpp" in setup_py.read_text(errors="replace"):
        return False, "setup.py still lists meta.cpp as an extension source"
    return True, "meta.cpp removed from setup.py sources"


def _register_fake_registrations(workdir: Path, package: str) -> dict[str, int]:
    """Count register_fake registrations per op name across the package."""
    counts: dict[str, int] = {}
    pattern = re.compile(
        r"(?:torch\.library\.)?register_fake\s*\(\s*[\"']" + package + r"::(\w+)[\"']"
    )
    for path in (workdir / package).rglob("*.py"):
        for match in pattern.finditer(path.read_text(errors="replace")):
            op = match.group(1)
            counts[op] = counts.get(op, 0) + 1
    return counts


def _fake_registered_and_imported(workdir: Path, package: str, ops: list[str]) -> tuple[bool, str]:
    """Each op has a Python register_fake, reachable from package import."""
    counts = _register_fake_registrations(workdir, package)
    missing = [op for op in ops if counts.get(op, 0) == 0]
    if missing:
        return False, f"no register_fake found for: {', '.join(missing)}"

    # The registration module must actually load at import time. Rather than
    # tracing imports statically, require that the string "register_fake"
    # appears in the package and that __init__.py imports every non-private
    # sibling module that contains it.
    init = workdir / package / "__init__.py"
    if not init.exists():
        return False, f"{package}/__init__.py is missing"
    init_text = init.read_text(errors="replace")
    unimported = []
    for path in (workdir / package).glob("*.py"):
        if path.name == "__init__.py":
            continue
        if "register_fake" not in path.read_text(errors="replace"):
            continue
        module = path.stem
        if not re.search(rf"(?:from|import)[^\n]*\b{re.escape(module)}\b", init_text):
            unimported.append(path.name)
    if unimported:
        return False, f"__init__.py does not import fake-registration module(s): {unimported}"
    return True, f"register_fake present and imported for: {', '.join(ops)}"


def _check_migration(
    workdir: Path, package: str, ops: list[str], *, max_regs_per_op: int = 1
) -> dict:
    failures = []
    details = []
    for passed, detail in (
        _cpp_meta_impl_gone(workdir),
        _meta_cpp_unregistered(workdir),
        _fake_registered_and_imported(workdir, package, ops),
    ):
        details.append(detail)
        if not passed:
            failures.append(detail)

    counts = _register_fake_registrations(workdir, package)
    for op in ops:
        if counts.get(op, 0) > max_regs_per_op:
            failures.append(
                f"op {package}::{op} has {counts[op]} register_fake registrations "
                f"(expected at most {max_regs_per_op})"
            )

    if failures:
        return {"passed": False, "details": "; ".join(failures)}
    return {"passed": True, "details": "; ".join(details)}


def check_basic_migration(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    return _check_migration(workdir, "extpkg", ["mymuladd"])


def check_symint_migration(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    result = _check_migration(workdir, "windowops", ["window_avg", "pack_pair"])
    if not result["passed"]:
        return result

    # Secondary signal for critical rule 3: no int() casts on shape values
    # inside fake-registration modules (the runtime dynamic-shape check in
    # checks/dynamic_smoke.py is the primary signal).
    int_cast = re.compile(r"int\s*\(\s*\w+\.(?:shape|size)")
    offenders = [
        str(path.relative_to(workdir))
        for path in (workdir / "windowops").rglob("*.py")
        if "register_fake" in path.read_text(errors="replace")
        and int_cast.search(path.read_text(errors="replace"))
    ]
    if offenders:
        return {
            "passed": False,
            "details": f"fake kernel casts SymInt shapes via int(...) in: {', '.join(offenders)}",
        }
    return result


def check_purity_migration(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    return _check_migration(workdir, "statops", ["row_l2norm"])


def check_existing_fake_migration(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    # colwise_scale already had a Python fake: there must still be EXACTLY one
    # registration for it (no duplicate added), and exactly one for tile_rows.
    return _check_migration(workdir, "gridops", ["colwise_scale", "tile_rows"])
