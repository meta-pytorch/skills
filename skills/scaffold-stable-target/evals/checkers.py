# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Checkers for the scaffold-stable-target skill eval.

The skill's outcome is structural: setup.py must gain a second, ABI-stable
ext_modules entry (empty sources + TORCH_TARGET_VERSION) while the legacy
entry stays byte-for-byte equivalent, and no C++/CUDA file may move or change.
These checkers parse setup.py with `ast` (no torch import needed in-process)
and hash-compare the fixture sources.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from functools import cache
from pathlib import Path
from typing import Any

TARGET_HEX = "0x020c000000000000"
PY_LIMITED_HEX = "0x030a0000"

_UNRESOLVED = object()


# ---------------------------------------------------------------------------
# setup.py AST parsing
# ---------------------------------------------------------------------------


def _module_assignments(tree: ast.Module) -> dict[str, ast.expr]:
    assignments: dict[str, ast.expr] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and isinstance(node.target, ast.Name)
        ):
            assignments[node.target.id] = node.value
    return assignments


def _resolve(node: ast.expr | None, assignments: dict[str, ast.expr], depth: int = 0) -> Any:
    """Best-effort static evaluation of a setup.py expression."""
    if node is None or depth > 8:
        return _UNRESOLVED
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in assignments:
            return _resolve(assignments[node.id], assignments, depth + 1)
        return _UNRESOLVED
    if isinstance(node, ast.List | ast.Tuple):
        items = [_resolve(elt, assignments, depth + 1) for elt in node.elts]
        return _UNRESOLVED if any(i is _UNRESOLVED for i in items) else items
    if isinstance(node, ast.Dict):
        out: dict[Any, Any] = {}
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            key = _resolve(key_node, assignments, depth + 1)
            value = _resolve(value_node, assignments, depth + 1)
            if key is _UNRESOLVED or value is _UNRESOLVED:
                return _UNRESOLVED
            out[key] = value
        return out
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append(str(value.value))
            elif isinstance(value, ast.FormattedValue):
                inner = _resolve(value.value, assignments, depth + 1)
                if inner is _UNRESOLVED:
                    return _UNRESOLVED
                parts.append(str(inner))
            else:
                return _UNRESOLVED
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _resolve(node.left, assignments, depth + 1)
        right = _resolve(node.right, assignments, depth + 1)
        if left is _UNRESOLVED or right is _UNRESOLVED:
            return _UNRESOLVED
        try:
            return left + right
        except TypeError:
            return _UNRESOLVED
    return _UNRESOLVED


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _parse_setup_py(workdir: Path) -> dict[str, Any]:
    """Return {'ext_modules': [entry, ...], 'options': ...} or {'error': str}.

    Each entry: {'name', 'sources', 'extra_compile_args', 'py_limited_api'}.
    """
    setup_path = workdir / "setup.py"
    if not setup_path.exists():
        return {"error": "setup.py not found in workdir"}
    try:
        tree = ast.parse(setup_path.read_text())
    except SyntaxError as exc:
        return {"error": f"setup.py does not parse: {exc}"}
    assignments = _module_assignments(tree)

    setup_call = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "setup":
            setup_call = node
            break
    if setup_call is None:
        return {"error": "no setup(...) call found in setup.py"}

    kwargs = {kw.arg: kw.value for kw in setup_call.keywords if kw.arg is not None}
    ext_modules_node = kwargs.get("ext_modules")
    if ext_modules_node is None:
        return {"error": "setup(...) has no ext_modules kwarg"}
    if isinstance(ext_modules_node, ast.Name) and ext_modules_node.id in assignments:
        ext_modules_node = assignments[ext_modules_node.id]
    if not isinstance(ext_modules_node, ast.List | ast.Tuple):
        return {"error": "ext_modules is not a statically-visible list"}

    entries: list[dict[str, Any]] = []
    for elt in ext_modules_node.elts:
        if not isinstance(elt, ast.Call):
            return {"error": f"ext_modules element is not an Extension call: {ast.dump(elt)[:120]}"}
        ext_kwargs = {kw.arg: kw.value for kw in elt.keywords if kw.arg is not None}
        name_node = elt.args[0] if elt.args else ext_kwargs.get("name")
        sources_node = elt.args[1] if len(elt.args) > 1 else ext_kwargs.get("sources")
        entry = {
            "ctor": _call_name(elt),
            "name": _resolve(name_node, assignments),
            "sources": _resolve(sources_node, assignments),
            "extra_compile_args": (
                _resolve(ext_kwargs["extra_compile_args"], assignments)
                if "extra_compile_args" in ext_kwargs
                else None
            ),
            "has_extra_compile_args": "extra_compile_args" in ext_kwargs,
            "py_limited_api": (
                _resolve(ext_kwargs["py_limited_api"], assignments)
                if "py_limited_api" in ext_kwargs
                else None
            ),
        }
        entries.append(entry)

    options = _resolve(kwargs["options"], assignments) if "options" in kwargs else None
    return {"ext_modules": entries, "options": options}


# ---------------------------------------------------------------------------
# Shared assertions
# ---------------------------------------------------------------------------


def _flags_of(entry: dict[str, Any], key: str) -> list[str]:
    eca = entry.get("extra_compile_args")
    if not isinstance(eca, dict):
        return []
    flags = eca.get(key, [])
    return [str(f) for f in flags] if isinstance(flags, list) else []


def _has_define(flags: list[str], define: str, value: str | None = None) -> bool:
    expected = f"-d{define.lower()}" + (f"={value.lower()}" if value is not None else "")
    for flag in flags:
        normalized = flag.replace(" ", "").lower()
        if value is None:
            if normalized.startswith(expected + "=") or normalized == expected:
                return True
        elif normalized == expected:
            return True
    return False


def _mentions_define(entry: dict[str, Any], define: str) -> bool:
    for key in ("cxx", "nvcc"):
        for flag in _flags_of(entry, key):
            if define.lower() in flag.lower():
                return True
    return False


def _check_legacy_entry(
    entry: dict[str, Any], expected_name: str, expected_sources: list[str], errors: list[str]
) -> None:
    if entry["name"] != expected_name:
        errors.append(f"legacy entry renamed: expected {expected_name!r}, got {entry['name']!r}")
    sources = entry["sources"]
    if not isinstance(sources, list) or sorted(sources) != sorted(expected_sources):
        errors.append(
            f"legacy sources changed: expected {sorted(expected_sources)}, got {sources!r}"
        )
    if entry["has_extra_compile_args"]:
        errors.append(
            "legacy entry gained extra_compile_args (critical rule 5: TORCH_TARGET_VERSION / "
            f"Py_LIMITED_API belong only on the stable target); got {entry['extra_compile_args']!r}"
        )
    if entry["py_limited_api"]:
        errors.append("legacy entry must not set py_limited_api")


def _split_entries(parsed: dict[str, Any], errors: list[str]) -> tuple[Any, Any]:
    """Return (legacy_entry, stable_entry) from a parsed setup.py."""
    entries = parsed["ext_modules"]
    if len(entries) != 2:
        errors.append(f"expected exactly 2 ext_modules entries, found {len(entries)}")
        return None, None
    stable = [e for e in entries if isinstance(e["name"], str) and "_stable" in e["name"]]
    legacy = [e for e in entries if e not in stable]
    if len(stable) != 1:
        names = [e["name"] for e in entries]
        errors.append(f"expected exactly one '*_stable' ext_modules entry; names={names!r}")
        return None, None
    return legacy[0], stable[0]


def _check_hashes(workdir: Path, errors: list[str]) -> None:
    """Verify files listed in checks/orig_hashes.json are unmodified (no moves/rewrites)."""
    hashes_path = workdir / "checks" / "orig_hashes.json"
    if not hashes_path.exists():
        errors.append("checks/orig_hashes.json missing from workdir (fixture broken?)")
        return
    for rel_path, expected_sha in json.loads(hashes_path.read_text()).items():
        target = workdir / rel_path
        if not target.exists():
            errors.append(f"{rel_path} was moved or deleted (critical rule 1: no files move)")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected_sha:
            errors.append(f"{rel_path} was modified (this skill is setup-only; no rewrites)")


def _find_stub(workdir: Path, sources: list[str]) -> Path | None:
    for src in sources:
        if str(src).endswith((".cpp", ".cc", ".cu")):
            candidate = workdir / str(src)
            if candidate.exists():
                return candidate
    return None


def _strip_cpp_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _check_stub_content(stub: Path, lib_namespace: str, errors: list[str]) -> None:
    text = _strip_cpp_comments(stub.read_text())
    macro_uses = re.findall(
        r"STABLE_TORCH_LIBRARY(?:_IMPL|_FRAGMENT)?\s*\(\s*([A-Za-z0-9_]+)", text
    )
    if not macro_uses:
        errors.append(f"stub {stub.name} has no STABLE_TORCH_LIBRARY(...) block")
    for ns in macro_uses:
        if ns != lib_namespace:
            errors.append(
                f"stub {stub.name} registers into torch.ops namespace {ns!r}; must be "
                f"{lib_namespace!r} (critical rules 2-3: same namespace, never a _stable suffix)"
            )
    # A stub compiled with TORCH_TARGET_VERSION cannot use legacy registration or headers.
    if re.search(r"(?<!STABLE_)TORCH_LIBRARY(?:_IMPL|_FRAGMENT)?\s*\(", text):
        errors.append(f"stub {stub.name} uses legacy TORCH_LIBRARY; must use STABLE_TORCH_LIBRARY")
    for legacy_header in ("torch/extension.h", "ATen/", "c10/", "torch/library.h"):
        if f"<{legacy_header}" in text or f'"{legacy_header}' in text:
            errors.append(f"stub {stub.name} includes legacy header {legacy_header!r}")


@cache
def _torch_python() -> str | None:
    """Find a python3 with torch installed (the harness venv does not have it)."""
    candidates: list[Path] = [Path(sys.executable)]
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry:
            candidates.append(Path(entry) / "python3")
    for cand in candidates:
        if not cand.is_file():
            continue
        probe = subprocess.run(
            [
                str(cand),
                "-c",
                "import importlib.util as u, sys; sys.exit(0 if u.find_spec('torch') else 1)",
            ],
            capture_output=True,
            timeout=60,
            check=False,
        )
        if probe.returncode == 0:
            return str(cand)
    return None


def _compile_stub(stub: Path, errors: list[str]) -> None:
    python = _torch_python()
    if python is None:
        errors.append("no python3 with torch found on PATH; cannot compile-check the stub")
        return
    try:
        torch_inc = subprocess.run(
            [
                python,
                "-c",
                "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'include'))",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        ).stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        errors.append(f"could not locate torch include dir: {exc}")
        return
    result = subprocess.run(
        [
            "g++",
            "-fsyntax-only",
            "-std=c++17",
            f"-DTORCH_TARGET_VERSION={TARGET_HEX}",
            f"-I{torch_inc}",
            str(stub),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if result.returncode != 0:
        errors.append(
            f"stub {stub.name} does not compile under -DTORCH_TARGET_VERSION={TARGET_HEX}: "
            f"{result.stderr[-800:]}"
        )


def _check_stable_sources(
    workdir: Path, stable: dict[str, Any], lib_namespace: str, errors: list[str]
) -> None:
    """Stable sources must be [] or a single trivial stub (which must be valid)."""
    sources = stable["sources"]
    if sources is _UNRESOLVED or not isinstance(sources, list):
        errors.append(f"stable entry sources are not statically resolvable: {sources!r}")
    elif len(sources) == 0:
        pass  # empty sources is the default correct scaffold
    elif len(sources) == 1:
        stub = _find_stub(workdir, sources)
        if stub is None:
            errors.append(f"stable entry references stub {sources[0]!r} but the file is missing")
        else:
            _check_stub_content(stub, lib_namespace, errors)
    else:
        errors.append(
            f"stable entry must start with sources=[] (or one trivial stub), got {sources!r} "
            "(moving real files is migrate-to-stable-apis' job)"
        )


def _result(errors: list[str], ok_details: str) -> dict[str, Any]:
    if errors:
        return {"passed": False, "details": "; ".join(errors)}
    return {"passed": True, "details": ok_details}


# ---------------------------------------------------------------------------
# Per-case checkers
# ---------------------------------------------------------------------------

_CUDA_FIXTURE_SOURCES = [
    "extension_cpp/csrc/muladd.cu",
    "extension_cpp/csrc/attention.cu",
]
_CPU_FIXTURE_SOURCES = [
    "extension_cpp/csrc/rmsnorm.cpp",
    "extension_cpp/csrc/quantize.cpp",
]


def _check_scaffold_common(
    workdir: Path, expected_sources: list[str]
) -> tuple[list[str], dict[str, Any] | None, dict[str, Any] | None]:
    errors: list[str] = []
    parsed = _parse_setup_py(workdir)
    if "error" in parsed:
        return [parsed["error"]], None, None
    legacy, stable = _split_entries(parsed, errors)
    if legacy is None or stable is None:
        return errors, None, parsed
    _check_legacy_entry(legacy, "extension_cpp._C", expected_sources, errors)
    _check_stable_sources(workdir, stable, "extension_cpp", errors)
    for key in ("cxx", "nvcc"):
        if not _has_define(_flags_of(stable, key), "TORCH_TARGET_VERSION", TARGET_HEX):
            errors.append(
                f"stable entry missing -DTORCH_TARGET_VERSION={TARGET_HEX} in "
                f"extra_compile_args[{key!r}]; got {_flags_of(stable, key)!r}"
            )
    _check_hashes(workdir, errors)
    return errors, stable, parsed


def check_basic_scaffold(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict[str, Any]:
    """Case 1: parallel stable target, no Py_LIMITED_API, nothing moved."""
    errors, stable, parsed = _check_scaffold_common(workdir, _CUDA_FIXTURE_SOURCES)
    if stable is not None:
        if _mentions_define(stable, "Py_LIMITED_API"):
            errors.append("Py_LIMITED_API was enabled but the user explicitly declined it")
        if stable["py_limited_api"]:
            errors.append("py_limited_api=True set but the user explicitly declined it")
    if parsed is not None and isinstance(parsed.get("options"), dict):
        bdist = parsed["options"].get("bdist_wheel", {})
        if isinstance(bdist, dict) and bdist.get("py_limited_api"):
            errors.append("bdist_wheel py_limited_api set but the user explicitly declined it")
    return _result(errors, "setup.py has correct parallel stable target; sources untouched")


def check_py_limited_api(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict[str, Any]:
    """Case 2: same scaffold plus the full Py_LIMITED_API triple (floor cp310)."""
    errors, stable, parsed = _check_scaffold_common(workdir, _CUDA_FIXTURE_SOURCES)
    if stable is not None:
        if not _has_define(_flags_of(stable, "cxx"), "Py_LIMITED_API", PY_LIMITED_HEX):
            errors.append(
                f"stable entry missing -DPy_LIMITED_API={PY_LIMITED_HEX} in "
                f"extra_compile_args['cxx']; got {_flags_of(stable, 'cxx')!r}"
            )
        if stable["py_limited_api"] is not True:
            errors.append("stable entry must set py_limited_api=True")
    if parsed is not None:
        options = parsed.get("options")
        bdist = options.get("bdist_wheel") if isinstance(options, dict) else None
        tag = bdist.get("py_limited_api") if isinstance(bdist, dict) else None
        if tag != "cp310":
            errors.append(
                f"setup(options=...) must set bdist_wheel py_limited_api='cp310'; got {tag!r}"
            )
    return _result(errors, "stable target correctly enables Py_LIMITED_API with cp310 floor")


def check_namespace_trap(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict[str, Any]:
    """Case 3: stub is required; it must use the SAME torch.ops namespace and compile."""
    errors: list[str] = []
    parsed = _parse_setup_py(workdir)
    if "error" in parsed:
        return {"passed": False, "details": parsed["error"]}
    legacy, stable = _split_entries(parsed, errors)
    if legacy is not None and stable is not None:
        _check_legacy_entry(legacy, "extension_cpp._C", _CPU_FIXTURE_SOURCES, errors)
        for key in ("cxx",):
            if not _has_define(_flags_of(stable, key), "TORCH_TARGET_VERSION", TARGET_HEX):
                errors.append(
                    f"stable entry missing -DTORCH_TARGET_VERSION={TARGET_HEX} in "
                    f"extra_compile_args[{key!r}]; got {_flags_of(stable, key)!r}"
                )
        sources = stable["sources"]
        if not isinstance(sources, list) or len(sources) != 1:
            errors.append(
                "this build rejects empty sources: the stable entry needs exactly one trivial "
                f"stub source; got {sources!r}"
            )
        else:
            stub = _find_stub(workdir, sources)
            if stub is None:
                errors.append(f"stub {sources[0]!r} referenced by stable entry does not exist")
            else:
                _check_stub_content(stub, "extension_cpp", errors)
                if not errors:
                    _compile_stub(stub, errors)
    # No STABLE_TORCH_LIBRARY anywhere in the workdir may use a _stable-suffixed namespace.
    for pattern in ("*.cpp", "*.cc", "*.cu", "*.h", "*.cuh"):
        for path in workdir.rglob(pattern):
            for ns in re.findall(
                r"STABLE_TORCH_LIBRARY(?:_IMPL|_FRAGMENT)?\s*\(\s*([A-Za-z0-9_]+)",
                _strip_cpp_comments(path.read_text(errors="replace")),
            ):
                if ns.endswith("_stable"):
                    errors.append(
                        f"{path.relative_to(workdir)} registers into {ns!r} — never suffix the "
                        "torch.ops namespace with _stable (critical rule 3)"
                    )
    _check_hashes(workdir, errors)
    return _result(errors, "stub uses shared extension_cpp namespace and compiles under target")


def check_one_shot_plan_trap(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict[str, Any]:
    """Case 4: plan says one-shot — the skill must be skipped; nothing may be scaffolded."""
    errors: list[str] = []
    _check_hashes(workdir, errors)  # covers setup.py and extension_cpp/__init__.py
    setup_path = workdir / "setup.py"
    if setup_path.exists() and "_stable" in setup_path.read_text():
        errors.append(
            "setup.py gained a _stable target, but the plan records a one-shot cadence and "
            "omits scaffold-stable-target from dispatch_order — the skill must be skipped"
        )
    return _result(errors, "correctly declined to scaffold: setup.py and __init__.py untouched")
