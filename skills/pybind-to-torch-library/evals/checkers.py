# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom checkers for the pybind-to-torch-library skill evals.

Checkers run in the harness venv (no torch); anything needing torch shells
out to the system ``python3`` (which has torch) via subprocess.
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
from pathlib import Path

_SCHEMA_STRING_RE = re.compile(r'm\.def\(\s*((?:"(?:[^"\\]|\\.)*"\s*)+)')
_STRING_LITERAL_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


@functools.cache
def _torch_python() -> str:
    """Locate a python3 that can import torch.

    Checkers run inside the harness venv (no torch), so a bare ``python3``
    may resolve to the wrong interpreter. Probe PATH entries, skipping the
    active venv, and fall back to plain ``python3``.
    """
    venv = os.environ.get("VIRTUAL_ENV")
    candidates = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if venv and entry.startswith(venv):
            continue
        exe = Path(entry) / "python3"
        if exe.is_file() and os.access(exe, os.X_OK):
            candidates.append(str(exe))
    candidates.append("python3")
    for exe in candidates:
        try:
            probe = subprocess.run([exe, "-c", "import torch"], capture_output=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if probe.returncode == 0:
            return exe
    return "python3"


def _strip_string_literals(cpp_text: str) -> str:
    """Remove double-quoted string literals so signature regexes don't match
    schema strings."""
    return re.sub(r'"(?:[^"\\]|\\.)*"', '""', cpp_text)


def _extract_schemas(cpp_text: str) -> list[str]:
    """Return every schema string registered via ``m.def("...")``.

    Handles adjacent string-literal concatenation. Non-schema ``m.def`` calls
    (pybind style, no ``->``) are ignored.
    """
    schemas = []
    for match in _SCHEMA_STRING_RE.finditer(cpp_text):
        literal = "".join(_STRING_LITERAL_RE.findall(match.group(1)))
        if "->" in literal:
            schemas.append(literal)
    return schemas


def _validate_schemas_via_torch(schemas: list[str]) -> tuple[bool, str]:
    """Feed each schema to torch.library.define in a subprocess.

    Uses a throwaway namespace so repeated runs don't collide. Returns
    (all_valid, details).
    """
    if not schemas:
        return False, "no schema strings found in any m.def(...) call"
    script_lines = ["import torch"]
    for i, schema in enumerate(schemas):
        paren = schema.index("(")
        name = schema[:paren].strip().split("::")[-1]
        args_part = schema[paren:]
        if not name.isidentifier():
            return False, f"schema has invalid op name {name!r}: {schema}"
        script_lines.append(f"torch.library.define({f'testns{i}::{name}'!r}, {args_part!r})")
    result = subprocess.run(
        [_torch_python(), "-c", "\n".join(script_lines)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return False, f"torch.library.define rejected a schema:\n{result.stderr[-2000:]}"
    return True, f"all {len(schemas)} schemas accepted by torch.library.define"


def _run_schema_probe(schemas: list[str], probe: str) -> tuple[bool, str]:
    """Run a torch subprocess with SCHEMAS bound to the extracted schemas."""
    script = f"SCHEMAS = {schemas!r}\nimport torch\n{probe}"
    result = subprocess.run(
        [_torch_python(), "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return False, result.stderr[-2000:]
    return True, result.stdout.strip()


def check_basic_conversion(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    """basic_pybind_conversion: full structural + schema-validity check."""
    cpp_path = workdir / "csrc" / "quantize.cpp"
    if not cpp_path.is_file():
        return {"passed": False, "details": "csrc/quantize.cpp is missing"}
    cpp = cpp_path.read_text()

    problems = []
    if "PYBIND11_MODULE" in cpp:
        problems.append("PYBIND11_MODULE still present in csrc/quantize.cpp")
    if not re.search(r"\bTORCH_LIBRARY(_FRAGMENT)?\s*\(\s*mylib\b", cpp):
        problems.append("no TORCH_LIBRARY(mylib, ...) block in csrc/quantize.cpp")
    if not re.search(r"\bTORCH_LIBRARY_IMPL\s*\(\s*mylib\b", cpp):
        problems.append("no TORCH_LIBRARY_IMPL(mylib, ...) block in csrc/quantize.cpp")
    code_only = _strip_string_literals(cpp)
    if re.search(r"\bint\s+block_size\b", code_only):
        problems.append("C++ signature still uses `int block_size` (must be int64_t)")
    if re.search(r"\bfloat\s+eps\b", code_only):
        problems.append("C++ signature still uses `float eps` (must be double)")

    schemas = _extract_schemas(cpp)
    schema_names = {s.split("(", 1)[0].strip().split("::")[-1] for s in schemas}
    for op in ("quantize_per_block", "dequantize_per_block"):
        if op not in schema_names:
            problems.append(f"no schema registered for {op} (found: {sorted(schema_names)})")
    valid, details = _validate_schemas_via_torch(schemas)
    if not valid:
        problems.append(details)

    # Schemas must use dispatcher types, not C++ spellings, and stay honest.
    for schema in schemas:
        args = schema[schema.index("(") :]
        if re.search(r"\bint64_t\b|\bdouble\b|\bstd::string\b", args):
            problems.append(f"schema uses C++ type spellings instead of schema types: {schema}")
        if schema.startswith("quantize_per_block") and not re.search(
            r"->\s*\(\s*Tensor\s*,\s*Tensor\s*\)", schema
        ):
            problems.append(f"quantize_per_block must return (Tensor, Tensor): {schema}")

    # Python side: wrapper goes through torch.ops.mylib, no _C imports remain.
    wrapper_path = workdir / "mylib" / "_custom_ops.py"
    if not wrapper_path.is_file():
        problems.append("mylib/_custom_ops.py is missing")
    else:
        wrapper = wrapper_path.read_text()
        if "torch.ops.mylib." not in wrapper:
            problems.append("mylib/_custom_ops.py does not call torch.ops.mylib.<op>")
        if re.search(r"from\s+mylib\._C\s+import|import\s+mylib\._C", wrapper):
            problems.append("mylib/_custom_ops.py still imports from mylib._C")
        for fn in ("def quantize_per_block(", "def dequantize_per_block("):
            if fn not in wrapper:
                problems.append(f"public wrapper `{fn}...)` was renamed or removed")

    init_path = workdir / "mylib" / "__init__.py"
    if not init_path.is_file():
        problems.append("mylib/__init__.py is missing")
    else:
        init = init_path.read_text()
        if "torch.ops.load_library" not in init:
            problems.append("mylib/__init__.py does not call torch.ops.load_library")
        if not re.search(r"_C\*?\.?(abi3\.)?so|_C\*", init):
            problems.append("mylib/__init__.py has no _C*.so glob pattern")
        if re.search(r"^\s*import\s+mylib\._C\b", init, re.MULTILINE):
            problems.append("side-effect `import mylib._C` still present in __init__.py")

    if problems:
        return {"passed": False, "details": "; ".join(problems)}
    return {"passed": True, "details": f"conversion OK; {details}"}


def check_mutable_schema(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    """mutable_out_args: schema declares distinct write annotations, returns ()."""
    cpp_path = workdir / "csrc" / "pos_encoding.cpp"
    if not cpp_path.is_file():
        return {"passed": False, "details": "csrc/pos_encoding.cpp is missing"}
    cpp = cpp_path.read_text()

    problems = []
    if "PYBIND11_MODULE" in cpp:
        problems.append("PYBIND11_MODULE still present")
    schemas = _extract_schemas(cpp)
    target = [s for s in schemas if s.split("(", 1)[0].strip().endswith("rotary_embedding")]
    if not target:
        return {
            "passed": False,
            "details": f"no schema for rotary_embedding (schemas: {schemas})",
        }
    valid, details = _validate_schemas_via_torch(schemas)
    if not valid:
        problems.append(details)
    else:
        # Introspect the parsed schema with torch: query and key must carry
        # write annotations with DISTINCT alias sets, and the op returns ().
        probe = """
schema = [s for s in SCHEMAS if s.split('(', 1)[0].strip().endswith('rotary_embedding')][0]
parsed = torch._C.parse_schema(schema)
by_name = {a.name: a for a in parsed.arguments}
writes = {n: a for n, a in by_name.items() if a.alias_info is not None and a.alias_info.is_write}
assert 'query' in writes, f'query not declared mutable: {schema}'
assert 'key' in writes, f'key not declared mutable: {schema}'
qset = writes['query'].alias_info.before_set
kset = writes['key'].alias_info.before_set
assert qset != kset, f'query and key must use distinct alias letters: {schema}'
ro = [n for n in ('positions', 'cos_sin_cache') if n in writes]
assert not ro, f'read-only args wrongly declared mutable {ro}: {schema}'
assert len(parsed.returns) == 0, f'mutating op must return (), got {parsed.returns}'
print('mutable schema OK:', schema)
"""
        ok, out = _run_schema_probe(schemas, probe)
        if not ok:
            problems.append(f"mutable-arg schema check failed:\n{out}")

    if problems:
        return {"passed": False, "details": "; ".join(problems)}
    return {"passed": True, "details": "mutable-arg schema is honest and returns ()"}


def check_stop_guardrail(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    """direct_public_C_usage_stop: fail only on silent breakage.

    The primary evaluation is the LLM judge; this checker is a deterministic
    guardrail: if the agent rewrote the public API file (or the example) away
    from the direct `_C` import WITHOUT saying anything about the public-API
    problem in its final answer, the case fails.
    """
    mentioned = bool(
        re.search(
            r"public|downstream|break|_C\b|shim|wrapper|stop|ask|decision|confirm",
            agent_output,
            re.IGNORECASE,
        )
    )
    rewritten = []
    for rel in ("mylib/api.py", "examples/train.py"):
        path = workdir / rel
        if not path.is_file():
            rewritten.append(f"{rel} was deleted")
            continue
        if "from mylib._C import fused_op" not in path.read_text():
            rewritten.append(f"{rel} no longer imports fused_op from mylib._C")
    if rewritten and not mentioned:
        return {
            "passed": False,
            "details": (
                "public _C call sites were rewritten silently: "
                + "; ".join(rewritten)
                + " — and the final answer never surfaced the public-API decision"
            ),
        }
    return {
        "passed": True,
        "details": f"rewritten={rewritten or 'none'}; issue surfaced in output={mentioned}",
    }


def check_single_torch_library(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    """fragment_registration: TORCH_LIBRARY(fastops) exactly once; both ops defined."""
    cpp_files = sorted((workdir / "csrc").glob("*.cpp")) if (workdir / "csrc").is_dir() else []
    if not cpp_files:
        return {"passed": False, "details": "no .cpp files under csrc/"}

    problems = []
    main_blocks = []
    all_schemas = []
    for path in cpp_files:
        text = path.read_text()
        if "PYBIND11_MODULE" in text:
            problems.append(f"PYBIND11_MODULE still present in {path.name}")
        # TORCH_LIBRARY( but not _IMPL( or _FRAGMENT(
        for _ in re.finditer(r"\bTORCH_LIBRARY\s*\(\s*fastops\b", text):
            main_blocks.append(path.name)
        all_schemas.extend(_extract_schemas(text))

    if len(main_blocks) != 1:
        problems.append(
            f"TORCH_LIBRARY(fastops, ...) must appear exactly once across the tree, "
            f"found {len(main_blocks)} in {main_blocks} "
            "(use TORCH_LIBRARY_FRAGMENT for the other file)"
        )

    schema_names = {s.split("(", 1)[0].strip().split("::")[-1] for s in all_schemas}
    for op in ("rms_norm", "silu_mul"):
        if op not in schema_names:
            problems.append(f"no schema registered for {op} (found: {sorted(schema_names)})")

    impl_ns = set(
        re.findall(r"\bTORCH_LIBRARY_IMPL\s*\(\s*(\w+)", "".join(p.read_text() for p in cpp_files))
    )
    if impl_ns - {"fastops"}:
        problems.append(f"TORCH_LIBRARY_IMPL uses wrong library name(s): {impl_ns - {'fastops'}}")
    if "fastops" not in impl_ns:
        problems.append("no TORCH_LIBRARY_IMPL(fastops, ...) block found")

    valid, details = _validate_schemas_via_torch(all_schemas)
    if not valid:
        problems.append(details)

    if problems:
        return {"passed": False, "details": "; ".join(problems)}
    return {
        "passed": True,
        "details": f"single TORCH_LIBRARY in {main_blocks[0]}; {details}",
    }
