# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom deterministic checkers for the migrate-to-stable-apis eval."""

from __future__ import annotations

import ast
import re
from pathlib import Path

# Matches TORCH_CHECK( that is NOT part of a longer identifier such as
# STD_TORCH_CHECK( or TORCH_ERROR_CODE_CHECK(.
BARE_TORCH_CHECK = re.compile(r"(?<![A-Za-z0-9_])TORCH_CHECK\s*\(")

# Matches at:: / c10:: usage that is not part of a longer identifier.
LEGACY_NAMESPACE = re.compile(r"(?<![A-Za-z0-9_:])(?:at|c10)::")

LEGACY_INCLUDE = re.compile(
    r"#\s*include\s*[<\"](?:ATen/|c10/|torch/extension\.h|torch/library\.h|"
    r"torch/torch\.h|torch/all\.h)"
)

# A forward declaration has a return type before the function name; a call
# site never does.
STREAM_FWD_DECL = re.compile(
    r'(?:extern\s+"C"\s+)?(?:AOTITorchError|int32_t|int64_t|int|void)\s+'
    r"aoti_torch_get_current_cuda_stream\s*\("
)

SHIM_INCLUDE = re.compile(r"#\s*include\s*[<\"]torch/csrc/(?:inductor/aoti_torch|stable)/c/shim\.h")


def _read(workdir: Path, rel: str) -> str | None:
    path = workdir / rel
    if not path.is_file():
        return None
    return path.read_text()


def _fail(details: str) -> dict:
    return {"passed": False, "details": details}


def _ok(details: str = "all checks passed") -> dict:
    return {"passed": True, "details": details}


def _strip_comments(source: str) -> str:
    """Remove // and /* */ comments so explanatory comments don't trip greps."""
    without_block = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", without_block)


def _common_cpp_problems(source: str, rel: str) -> list[str]:
    """Problems every fully migrated C++/CUDA file must not have."""
    source = _strip_comments(source)
    problems = []
    if LEGACY_INCLUDE.search(source):
        problems.append(f"{rel}: legacy include (<ATen/...>, <c10/...>, etc.) remains")
    if LEGACY_NAMESPACE.search(source):
        problems.append(f"{rel}: legacy at:: / c10:: usage remains")
    if BARE_TORCH_CHECK.search(source):
        problems.append(f"{rel}: bare TORCH_CHECK( remains (should be STD_TORCH_CHECK)")
    if re.search(r"(?<![A-Za-z0-9_])TORCH_LIBRARY(?:_IMPL|_FRAGMENT)?\s*\(", source):
        problems.append(f"{rel}: non-stable TORCH_LIBRARY* macro remains")
    return problems


def check_fa3_attention(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    """csrc/attention.cpp must be fully migrated with no legacy API remnants."""
    rel = "csrc/attention.cpp"
    source = _read(workdir, rel)
    if source is None:
        return _fail(f"{rel} is missing (file must be migrated in place)")

    problems = _common_cpp_problems(source, rel)
    source = _strip_comments(source)
    if ".data_ptr<" in source:
        problems.append(
            f"{rel}: typed .data_ptr<T>() remains; reads must use "
            ".const_data_ptr<T>() and writes .mutable_data_ptr<T>()"
        )
    if "std::optional" not in source:
        problems.append(f"{rel}: std::optional out parameter was dropped from the signature")
    if "TORCH_BOX(" not in source:
        problems.append(f"{rel}: m.impl must wrap the kernel pointer in TORCH_BOX(...)")
    if not re.search(r"STABLE_TORCH_LIBRARY\s*\(\s*fastattn\b", source):
        problems.append(f"{rel}: STABLE_TORCH_LIBRARY(fastattn, ...) not found")
    if not re.search(r"STABLE_TORCH_LIBRARY_IMPL\s*\(\s*fastattn\s*,\s*CPU\b", source):
        problems.append(f"{rel}: STABLE_TORCH_LIBRARY_IMPL(fastattn, CPU, ...) not found")

    if problems:
        return _fail("; ".join(problems))
    return _ok()


def check_cuda_stream_migration(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    """csrc/rope_kernel.cu must use tensor-derived device index and shim.h."""
    rel = "csrc/rope_kernel.cu"
    source = _read(workdir, rel)
    if source is None:
        return _fail(f"{rel} is missing (file must be migrated in place)")

    problems = _common_cpp_problems(source, rel)
    source = _strip_comments(source)

    if "getCurrentDeviceIndex" in source:
        problems.append(
            f"{rel}: uses torch::stable::accelerator::getCurrentDeviceIndex(); "
            "the device index must come from a tensor via .get_device_index() "
            "(wrong stream on multi-GPU otherwise)"
        )
    calls = re.findall(r"aoti_torch_get_current_cuda_stream\s*\(([^;]*?)\)\s*\)?\s*;", source)
    call_args = [args for args in calls if "&" in args]  # actual call sites pass &stream
    if not call_args:
        problems.append(f"{rel}: no call to aoti_torch_get_current_cuda_stream found")
    elif not any(".get_device_index()" in args for args in call_args):
        problems.append(
            f"{rel}: aoti_torch_get_current_cuda_stream is not passed a "
            "tensor-derived .get_device_index()"
        )
    if STREAM_FWD_DECL.search(source):
        problems.append(
            f"{rel}: aoti_torch_get_current_cuda_stream is forward-declared; "
            "it must come from torch/csrc/inductor/aoti_torch/c/shim.h"
        )
    if not SHIM_INCLUDE.search(source):
        problems.append(f"{rel}: missing shim.h include for aoti_torch_get_current_cuda_stream")
    if "C10_CUDA_KERNEL_LAUNCH_CHECK" in source:
        problems.append(f"{rel}: C10_CUDA_KERNEL_LAUNCH_CHECK remains")
    if "STD_CUDA_KERNEL_LAUNCH_CHECK" not in source:
        problems.append(f"{rel}: STD_CUDA_KERNEL_LAUNCH_CHECK() not found")
    if not re.search(r"DeviceGuard\s+\w+\s*\(\s*\w+\.get_device_index\(\)", source):
        problems.append(f"{rel}: DeviceGuard must be constructed from tensor.get_device_index()")

    if problems:
        return _fail("; ".join(problems))
    return _ok()


def check_strided_trap(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    """new_empty_strided does not exist; must use new_empty + transpose."""
    rel = "csrc/colmajor_scale.cpp"
    source = _read(workdir, rel)
    if source is None:
        return _fail(f"{rel} is missing (file must be migrated in place)")

    problems = _common_cpp_problems(source, rel)
    source = _strip_comments(source)
    if "new_empty_strided" in source:
        problems.append(
            f"{rel}: torch::stable::new_empty_strided does not exist; use "
            "new_empty with transposed dims + torch::stable::transpose"
        )
    if "empty_strided" in source:
        problems.append(f"{rel}: empty_strided remains")
    if not re.search(r"torch::stable::transpose\s*\(", source):
        problems.append(
            f"{rel}: torch::stable::transpose(...) not found; the column-major "
            "output must be built as a transposed contiguous tensor"
        )
    if not re.search(r"torch::stable::new_empty\s*\(", source):
        problems.append(f"{rel}: torch::stable::new_empty(...) not found")

    if problems:
        return _fail("; ".join(problems))
    return _ok()


# Expected content of the untouched Python wrapper in the
# namespace_preservation_trap fixture (whitespace-normalized comparison).
EXPECTED_OPS_PY = '''\
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Python wrapper API for the fastkernels extension."""

import torch


@torch.library.register_fake("fastkernels::rmsnorm")
def _rmsnorm_fake(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    torch._check(input.dim() == 2)
    torch._check(weight.dim() == 1)
    torch._check(input.size(1) == weight.size(0))
    return torch.empty_like(input)


def rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMS normalization over the last dimension."""
    return torch.ops.fastkernels.rmsnorm(input, weight, eps)
'''


def _normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def check_namespace_preserved(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    """The registration namespace must stay `fastkernels`; wrapper untouched."""
    rel = "csrc/rmsnorm.cpp"
    source = _read(workdir, rel)
    if source is None:
        return _fail(f"{rel} is missing (file must be migrated in place)")

    problems = _common_cpp_problems(source, rel)
    source = _strip_comments(source)
    if "fastkernels_stable" in source:
        problems.append(
            f"{rel}: namespace was renamed to fastkernels_stable; the torch.ops "
            "namespace must stay exactly `fastkernels`"
        )
    if not re.search(r"STABLE_TORCH_LIBRARY\s*\(\s*fastkernels\s*,", source):
        problems.append(f"{rel}: STABLE_TORCH_LIBRARY(fastkernels, ...) not found")
    if not re.search(r"STABLE_TORCH_LIBRARY_IMPL\s*\(\s*fastkernels\s*,\s*CPU\b", source):
        problems.append(f"{rel}: STABLE_TORCH_LIBRARY_IMPL(fastkernels, CPU, ...) not found")

    ops_py = _read(workdir, "fastkernels/_ops.py")
    if ops_py is None:
        problems.append("fastkernels/_ops.py was deleted; it must remain unchanged")
    elif _normalize(ops_py) != _normalize(EXPECTED_OPS_PY):
        problems.append(
            "fastkernels/_ops.py was modified; the Python wrapper must remain "
            "unchanged (torch.ops.fastkernels.* keeps working through the migration)"
        )

    if problems:
        return _fail("; ".join(problems))
    return _ok()


def check_setup_py_wiring(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    """setup.py must gain TORCH_TARGET_VERSION for 2.12 on the existing entry."""
    setup_src = _read(workdir, "setup.py")
    if setup_src is None:
        return _fail("setup.py is missing")

    try:
        tree = ast.parse(setup_src)
    except SyntaxError as exc:
        return _fail(f"setup.py does not parse: {exc}")

    problems = []

    # Exactly one ext_modules entry (one-shot mode adds flags to the existing
    # entry; it never adds a second, parallel stable target).
    ext_modules_len = None
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.keyword)
            and node.arg == "ext_modules"
            and isinstance(node.value, ast.List | ast.Tuple)
        ):
            ext_modules_len = len(node.value.elts)
    if ext_modules_len is None:
        problems.append("could not find an ext_modules list in setup.py")
    elif ext_modules_len != 1:
        problems.append(
            f"ext_modules has {ext_modules_len} entries; one-shot mode must keep "
            "exactly the single existing entry (no parallel stable target)"
        )

    # A TORCH_TARGET_VERSION flag for PyTorch 2.12 must appear in a string
    # constant (i.e., inside extra_compile_args). Be lenient about the exact
    # zero-padding of the hex, but require major.minor == 2.12.
    ttv_strings = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "TORCH_TARGET_VERSION" in node.value
    ]
    if not ttv_strings:
        problems.append("no -DTORCH_TARGET_VERSION flag found in setup.py")
    else:
        version_ok = False
        for flag in ttv_strings:
            match = re.search(r"TORCH_TARGET_VERSION\s*=\s*(0[xX][0-9a-fA-F]+)", flag)
            if match:
                value = int(match.group(1), 16)
                if (value >> 48) == 0x020C:
                    version_ok = True
        if not version_ok:
            problems.append(
                f"TORCH_TARGET_VERSION flag(s) {ttv_strings!r} do not encode "
                "PyTorch 2.12 (expected hex of the form 0x020c000000000000)"
            )

    # The user declined Py_LIMITED_API; it must not be enabled.
    limited_api = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "Py_LIMITED_API" in node.value
    ]
    if limited_api:
        problems.append(f"Py_LIMITED_API was enabled ({limited_api!r}) but the user declined it")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.keyword)
            and node.arg == "py_limited_api"
            and isinstance(node.value, ast.Constant)
            and node.value.value
        ):
            problems.append("py_limited_api=True was set but the user declined it")

    # Both sources must still be in the (single) entry.
    for src in ("csrc/muladd.cpp", "csrc/relu_op.cpp"):
        if src not in setup_src:
            problems.append(f"{src} is no longer listed in setup.py sources")

    # And both migrated sources must be clean of legacy APIs.
    for rel in ("csrc/muladd.cpp", "csrc/relu_op.cpp"):
        source = _read(workdir, rel)
        if source is None:
            problems.append(f"{rel} is missing")
            continue
        problems.extend(_common_cpp_problems(source, rel))
        if "TORCH_BOX(" not in source:
            problems.append(f"{rel}: m.impl must wrap the kernel pointer in TORCH_BOX(...)")

    if problems:
        return _fail("; ".join(problems))
    return _ok()


# The exact TORCH_CHECK argument bodies from the fixture. Per critical rule 8,
# only the macro name may change during migration; every condition and message
# argument must survive verbatim (modulo whitespace / literal re-wrapping).
EXPECTED_CHECK_BODIES = [
    'input.dim() == expected_rank, "checked_flatten: expected rank ", '
    'expected_rank, ", but got tensor of rank ", input.dim(), '
    '" (shape has ", input.numel(), " total elements)"',
    'input.is_contiguous(), "checked_flatten: input must be contiguous; '
    'call .contiguous() first, then retry"',
    "input.numel() > 0 && input.numel() < (int64_t{1} << 31), "
    '"checked_flatten: numel must be in (0, 2^31), got ", input.numel(), '
    '"; empty and oversized tensors are not supported"',
    "input.size(0) % 2 == 0 || input.size(0) == 1, "
    '"checked_flatten: leading dim must be even or 1, got ", input.size(0)',
]


def _normalize_cpp(text: str) -> str:
    """Strip all whitespace and merge adjacent string literals.

    Both the expected bodies and the migrated source get the same treatment,
    so the comparison is whitespace- and line-wrapping-insensitive while still
    requiring every condition and message argument to survive verbatim.
    """
    stripped = re.sub(r"\s+", "", text)
    return stripped.replace('""', "")


def check_torch_check_bodies(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    """Every TORCH_CHECK body must survive verbatim under STD_TORCH_CHECK."""
    rel = "csrc/validate.cpp"
    source = _read(workdir, rel)
    if source is None:
        return _fail(f"{rel} is missing (file must be migrated in place)")

    problems = _common_cpp_problems(source, rel)

    normalized = _normalize_cpp(source)
    for body in EXPECTED_CHECK_BODIES:
        expected = _normalize_cpp(f"STD_TORCH_CHECK({body})")
        if expected not in normalized:
            problems.append(
                f"{rel}: check body not preserved verbatim under STD_TORCH_CHECK: {body[:70]}..."
            )

    if problems:
        return _fail("; ".join(problems))
    return _ok()
