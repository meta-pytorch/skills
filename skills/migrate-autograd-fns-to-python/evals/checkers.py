# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom checkers for the migrate-autograd-fns-to-python eval.

Structural checks that complement the runtime grad_smoke.py checks: the
C++ Autograd translation unit must actually be gone (not just shadowed by
the Python registration), and the Python registration must be wired into
the package import.
"""

from pathlib import Path


def check_cpp_autograd_removed(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    """The C++ Autograd impl is fully removed and Python autograd is wired in.

    Verifies:
    1. setup.py no longer lists csrc/autograd.cpp.
    2. No csrc/ file registers the Autograd dispatch key or subclasses
       torch::autograd::Function.
    3. register_autograd appears in the gradext package.
    4. The module calling register_autograd is imported by
       gradext/__init__.py (or the call lives in __init__.py itself).
    """
    problems: list[str] = []

    setup_py = workdir / "setup.py"
    if not setup_py.exists():
        problems.append("setup.py is missing")
    elif "autograd.cpp" in setup_py.read_text():
        problems.append("setup.py still lists csrc/autograd.cpp as a source")

    csrc = workdir / "csrc"
    for cpp in sorted(csrc.glob("**/*.cpp")) if csrc.is_dir() else []:
        text = cpp.read_text()
        if "Autograd" in text and "TORCH_LIBRARY_IMPL" in text:
            for line in text.splitlines():
                if "TORCH_LIBRARY_IMPL" in line and "Autograd" in line:
                    problems.append(
                        f"{cpp.relative_to(workdir)} still has an Autograd "
                        f"TORCH_LIBRARY_IMPL block: {line.strip()}"
                    )
        if "torch::autograd::Function" in text:
            problems.append(
                f"{cpp.relative_to(workdir)} still defines a torch::autograd::Function subclass"
            )
        if "torch/autograd.h" in text:
            problems.append(f"{cpp.relative_to(workdir)} still includes <torch/autograd.h>")

    pkg = workdir / "gradext"
    registering: list[Path] = []
    for py in sorted(pkg.glob("*.py")) if pkg.is_dir() else []:
        if "register_autograd" in py.read_text():
            registering.append(py)
    if not registering:
        problems.append("no module under gradext/ calls torch.library.register_autograd")
    else:
        init = pkg / "__init__.py"
        init_text = init.read_text() if init.exists() else ""
        wired = False
        for py in registering:
            if py.name == "__init__.py" or py.stem in init_text:
                wired = True
                break
        if not wired:
            problems.append(
                f"gradext/__init__.py does not import the module that calls "
                f"register_autograd ({', '.join(p.name for p in registering)})"
            )

    if problems:
        return {"passed": False, "details": "; ".join(problems)}
    return {
        "passed": True,
        "details": (
            "C++ Autograd impl removed from setup.py and csrc/; "
            "register_autograd present and imported at package init"
        ),
    }
