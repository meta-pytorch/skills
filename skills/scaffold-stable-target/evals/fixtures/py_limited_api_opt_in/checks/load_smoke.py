# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Success contract for extension_cpp/__init__.py, exercised without compiling
# anything: the package must load the legacy .so when it is the only one
# present (the stable .so does not exist until the first file migrates), and
# must load BOTH .so files once the stable one exists.
#
# Usage: python3 checks/load_smoke.py {legacy_only|both}

import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path


def _reexec_with_torch_python() -> None:
    """If this interpreter lacks torch (e.g. the eval harness venv), re-exec
    with the first python3 on PATH that has it (the system python does)."""
    if importlib.util.find_spec("torch") is not None:
        return
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        cand = Path(entry) / "python3" if entry else None
        if cand is None or not cand.is_file() or cand.resolve() == Path(sys.executable).resolve():
            continue
        probe = subprocess.run(
            [str(cand), "-c", "import torch"], capture_output=True, timeout=120, check=False
        )
        if probe.returncode == 0:
            os.execv(str(cand), [str(cand), *sys.argv])
    raise SystemExit("no python3 with torch found on PATH")


def main() -> None:
    scenario = sys.argv[1]
    assert scenario in ("legacy_only", "both"), f"unknown scenario {scenario!r}"
    workdir = Path(__file__).resolve().parent.parent
    suffix = sysconfig.get_config_var("EXT_SUFFIX")

    with tempfile.TemporaryDirectory() as td:
        site = Path(td)
        pkg = site / "extension_cpp"
        shutil.copytree(workdir / "extension_cpp", pkg)
        for stray in pkg.glob("*.so"):
            stray.unlink()
        (pkg / f"_C{suffix}").touch()
        if scenario == "both":
            (pkg / f"_C_stable{suffix}").touch()

        import torch

        loaded: list[str] = []
        torch.ops.load_library = lambda path, *a, **k: loaded.append(Path(str(path)).name)

        sys.path.insert(0, str(site))
        import extension_cpp  # noqa: F401

        legacy = [n for n in loaded if "_stable" not in n]
        stable = [n for n in loaded if "_stable" in n]
        if scenario == "legacy_only":
            assert len(legacy) == 1, f"expected exactly one legacy load, got {loaded}"
            assert not stable, f"stable .so does not exist yet but was loaded: {loaded}"
        else:
            assert len(legacy) == 1, f"expected exactly one legacy load, got {loaded}"
            assert len(stable) == 1, f"expected exactly one stable load, got {loaded}"
        print(f"OK {scenario}: loaded {loaded}")


if __name__ == "__main__":
    _reexec_with_torch_python()
    main()
