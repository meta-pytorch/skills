# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom checkers for the orchestrate-stable-abi-migration eval."""

import hashlib
import json
import re
from pathlib import Path

_LIBRARY_NAMESPACE_RE = re.compile(
    r"(?:STABLE_)?TORCH_LIBRARY(?:_IMPL|_FRAGMENT)?\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)"
)

MANIFEST_RELPATH = "checks/immutable_manifest.sha256"


def check_sources_unmodified(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    """Verify no protected fixture file was modified or deleted.

    Used by the stop-and-ask cases (non-empty blockers, red baseline): the
    orchestrator must not touch any source/build/test file. The fixture ships
    a sha256 manifest under checks/ listing each protected file.
    """
    manifest_path = workdir / MANIFEST_RELPATH
    if not manifest_path.is_file():
        return {
            "passed": False,
            "details": f"manifest {MANIFEST_RELPATH} missing from workdir (was it deleted?)",
        }

    problems = []
    checked = 0
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected_hash, _, relpath = line.partition("  ")
        relpath = relpath.strip()
        target = workdir / relpath
        if not target.is_file():
            problems.append(f"{relpath}: deleted or missing")
            continue
        actual_hash = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            problems.append(f"{relpath}: modified (sha256 mismatch)")
        checked += 1

    if checked == 0 and not problems:
        return {"passed": False, "details": "manifest contained no entries"}
    if problems:
        return {
            "passed": False,
            "details": "protected files changed although the orchestrator should have "
            "stopped without migrating: " + "; ".join(problems),
        }
    return {"passed": True, "details": f"all {checked} protected files unmodified"}


def _entry_completed(entry, plan: dict) -> bool:
    """Leniently decide whether one dispatch_order entry is marked completed."""
    completed_lists = []
    for key in ("completed_steps", "completed", "steps_completed"):
        value = plan.get(key)
        if isinstance(value, list):
            completed_lists.append({str(v) for v in value})

    if isinstance(entry, dict):
        name = str(entry.get("skill") or entry.get("name") or entry.get("sub_skill") or "")
        if entry.get("completed") is True:
            return True
        status = str(entry.get("status", "")).lower()
        if status in {"completed", "complete", "done"}:
            return True
    else:
        name = str(entry)

    return any(name in names for names in completed_lists)


def check_plan_completed_and_namespace(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    """Verify the plan marks every dispatch_order step completed (per-step
    protocol step 6 / resume bookkeeping), and that no csrc file renamed the
    torch.ops library namespace with a `_stable` suffix (a trap the sub-skill
    explicitly forbids)."""
    plan_path = workdir / ".abi-migration-plan.json"
    if not plan_path.is_file():
        return {"passed": False, "details": ".abi-migration-plan.json missing from workdir"}
    try:
        plan = json.loads(plan_path.read_text())
    except json.JSONDecodeError as exc:
        return {"passed": False, "details": f"plan JSON no longer parses: {exc}"}

    dispatch_order = plan.get("dispatch_order", [])
    if not isinstance(dispatch_order, list) or not dispatch_order:
        return {
            "passed": False,
            "details": f"plan dispatch_order is empty/invalid: {dispatch_order!r}",
        }

    incomplete = [entry for entry in dispatch_order if not _entry_completed(entry, plan)]
    if incomplete:
        return {
            "passed": False,
            "details": "dispatch_order entries not marked completed in the plan: "
            f"{incomplete!r} (orchestrator per-step protocol step 6 requires marking "
            "each finished sub-skill completed)",
        }

    for cpp in sorted((workdir / "csrc").glob("*.cpp")):
        for namespace in _LIBRARY_NAMESPACE_RE.findall(cpp.read_text()):
            if namespace.endswith("_stable"):
                return {
                    "passed": False,
                    "details": f"{cpp.name} registers into namespace {namespace!r} — the "
                    "torch.ops library namespace must not be renamed with a _stable suffix",
                }

    return {
        "passed": True,
        "details": f"all {len(dispatch_order)} dispatch_order steps marked completed; "
        "no _stable namespace suffix introduced",
    }
