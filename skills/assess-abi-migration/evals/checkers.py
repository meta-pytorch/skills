# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Custom checkers for the assess-abi-migration eval.

All checkers validate the machine-readable plan the skill emits at
``<workdir>/.abi-migration-plan.json``. They are lenient about extra
schema fields but strict about the semantics under test.
"""

import hashlib
import json
from pathlib import Path

PLAN_NAME = ".abi-migration-plan.json"

CANONICAL_DISPATCH = [
    "pybind-to-torch-library",
    "migrate-meta-fns-to-python",
    "migrate-autograd-fns-to-python",
    "scaffold-stable-target",
    "migrate-to-stable-apis",
]


def _load_plan(workdir: Path):
    """Return (plan_dict, error_string). Exactly one is None."""
    plan_path = workdir / PLAN_NAME
    if not plan_path.is_file():
        return None, f"{PLAN_NAME} not found in workdir {workdir}"
    try:
        plan = json.loads(plan_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, f"{PLAN_NAME} is not valid JSON: {exc}"
    if not isinstance(plan, dict):
        return None, f"{PLAN_NAME} top level must be a JSON object, got {type(plan).__name__}"
    return plan, None


def _file_entries(plan):
    files = plan.get("files")
    return files if isinstance(files, list) else []


def _entry_for(plan, path_suffix):
    """Find a files[] entry whose path ends with path_suffix."""
    for entry in _file_entries(plan):
        if isinstance(entry, dict) and str(entry.get("path", "")).endswith(path_suffix):
            return entry
    return None


def _concerns_text(entry):
    concerns = entry.get("concerns", [])
    if not isinstance(concerns, list):
        concerns = [concerns]
    return " ".join(str(c).lower() for c in concerns)


def _cadence(plan):
    raw = str(plan.get("migration_cadence", "")).strip().lower()
    return raw.replace("_", "-").replace(" ", "-")


def check_full_repo_plan(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    plan, err = _load_plan(workdir)
    if err:
        return {"passed": False, "details": err}

    problems = []

    if "2.12" not in str(plan.get("target_torch_version", "")):
        problems.append(
            f"target_torch_version should contain '2.12', got {plan.get('target_torch_version')!r}"
        )

    if _cadence(plan) != "incremental":
        problems.append(
            f"migration_cadence should be 'incremental', got {plan.get('migration_cadence')!r}"
        )

    # Per-file concern tags. Lenient on exact tag names, strict on the key
    # concern being represented for each file.
    expected = {
        "csrc/bindings.cpp": "pybind",
        "csrc/meta.cpp": "meta",
        "csrc/autograd.cpp": "autograd",
        "csrc/kernels.cu": "unstable",
    }
    for suffix, keyword in expected.items():
        entry = _entry_for(plan, suffix)
        if entry is None:
            problems.append(f"files[] missing an entry for {suffix}")
        elif keyword not in _concerns_text(entry):
            problems.append(
                f"{suffix} concerns {entry.get('concerns')!r} do not include a '{keyword}' concern"
            )

    if _entry_for(plan, "experimental/prototype_kernel.cpp") is not None:
        problems.append(
            "experimental/prototype_kernel.cpp must be EXCLUDED from files[] (not shipped)"
        )

    fakes = plan.get("existing_python_fakes", [])
    if not isinstance(fakes, list) or not any("already_faked_op" in str(f) for f in fakes):
        problems.append(
            f"existing_python_fakes should include mixedext::already_faked_op, got {fakes!r}"
        )

    order = plan.get("dispatch_order")
    if not isinstance(order, list):
        problems.append(f"dispatch_order should be a list, got {order!r}")
    else:
        unknown = [s for s in order if s not in CANONICAL_DISPATCH]
        if unknown:
            problems.append(f"dispatch_order contains unknown sub-skills: {unknown}")
        missing = [s for s in CANONICAL_DISPATCH if s not in order]
        if missing:
            problems.append(
                f"dispatch_order missing {missing} "
                "(all four concerns present + incremental cadence)"
            )
        if not unknown and not missing:
            idx = {s: order.index(s) for s in CANONICAL_DISPATCH}
            if idx["pybind-to-torch-library"] != 0:
                problems.append("pybind-to-torch-library must come first in dispatch_order")
            if idx["migrate-to-stable-apis"] != len(order) - 1:
                problems.append("migrate-to-stable-apis must come last in dispatch_order")
            for pre in ("migrate-meta-fns-to-python", "migrate-autograd-fns-to-python"):
                if idx[pre] > idx["scaffold-stable-target"]:
                    problems.append(f"{pre} must come before scaffold-stable-target")

    if problems:
        return {"passed": False, "details": "; ".join(problems)}
    return {
        "passed": True,
        "details": "plan JSON has correct scope, concerns, fakes, cadence, and dispatch order",
    }


def check_coverage_gaps(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    plan, err = _load_plan(workdir)
    if err:
        return {"passed": False, "details": err}

    problems = []
    gaps = plan.get("coverage_gaps")
    if not isinstance(gaps, list):
        return {"passed": False, "details": f"coverage_gaps should be a list, got {gaps!r}"}
    gaps = [g for g in gaps if isinstance(g, dict)]

    def gap_matches(gap, needle):
        return needle in str(gap.get("old_api", "")).lower()

    # (a) at::empty_like exists in the installed stable headers (min 2.9) and
    # must NOT be reported as a gap — the skill has to grep, not assume.
    false_positives = [g for g in gaps if gap_matches(g, "empty_like")]
    if false_positives:
        problems.append(
            "empty_like is present in torch/csrc/stable/ops.h and must not be a coverage gap; "
            f"got {false_positives!r}"
        )

    # (b) at::TensorIterator is structurally absent from the stable surface.
    ti = [g for g in gaps if "tensoriterator" in json.dumps(g).lower()]
    if not ti:
        problems.append(
            "coverage_gaps missing an entry for at::TensorIterator (absent from stable headers)"
        )
    else:
        gap = ti[0]
        min_target = gap.get("min_target_to_close", "MISSING")
        ok_na = isinstance(min_target, str) and min_target.strip().lower() in ("n/a", "na", "none")
        ok_null = min_target is None and str(gap.get("reason") or "").strip() != ""
        if not (ok_na or ok_null):
            problems.append(
                "TensorIterator gap should have min_target_to_close 'n/a' (structural) "
                f"or null with a reason; got min_target_to_close={min_target!r}, "
                f"reason={gap.get('reason')!r}"
            )

    # (c) at::empty_strided has a documented workaround (new_empty + transpose).
    es = [g for g in gaps if gap_matches(g, "empty_strided")]
    if not es:
        problems.append("coverage_gaps missing an entry for at::empty_strided")
    else:
        gap = es[0]
        workaround_ok = str(gap.get("workaround") or "").strip() != ""
        not_blocked = gap.get("blocked") is False
        if not (workaround_ok or not_blocked):
            problems.append(
                "empty_strided gap should carry a non-empty workaround "
                f"(new_empty + transpose) or blocked=false; got "
                f"workaround={gap.get('workaround')!r}, blocked={gap.get('blocked')!r}"
            )

    if problems:
        return {"passed": False, "details": "; ".join(problems)}
    return {
        "passed": True,
        "details": "coverage gaps correctly reflect the installed torch's stable headers",
    }


def check_no_tests_blocker(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    plan, err = _load_plan(workdir)
    if err:
        return {"passed": False, "details": err}

    blockers = plan.get("blockers")
    if not isinstance(blockers, list) or not blockers:
        return {
            "passed": False,
            "details": f"blockers[] should be non-empty (repo has no tests), got {blockers!r}",
        }
    test_related = [b for b in blockers if "test" in json.dumps(b).lower()]
    if not test_related:
        return {
            "passed": False,
            "details": f"no blocker mentions the missing test setup; blockers={blockers!r}",
        }
    return {"passed": True, "details": f"found test-related blocker: {test_related[0]!r}"}


def check_one_shot_readonly(
    *, workdir: Path, agent_output: str, model: str, test_case_name: str
) -> dict:
    plan, err = _load_plan(workdir)
    if err:
        return {"passed": False, "details": err}

    problems = []

    if _cadence(plan) != "one-shot":
        problems.append(
            f"migration_cadence should be 'one-shot', got {plan.get('migration_cadence')!r}"
        )

    order = plan.get("dispatch_order")
    if not isinstance(order, list):
        problems.append(f"dispatch_order should be a list, got {order!r}")
    elif "scaffold-stable-target" in order:
        problems.append(
            "dispatch_order must EXCLUDE scaffold-stable-target for a one-shot migration"
        )

    # Read-only discipline: every fixture source file must be byte-identical.
    manifest_path = workdir / "checks" / "hashes.json"
    if not manifest_path.is_file():
        problems.append("checks/hashes.json manifest missing from workdir")
    else:
        manifest = json.loads(manifest_path.read_text())
        for rel, expected_sha in sorted(manifest.items()):
            p = workdir / rel
            if not p.is_file():
                problems.append(f"fixture file deleted or moved: {rel}")
                continue
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            if actual != expected_sha:
                problems.append(f"fixture file MODIFIED (skill must not modify code): {rel}")

    if problems:
        return {"passed": False, "details": "; ".join(problems)}
    return {
        "passed": True,
        "details": "one-shot cadence recorded, scaffold omitted, and no fixture file modified",
    }
