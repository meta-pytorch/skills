# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Built-in deterministic checks for skill evals."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from skill_eval.results import DeterministicCheckResult
from skill_eval.schema import DeterministicCheckSpec


def run_deterministic_checks(
    checks: list[DeterministicCheckSpec],
    *,
    workdir: Path,
    agent_output: str,
    tools_invoked: list[dict[str, Any]] | None = None,
) -> list[DeterministicCheckResult]:
    """Run all built-in deterministic checks for a test case."""
    return [
        run_deterministic_check(
            check,
            workdir=workdir,
            agent_output=agent_output,
            tools_invoked=tools_invoked or [],
        )
        for check in checks
    ]


def run_deterministic_check(
    check: DeterministicCheckSpec,
    *,
    workdir: Path,
    agent_output: str,
    tools_invoked: list[dict[str, Any]],
) -> DeterministicCheckResult:
    """Run one deterministic check and capture evidence."""
    match check.type:
        case "output_equals":
            expected = _require_param(check, "value")
            passed = agent_output.strip() == str(expected).strip()
            details = f"expected exact output {expected!r}"
        case "output_contains":
            expected = _require_param(check, "value")
            passed = str(expected) in agent_output
            details = f"expected output to contain {expected!r}"
        case "output_contains_ci":
            expected = str(_require_param(check, "value"))
            passed = expected.casefold() in agent_output.casefold()
            details = f"expected output to contain {expected!r} case-insensitively"
        case "file_exists":
            target = workdir / str(_require_param(check, "path"))
            passed = target.exists()
            details = f"expected file {target.relative_to(workdir)} to exist"
        case "file_contains":
            target = workdir / str(_require_param(check, "path"))
            needle = str(_require_param(check, "value"))
            passed = needle in target.read_text() if target.exists() else False
            details = f"expected file {target.relative_to(workdir)} to contain {needle!r}"
        case "file_json_equals":
            target = workdir / str(_require_param(check, "path"))
            key = str(_require_param(check, "key"))
            expected = check.params["value"]
            actual = None
            error = None
            if target.exists():
                try:
                    actual = _lookup_json_value(json.loads(target.read_text()), key)
                except (json.JSONDecodeError, KeyError) as exc:
                    error = str(exc)
            passed = error is None and actual == expected
            details = _file_json_equals_details(
                target=target,
                workdir=workdir,
                key=key,
                expected=expected,
                actual=actual,
                error=error,
            )
        case "command_exit_code":
            command = str(_require_param(check, "command"))
            expected_exit_code = int(check.params.get("exit_code", 0))
            completed = subprocess.run(
                command,
                shell=True,
                cwd=workdir,
                capture_output=True,
                text=True,
                check=False,
            )
            passed = completed.returncode == expected_exit_code
            details = (
                f"expected command {command!r} to exit {expected_exit_code}; "
                f"actual={completed.returncode}; "
                f"stdout={completed.stdout!r}; stderr={completed.stderr!r}"
            )
        case "skill_invoked":
            expected_skill = str(_require_param(check, "skill"))
            invoked_skills = [t.get("skill", "") for t in tools_invoked if t.get("tool") == "Skill"]
            passed = any(
                s == expected_skill or s.endswith(f":{expected_skill}") for s in invoked_skills
            )
            details = (
                f"expected skill {expected_skill!r} to be invoked; "
                f"skills invoked: {invoked_skills or '(none)'}"
            )
        case _:
            raise ValueError(f"Unsupported deterministic check type: {check.type}")

    return DeterministicCheckResult(
        check_type=check.type,
        passed=passed,
        details=details,
    )


def _require_param(check: DeterministicCheckSpec, key: str) -> Any:
    if key not in check.params:
        raise ValueError(f"Deterministic check {check.type!r} requires parameter {key!r}")
    return check.params[key]


def _lookup_json_value(payload: Any, key: str) -> Any:
    current = payload
    for part in key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        raise KeyError(f"JSON key {key!r} not found")
    return current


def _file_json_equals_details(
    *,
    target: Path,
    workdir: Path,
    key: str,
    expected: Any,
    actual: Any,
    error: str | None,
) -> str:
    details = (
        f"expected JSON key {key!r} in {target.relative_to(workdir)} "
        f"to equal {expected!r}; actual={actual!r}"
    )
    if error is None:
        return details
    return f"{details}; error={error}"
