"""Custom checker loading for skill evals."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from skill_eval.results import CheckerResult


def run_checker(
    checker_ref: str,
    *,
    eval_dir: Path,
    workdir: Path,
    agent_output: str,
    model: str,
    test_case_name: str,
) -> CheckerResult:
    """Load and run a checker from the eval directory."""
    module_name, _, function_name = checker_ref.partition(":")
    if not module_name or not function_name:
        raise ValueError(f"Checker reference must look like 'module:function', got {checker_ref!r}")

    module_path = eval_dir / f"{module_name}.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Checker module {module_path} does not exist")

    spec = importlib.util.spec_from_file_location(
        f"skill_eval.custom_checker.{module_name}",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load checker module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    checker = getattr(module, function_name, None)
    if checker is None:
        raise AttributeError(f"Checker function {function_name!r} not found in {module_path}")

    result = checker(
        workdir=workdir,
        agent_output=agent_output,
        model=model,
        test_case_name=test_case_name,
    )
    return _normalize_checker_result(checker_ref, result)


def _normalize_checker_result(name: str, result: Any) -> CheckerResult:
    if isinstance(result, CheckerResult):
        return result
    if isinstance(result, bool):
        return CheckerResult(name=name, passed=result, details=None)
    if isinstance(result, dict):
        if "passed" not in result:
            raise ValueError("Checker dict result must include 'passed'")
        return CheckerResult(
            name=name,
            passed=bool(result["passed"]),
            details=result.get("details"),
        )
    raise TypeError(
        "Checker must return bool, dict(passed=..., details=...), or CheckerResult",
    )
