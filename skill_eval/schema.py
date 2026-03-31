# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Schema parsing and validation for skill eval contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class SkillEvalValidationError(ValueError):
    """Raised when an eval contract is invalid."""


@dataclass(slots=True)
class DeterministicCheckSpec:
    type: str
    params: dict[str, Any]


@dataclass(slots=True)
class QualityFactorSpec:
    factor: str
    weight: float


@dataclass(slots=True)
class LLMJudgeSpec:
    critical_requirements: list[str]
    quality_factors: list[QualityFactorSpec]


@dataclass(slots=True)
class TestCaseSpec:
    name: str
    user_message: str
    description: str | None
    fixture: str | None
    setup: list[str]
    deterministic_checks: list[DeterministicCheckSpec]
    checker: str | None
    llm_judge: LLMJudgeSpec | None
    tags: list[str]


@dataclass(slots=True)
class EvalSpec:
    path: Path
    eval_version: str
    skill_name: str
    skill_path: str
    models: list[str]
    test_cases: list[TestCaseSpec]


def load_eval_spec(path: Path) -> EvalSpec:
    """Load and validate an eval.yaml file."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise SkillEvalValidationError(f"{path} must contain a top-level mapping")

    eval_version = str(_require_str(raw, "eval_version", path))
    skill_name = _require_str(raw, "skill_name", path)
    skill_path = _require_str(raw, "skill_path", path)
    models = _require_string_list(raw, "models", path)
    test_cases_raw = raw.get("test_cases")
    if not isinstance(test_cases_raw, list) or not test_cases_raw:
        raise SkillEvalValidationError(f"{path} must define a non-empty test_cases list")

    test_cases = [_parse_test_case(item, path) for item in test_cases_raw]
    _validate_unique_test_case_names(test_cases, path)
    return EvalSpec(
        path=path,
        eval_version=eval_version,
        skill_name=skill_name,
        skill_path=skill_path,
        models=models,
        test_cases=test_cases,
    )


def _parse_test_case(raw: Any, path: Path) -> TestCaseSpec:
    if not isinstance(raw, dict):
        raise SkillEvalValidationError(f"{path} test cases must be mappings")

    name = _require_str(raw, "name", path)
    user_message = _require_str(raw, "user_message", path)
    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        raise SkillEvalValidationError(f"{path} test case {name!r} has a non-string description")
    fixture = raw.get("fixture")
    if fixture is not None and not isinstance(fixture, str):
        raise SkillEvalValidationError(f"{path} test case {name!r} has a non-string fixture")
    setup = _optional_string_list(raw, "setup", path)
    tags = _optional_string_list(raw, "tags", path)
    deterministic_checks = _parse_deterministic_checks(raw.get("deterministic_checks"), path, name)
    checker = raw.get("checker")
    if checker is not None and not isinstance(checker, str):
        raise SkillEvalValidationError(f"{path} test case {name!r} has a non-string checker ref")
    llm_judge = _parse_llm_judge(raw.get("llm_judge"), path, name)

    if not deterministic_checks and checker is None and llm_judge is None:
        raise SkillEvalValidationError(
            f"{path} test case {name!r} must define deterministic_checks, checker, or llm_judge"
        )

    return TestCaseSpec(
        name=name,
        user_message=user_message,
        description=description,
        fixture=fixture,
        setup=setup,
        deterministic_checks=deterministic_checks,
        checker=checker,
        llm_judge=llm_judge,
        tags=tags,
    )


def _parse_deterministic_checks(
    raw: Any,
    path: Path,
    test_case_name: str,
) -> list[DeterministicCheckSpec]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise SkillEvalValidationError(
            f"{path} test case {test_case_name!r} deterministic_checks must be a list"
        )
    checks = []
    for item in raw:
        if not isinstance(item, dict):
            raise SkillEvalValidationError(
                f"{path} test case {test_case_name!r} deterministic checks must be mappings"
            )
        checks.append(
            DeterministicCheckSpec(
                type=_require_str(item, "type", path),
                params={key: value for key, value in item.items() if key != "type"},
            )
        )
    return checks


def _parse_llm_judge(raw: Any, path: Path, test_case_name: str) -> LLMJudgeSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SkillEvalValidationError(
            f"{path} test case {test_case_name!r} llm_judge must be a mapping"
        )
    rubric = raw.get("rubric")
    if not isinstance(rubric, dict):
        raise SkillEvalValidationError(
            f"{path} test case {test_case_name!r} llm_judge.rubric must be a mapping"
        )

    critical_requirements = _require_string_list(rubric, "critical_requirements", path)
    quality_factors_raw = rubric.get("quality_factors")
    if not isinstance(quality_factors_raw, list) or not quality_factors_raw:
        raise SkillEvalValidationError(
            f"{path} test case {test_case_name!r} "
            "llm_judge.rubric.quality_factors must be a non-empty list"
        )

    quality_factors = []
    for item in quality_factors_raw:
        if not isinstance(item, dict):
            raise SkillEvalValidationError(
                f"{path} test case {test_case_name!r} quality factors must be mappings"
            )
        factor = _require_str(item, "factor", path)
        weight = item.get("weight")
        if not isinstance(weight, int | float):
            raise SkillEvalValidationError(
                f"{path} test case {test_case_name!r} quality factor {factor!r} "
                "must include numeric weight"
            )
        quality_factors.append(QualityFactorSpec(factor=factor, weight=float(weight)))

    return LLMJudgeSpec(
        critical_requirements=critical_requirements,
        quality_factors=quality_factors,
    )


def _require_str(raw: dict[str, Any], key: str, path: Path) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise SkillEvalValidationError(f"{path} must define non-empty string field {key!r}")
    return value


def _require_string_list(raw: dict[str, Any], key: str, path: Path) -> list[str]:
    value = raw.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise SkillEvalValidationError(f"{path} must define non-empty string list field {key!r}")
    return list(value)


def _optional_string_list(raw: dict[str, Any], key: str, path: Path) -> list[str]:
    value = raw.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SkillEvalValidationError(f"{path} field {key!r} must be a string list when present")
    return list(value)


def _validate_unique_test_case_names(test_cases: list[TestCaseSpec], path: Path) -> None:
    seen_names: set[str] = set()
    duplicate_names: set[str] = set()
    for test_case in test_cases:
        if test_case.name in seen_names:
            duplicate_names.add(test_case.name)
            continue
        seen_names.add(test_case.name)
    if duplicate_names:
        dupes = ", ".join(repr(name) for name in sorted(duplicate_names))
        raise SkillEvalValidationError(f"{path} defines duplicate test case names: {dupes}")
