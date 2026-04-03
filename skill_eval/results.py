"""Result objects and reporting for skill evals."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class DeterministicCheckResult:
    check_type: str
    passed: bool
    details: str


@dataclass(slots=True)
class CheckerResult:
    name: str
    passed: bool
    details: str | None


@dataclass(slots=True)
class JudgeRequirementResult:
    requirement: str
    passed: bool
    evidence: str | None


@dataclass(slots=True)
class JudgeFactorResult:
    factor: str
    weight: float
    score: float
    reasoning: str | None


@dataclass(slots=True)
class JudgeResult:
    critical_requirements: list[JudgeRequirementResult]
    quality_factors: list[JudgeFactorResult]
    overall_passed: bool
    overall_score: float

    @classmethod
    def from_payload(cls, payload: dict, spec) -> JudgeResult:
        requirements = _parse_requirements(
            payload.get("critical_requirements"),
            spec.critical_requirements,
        )
        factors = _parse_quality_factors(
            payload.get("quality_factors"),
            spec.quality_factors,
        )
        score = _weighted_quality_score(factors)
        return cls(
            critical_requirements=requirements,
            quality_factors=factors,
            overall_passed=all(item.passed for item in requirements) and score >= 0.5,
            overall_score=score,
        )


@dataclass(slots=True)
class TestCaseRunResult:
    variant: str
    test_case_name: str
    model: str
    passed: bool
    score: float
    duration_seconds: float
    exit_code: int
    agent_output: str
    agent_error: str
    deterministic_checks: list[DeterministicCheckResult]
    checker_result: CheckerResult | None
    judge_result: JudgeResult | None


@dataclass(slots=True)
class ModelSummary:
    model: str
    total_tests: int
    passed_tests: int
    pass_rate: float
    average_score: float
    average_duration_seconds: float


@dataclass(slots=True)
class EvalRunResult:
    variant: str
    skill_name: str
    eval_version: str
    model_summaries: list[ModelSummary]
    test_results: list[TestCaseRunResult]
    runtime_command: str
    judge_command: str | None

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    def format_summary(self) -> str:
        rows = [
            (
                summary.model,
                f"{summary.passed_tests}/{summary.total_tests} ({summary.pass_rate:.0%})",
                f"{summary.average_score:.2f}",
                f"{summary.average_duration_seconds:.2f}",
            )
            for summary in self.model_summaries
        ]
        return "\n".join(
            [
                f"Eval: {self.skill_name} [{self.variant}] (eval version {self.eval_version})",
                _format_table(
                    ("Model", "Pass Rate", "Avg Score", "Avg Duration (s)"),
                    rows,
                ),
            ]
        )


@dataclass(slots=True)
class ABModelComparison:
    model: str
    baseline: ModelSummary
    treatment: ModelSummary
    delta_pass_rate: float
    delta_average_score: float
    delta_average_duration_seconds: float


@dataclass(slots=True)
class EvalABResult:
    skill_name: str
    eval_version: str
    baseline: EvalRunResult
    treatment: EvalRunResult
    comparisons: list[ABModelComparison]

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    def format_summary(self) -> str:
        rows = []
        for comparison in self.comparisons:
            rows.append(
                (
                    comparison.model,
                    (
                        f"{comparison.baseline.passed_tests}/"
                        f"{comparison.baseline.total_tests} "
                        f"({comparison.baseline.average_score:.2f})"
                    ),
                    (
                        f"{comparison.treatment.passed_tests}/"
                        f"{comparison.treatment.total_tests} "
                        f"({comparison.treatment.average_score:.2f})"
                    ),
                    f"{comparison.delta_average_score:+.2f}",
                    f"{comparison.delta_pass_rate:+.0%}",
                    f"{comparison.delta_average_duration_seconds:+.2f}",
                )
            )
        return "\n".join(
            [
                f"A/B Eval: {self.skill_name} (eval version {self.eval_version})",
                _format_table(
                    (
                        "Model",
                        "Baseline",
                        "Treatment",
                        "Delta Score",
                        "Delta Pass",
                        "Delta Duration (s)",
                    ),
                    rows,
                ),
            ]
        )


def summarize_results(model: str, results: list[TestCaseRunResult]) -> ModelSummary:
    total_tests = len(results)
    passed_tests = sum(1 for result in results if result.passed)
    return ModelSummary(
        model=model,
        total_tests=total_tests,
        passed_tests=passed_tests,
        pass_rate=passed_tests / total_tests if total_tests else 0.0,
        average_score=sum(result.score for result in results) / total_tests if total_tests else 0.0,
        average_duration_seconds=(
            sum(result.duration_seconds for result in results) / total_tests if total_tests else 0.0
        ),
    )


def compare_model_summaries(
    baseline_results: EvalRunResult,
    treatment_results: EvalRunResult,
) -> list[ABModelComparison]:
    baseline_by_model = {summary.model: summary for summary in baseline_results.model_summaries}
    treatment_by_model = {summary.model: summary for summary in treatment_results.model_summaries}
    all_models = sorted(set(baseline_by_model) | set(treatment_by_model))
    comparisons = []
    for model in all_models:
        baseline = baseline_by_model[model]
        treatment = treatment_by_model[model]
        comparisons.append(
            ABModelComparison(
                model=model,
                baseline=baseline,
                treatment=treatment,
                delta_pass_rate=treatment.pass_rate - baseline.pass_rate,
                delta_average_score=treatment.average_score - baseline.average_score,
                delta_average_duration_seconds=(
                    treatment.average_duration_seconds - baseline.average_duration_seconds
                ),
            )
        )
    return comparisons


def compute_case_score(
    *,
    runtime_exit_code: int,
    deterministic_checks: list[DeterministicCheckResult],
    checker_result: CheckerResult | None,
    judge_result: JudgeResult | None,
) -> tuple[bool, float]:
    hard_fail = runtime_exit_code != 0 or any(not check.passed for check in deterministic_checks)
    hard_fail = hard_fail or (checker_result is not None and not checker_result.passed)
    hard_fail = hard_fail or (judge_result is not None and not judge_result.overall_passed)

    if hard_fail:
        return False, 0.0
    if judge_result is not None:
        return True, judge_result.overall_score
    return True, 1.0


def _weighted_quality_score(factors: list[JudgeFactorResult]) -> float:
    total_weight = sum(factor.weight for factor in factors)
    if total_weight == 0:
        return 1.0
    return sum(factor.score * factor.weight for factor in factors) / total_weight


def _format_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    formatted_rows = [
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        formatted_rows.append(
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        )
    return "\n".join(formatted_rows)


def _parse_requirements(
    raw_requirements: object,
    expected_requirements: list[str],
) -> list[JudgeRequirementResult]:
    if not isinstance(raw_requirements, list):
        raise ValueError("Judge payload must include a critical_requirements list")
    if len(raw_requirements) != len(expected_requirements):
        raise ValueError("Judge payload critical_requirements did not match the eval rubric")

    requirements: list[JudgeRequirementResult] = []
    for raw_item, expected_requirement in zip(
        raw_requirements,
        expected_requirements,
        strict=True,
    ):
        if not isinstance(raw_item, dict):
            raise ValueError("Judge critical_requirements entries must be objects")

        requirement = raw_item.get("requirement")
        passed = raw_item.get("passed")
        evidence = raw_item.get("evidence")
        if requirement != expected_requirement:
            raise ValueError("Judge critical_requirements entries did not match the eval rubric")
        if not isinstance(passed, bool):
            raise ValueError("Judge critical_requirements passed must be a boolean")
        if evidence is not None and not isinstance(evidence, str):
            raise ValueError("Judge critical_requirements evidence must be a string when present")

        requirements.append(
            JudgeRequirementResult(
                requirement=requirement,
                passed=passed,
                evidence=evidence,
            )
        )
    return requirements


def _parse_quality_factors(
    raw_factors: object,
    expected_factors: list,
) -> list[JudgeFactorResult]:
    if not isinstance(raw_factors, list):
        raise ValueError("Judge payload must include a quality_factors list")
    if len(raw_factors) != len(expected_factors):
        raise ValueError("Judge payload quality_factors did not match the eval rubric")

    factors: list[JudgeFactorResult] = []
    for raw_item, expected_factor in zip(raw_factors, expected_factors, strict=True):
        if not isinstance(raw_item, dict):
            raise ValueError("Judge quality_factors entries must be objects")

        factor = raw_item.get("factor")
        score = raw_item.get("score")
        weight = raw_item.get("weight", expected_factor.weight)
        reasoning = raw_item.get("reasoning")
        if factor != expected_factor.factor:
            raise ValueError("Judge quality_factors entries did not match the eval rubric")
        if not isinstance(score, int | float):
            raise ValueError("Judge quality_factors score must be numeric")
        if not isinstance(weight, int | float):
            raise ValueError("Judge quality_factors weight must be numeric")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("Judge quality_factors reasoning must be a string when present")

        factors.append(
            JudgeFactorResult(
                factor=factor,
                weight=float(weight),
                score=float(score),
                reasoning=reasoning,
            )
        )
    return factors
