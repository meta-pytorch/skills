# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import unittest

from skill_eval.results import (
    EvalABResult,
    EvalRunResult,
    JudgeResult,
    ModelSummary,
    compare_model_summaries,
)
from skill_eval.schema import LLMJudgeSpec, QualityFactorSpec


def make_judge_spec() -> LLMJudgeSpec:
    return LLMJudgeSpec(
        critical_requirements=["Must find the issue"],
        quality_factors=[QualityFactorSpec(factor="Actionable", weight=1.0)],
    )


class JudgeResultTest(unittest.TestCase):
    def test_missing_requirements_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "critical_requirements"):
            JudgeResult.from_payload(
                {
                    "quality_factors": [
                        {"factor": "Actionable", "score": 1.0, "weight": 1.0},
                    ]
                },
                make_judge_spec(),
            )

    def test_top_level_verdict_is_ignored(self) -> None:
        result = JudgeResult.from_payload(
            {
                "critical_requirements": [
                    {
                        "requirement": "Must find the issue",
                        "passed": False,
                        "evidence": "missed it",
                    }
                ],
                "quality_factors": [
                    {"factor": "Actionable", "score": 1.0, "weight": 1.0},
                ],
                "overall_passed": True,
                "overall_score": 1.0,
            },
            make_judge_spec(),
        )
        self.assertFalse(result.overall_passed)
        self.assertEqual(result.overall_score, 1.0)


class SummaryFormattingTest(unittest.TestCase):
    def test_ab_summary_aligns_to_long_model_names(self) -> None:
        baseline = EvalRunResult(
            variant="baseline",
            skill_name="agents-are-coming",
            eval_version="1",
            model_summaries=[
                ModelSummary(
                    model="claude-opus-4-6",
                    total_tests=1,
                    passed_tests=0,
                    pass_rate=0.0,
                    average_score=0.0,
                    average_duration_seconds=23.36,
                )
            ],
            test_results=[],
            runtime_command="",
            judge_command=None,
        )
        treatment = EvalRunResult(
            variant="treatment",
            skill_name="agents-are-coming",
            eval_version="1",
            model_summaries=[
                ModelSummary(
                    model="claude-opus-4-6",
                    total_tests=1,
                    passed_tests=1,
                    pass_rate=1.0,
                    average_score=1.0,
                    average_duration_seconds=33.07,
                )
            ],
            test_results=[],
            runtime_command="",
            judge_command=None,
        )
        summary = EvalABResult(
            skill_name="agents-are-coming",
            eval_version="1",
            baseline=baseline,
            treatment=treatment,
            comparisons=compare_model_summaries(baseline, treatment),
        ).format_summary()
        self.assertIn("Model            Baseline", summary)
        self.assertIn("claude-opus-4-6  0/1 (0.00)", summary)


if __name__ == "__main__":
    unittest.main()
