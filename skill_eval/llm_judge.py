# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optional LLM-as-judge integration for skill evals."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from skill_eval.results import JudgeResult
from skill_eval.schema import LLMJudgeSpec


def run_llm_judge(
    spec: LLMJudgeSpec,
    *,
    judge_command: str,
    workdir: Path,
    model: str,
    agent_output: str,
    user_message: str,
) -> JudgeResult:
    """Invoke an external judge command with a JSON payload."""
    payload = {
        "user_message": user_message,
        "agent_output": agent_output,
        "model": model,
        "rubric": {
            "critical_requirements": spec.critical_requirements,
            "quality_factors": [
                {"factor": factor.factor, "weight": factor.weight}
                for factor in spec.quality_factors
            ],
        },
    }
    env = os.environ.copy()
    env["SKILL_EVAL_MODEL"] = model
    completed = subprocess.run(
        judge_command,
        shell=True,
        cwd=workdir,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Judge command failed with "
            f"exit code {completed.returncode}: {completed.stderr.strip()}"
        )
    return JudgeResult.from_payload(json.loads(completed.stdout), spec)
