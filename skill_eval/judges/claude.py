"""Claude CLI judge for skill evals."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Any

SYSTEM_PROMPT = """\
You are an expert evaluation judge for AI coding skill benchmarks.
Given a user message, an agent output, and a rubric, return strict JSON with:
{
  "critical_requirements": [
    {"requirement": "...", "passed": true, "evidence": "..."}
  ],
  "quality_factors": [
    {"factor": "...", "weight": 1.0, "score": 0.0, "reasoning": "..."}
  ],
  "overall_passed": true,
  "overall_score": 0.0
}

Rules:
- All critical requirements must pass for overall_passed to be true.
- overall_score should be a weighted average from 0.0 to 1.0.
- If the response is clearly wrong, score low.
- Return JSON only, with no markdown fences or extra prose.
"""


def main() -> int:
    model = os.environ.get("SKILL_EVAL_JUDGE_MODEL", os.environ.get("SKILL_EVAL_MODEL", ""))
    if not model:
        print("Missing SKILL_EVAL_MODEL or SKILL_EVAL_JUDGE_MODEL", file=sys.stderr)
        return 2

    if shutil.which("claude") is None:
        print("Claude judge failed: 'claude' binary not found on PATH", file=sys.stderr)
        return 2

    payload = json.loads(sys.stdin.read())
    result = judge(model=model, payload=payload)
    sys.stdout.write(json.dumps(result))
    return 0


def judge(*, model: str, payload: dict[str, Any]) -> dict[str, Any]:
    prompt = f"{SYSTEM_PROMPT}\n\n{json.dumps(payload, indent=2, sort_keys=True)}"
    completed = subprocess.run(
        [
            "claude",
            "-p",
            prompt,
            "--model",
            model,
            "--output-format",
            "text",
            "--no-session-persistence",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Claude judge failed (exit {completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return json.loads(completed.stdout.strip())


if __name__ == "__main__":
    raise SystemExit(main())
