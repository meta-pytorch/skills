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
    command = [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "text",
        "--no-session-persistence",
        "--permission-mode",
        "bypassPermissions",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Claude judge failed (exit {completed.returncode}): "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    raw_output = completed.stdout.strip()
    if not raw_output:
        raise RuntimeError(
            f"Claude judge returned empty output. stderr: {completed.stderr.strip()}"
        )
    output = _extract_json(raw_output)
    if not output:
        raise RuntimeError(
            f"Claude judge returned no extractable JSON.\n"
            f"Raw output ({len(raw_output)} chars): {raw_output[:1000]}"
        )
    try:
        return json.loads(output)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Claude judge returned invalid JSON: {e}\n"
            f"Raw output ({len(raw_output)} chars): {raw_output[:1000]}"
        ) from e


def _extract_json(text: str) -> str:
    """Extract JSON from text that may contain markdown fences or surrounding prose."""
    import re

    # Try to find JSON inside markdown fences anywhere in the text
    fenced = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    # Fall back to finding the first top-level JSON object
    brace = text.find("{")
    if brace != -1:
        return text[brace:]
    return text


if __name__ == "__main__":
    raise SystemExit(main())
