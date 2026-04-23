"""Run the OSS general eval set against a skill using the existing eval harness.

Assembles a temporary eval.yaml from individual test case files in
oss_general_eval_set/, then delegates to skill_eval's run_ab_eval.

Usage:
    uv run python scripts/run_oss_general_eval.py skills/my-skill
    uv run python scripts/run_oss_general_eval.py skills/my-skill --verbose
    uv run python scripts/run_oss_general_eval.py skills/my-skill --model claude-sonnet-4-6
    uv run python scripts/run_oss_general_eval.py skills/my-skill --output-json results.json
"""

from __future__ import annotations

import argparse
import shlex
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SET_DIR = REPO_ROOT / "oss_general_eval_set"
DEFAULT_MODEL = "claude-opus-4-6"


def find_skill_path(skill_dir: Path) -> str:
    """Return the repo-relative path to SKILL.md for the given skill directory."""
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        raise FileNotFoundError(f"No SKILL.md found in {skill_dir}")
    return str(skill_md.relative_to(REPO_ROOT))


def load_test_cases(eval_set_dir: Path) -> list[dict]:
    """Load all YAML files from the eval set directory as test case dicts."""
    test_cases = []
    for yaml_file in sorted(eval_set_dir.glob("*.yaml")):
        test_cases.append(yaml.safe_load(yaml_file.read_text()))
    return test_cases


def build_eval_spec(skill_dir: Path, models: list[str]) -> dict:
    """Assemble a complete eval.yaml dict from the skill and test case files."""
    skill_name = skill_dir.name
    skill_path = find_skill_path(skill_dir)
    test_cases = load_test_cases(EVAL_SET_DIR)
    if not test_cases:
        raise FileNotFoundError(f"No YAML files found in {EVAL_SET_DIR}")
    return {
        "eval_version": "1",
        "skill_name": skill_name,
        "skill_path": skill_path,
        "models": models,
        "test_cases": test_cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OSS general eval set against a skill.")
    parser.add_argument("skill_dir", type=Path, help="Path to the skill directory")
    parser.add_argument("--model", action="append", help="Model(s) to use (default: claude-opus-4-6)")
    parser.add_argument("--verbose", action="store_true", help="Show per-test details")
    parser.add_argument("--output-json", type=Path, help="Write results to JSON file")
    parser.add_argument("--debug-dir", type=Path, help="Save debug artifacts to this directory")
    args = parser.parse_args()

    skill_dir = args.skill_dir.resolve()
    models = args.model or [DEFAULT_MODEL]
    eval_spec = build_eval_spec(skill_dir, models)

    # Write assembled eval.yaml to a temp file, then run through the harness
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="oss-general-eval-", delete=False
    ) as f:
        yaml.dump(eval_spec, f, default_flow_style=False, sort_keys=False)
        eval_yaml_path = Path(f.name)

    try:
        from skill_eval.runner import run_ab_eval

        runtime_command = f"{shlex.quote(sys.executable)} -m skill_eval.runtime.claude"
        judge_command = f"{shlex.quote(sys.executable)} -m skill_eval.judges.claude"

        result = run_ab_eval(
            eval_yaml_path,
            runtime_command=runtime_command,
            judge_command=judge_command,
            models=models,
            status_stream=sys.stderr,
            verbose=args.verbose,
            debug_dir=args.debug_dir,
        )

        if args.output_json:
            result.write_json(args.output_json)
        print(result.format_summary())
        return 0
    finally:
        eval_yaml_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
