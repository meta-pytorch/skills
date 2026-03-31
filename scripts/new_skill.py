# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scaffold a new skill with the minimum eval structure."""

from __future__ import annotations

import re
from pathlib import Path
from textwrap import dedent
from typing import Annotated

import typer

ROOT = Path.cwd()
SKILLS_DIR = ROOT / "skills"
SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def main(
    name: Annotated[str, typer.Argument(help="Lowercase hyphenated skill name.")],
) -> None:
    try:
        skill_dir = create_skill_scaffold(name)
    except (FileExistsError, ValueError) as exc:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc

    typer.secho(f"Created skill scaffold at {skill_dir}", fg=typer.colors.GREEN)
    typer.echo("Next steps:")
    typer.echo(f"  1. Edit {skill_dir / 'SKILL.md'}")
    typer.echo(f"  2. Edit {skill_dir / 'evals' / 'eval.yaml'}")
    typer.echo("  3. Run uv run validate-plugin")
    typer.echo(f"  4. Run uv run validate-evals {skill_dir}")


def create_skill_scaffold(skill_name: str) -> Path:
    if not SKILL_NAME_RE.fullmatch(skill_name):
        raise ValueError("skill name must use lowercase letters, numbers, and single hyphens only")

    skill_dir = SKILLS_DIR / skill_name
    if skill_dir.exists():
        raise FileExistsError(f"{skill_dir} already exists")

    eval_dir = skill_dir / "evals"
    fixtures_dir = eval_dir / "fixtures"
    fixtures_dir.mkdir(parents=True)

    skill_dir.joinpath("SKILL.md").write_text(render_skill_markdown(skill_name))
    eval_dir.joinpath("eval.yaml").write_text(render_eval_yaml(skill_name))
    eval_dir.joinpath("checkers.py").write_text(render_checkers_py())
    fixtures_dir.joinpath(".gitkeep").write_text("")
    return skill_dir


def render_skill_markdown(skill_name: str) -> str:
    return dedent(
        f"""\
        ---
        name: {skill_name}
        description: >-
          Describe what this skill does and when to use it.
          Include trigger words that should make an agent reach for it.
        license: BSD-3-Clause
        metadata:
          author: your-name
          version: "1.0"
        ---

        # {skill_name.replace("-", " ").title()}

        ## When to use this skill

        - Replace with the concrete situations where this helps.

        ## Instructions

        Replace this section with the actual workflow the agent should follow.
        """
    )


def render_eval_yaml(skill_name: str) -> str:
    return dedent(
        f"""\
        eval_version: "1"
        skill_name: "{skill_name}"
        skill_path: "skills/{skill_name}/SKILL.md"

        models:
          - "claude-opus-4-6"

        test_cases:
          - name: "happy_path"
            description: "Replace this with a real task outcome the skill should improve."
            user_message: "Replace this with the prompt or slash command to test."
            deterministic_checks:
              - type: "output_contains_ci"
                value: "replace me"
        """
    )


def render_checkers_py() -> str:
    return dedent(
        """\
        \"\"\"Optional custom outcome checks for this skill.\"\"\"

        from __future__ import annotations


        def check_example(*, workdir, agent_output: str, model: str, test_case_name: str) -> dict:
            del workdir, model, test_case_name
            return {
                "passed": bool(agent_output.strip()),
                "details": "Replace this example checker or delete the checker reference.",
            }
        """
    )


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    cli()
