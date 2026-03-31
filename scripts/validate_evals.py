# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validate skill definitions and eval contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml

from skill_eval.runner import discover_eval_paths
from skill_eval.schema import SkillEvalValidationError, load_eval_spec


def main(
    paths: Annotated[
        list[str] | None,
        typer.Argument(help="Skill directories or eval.yaml files to validate."),
    ] = None,
) -> None:
    try:
        for eval_path in discover_eval_paths(
            paths or [str(path) for path in sorted(Path("skills").glob("*")) if path.is_dir()]
        ):
            validate_skill(eval_path, load_eval_spec=load_eval_spec)
            typer.secho(f"validated {eval_path}", fg=typer.colors.GREEN)
    except (FileNotFoundError, SkillEvalValidationError, ValueError) as exc:
        typer.secho(f"validation error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc


def validate_skill(eval_path: Path, *, load_eval_spec) -> None:
    skill_dir = eval_path.parent.parent
    skill_name = skill_dir.name
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.is_file():
        raise ValueError(f"{skill_file} is missing")

    frontmatter = load_frontmatter(skill_file)
    declared_name = frontmatter.get("name")
    if declared_name != skill_name:
        raise ValueError(
            f"{skill_file} frontmatter name {declared_name!r} "
            f"does not match directory {skill_name!r}"
        )
    if "description" not in frontmatter or not isinstance(frontmatter["description"], str):
        raise ValueError(f"{skill_file} is missing required 'description' frontmatter")

    spec = load_eval_spec(eval_path)
    if spec.skill_name != skill_name:
        raise ValueError(
            f"{eval_path} declares skill_name {spec.skill_name!r}; expected {skill_name!r}"
        )
    expected_skill_path = f"skills/{skill_name}/SKILL.md"
    if spec.skill_path != expected_skill_path:
        raise ValueError(
            f"{eval_path} declares skill_path {spec.skill_path!r}; expected {expected_skill_path!r}"
        )

    eval_dir = eval_path.parent
    for test_case in spec.test_cases:
        if test_case.fixture is not None and not (eval_dir / test_case.fixture).exists():
            raise ValueError(f"{eval_path} fixture path {test_case.fixture!r} does not exist")
        if test_case.checker is not None:
            module_name, _, _function_name = test_case.checker.partition(":")
            checker_path = eval_dir / f"{module_name}.py"
            if not checker_path.is_file():
                raise ValueError(f"{eval_path} checker module {checker_path} does not exist")


def load_frontmatter(path: Path) -> dict:
    from skill_eval.schema import SkillEvalValidationError

    lines = path.read_text().splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillEvalValidationError(f"{path} is missing YAML frontmatter")

    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
    if end_index is None:
        raise SkillEvalValidationError(f"{path} has an unterminated YAML frontmatter block")

    payload = yaml.safe_load("\n".join(lines[1:end_index]))
    if not isinstance(payload, dict):
        raise SkillEvalValidationError(f"{path} frontmatter must be a mapping")
    return payload


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    cli()
