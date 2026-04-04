"""CLI for the skill eval harness."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Annotated

import typer

from skill_eval.runner import discover_eval_paths, run_ab_eval, run_eval
from skill_eval.schema import load_eval_spec

app = typer.Typer(
    add_completion=False,
    help="Run and compare skill evaluation contracts.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.command("list")
def list_skills() -> None:
    """List available skills with evals."""
    eval_paths = discover_eval_paths([])
    if not eval_paths:
        typer.echo("No skills with evals found.")
        raise typer.Exit(1)
    for eval_path in eval_paths:
        spec = load_eval_spec(eval_path)
        skill_dir = eval_path.parent.parent
        n_tests = len(spec.test_cases)
        has_judge = any(tc.llm_judge is not None for tc in spec.test_cases)
        judge_marker = " [judge]" if has_judge else ""
        typer.echo(f"  {spec.skill_name}  ({n_tests} test(s){judge_marker})  {skill_dir}")


@app.command()
def validate(
    paths: Annotated[
        list[str] | None,
        typer.Argument(help="Skill directories or eval.yaml files to validate."),
    ] = None,
) -> None:
    """Validate one or more eval contracts."""
    try:
        for eval_path in discover_eval_paths(paths or []):
            load_eval_spec(eval_path)
            typer.secho(f"validated {eval_path}", fg=typer.colors.GREEN)
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(f"validation error: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc


@app.command()
def run(
    path: Annotated[
        Path,
        typer.Argument(help="Skill directory or eval.yaml path."),
    ],
    runtime_command: Annotated[
        str | None,
        typer.Option(
            help=(
                "Shell command used to invoke the agent runtime; "
                "defaults to the built-in Claude runtime."
            ),
        ),
    ] = None,
    judge_command: Annotated[
        str | None,
        typer.Option(
            help=(
                "Optional shell command used to invoke an LLM judge; "
                "defaults to the built-in Claude judge."
            ),
        ),
    ] = None,
    models: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            help="Override models from eval.yaml; may be passed multiple times.",
        ),
    ] = None,
    output_json: Annotated[
        Path | None,
        typer.Option(help="Optional path to write the machine-readable result bundle."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(help="Emit per-test runtime details to stderr while the eval runs."),
    ] = False,
    debug_dir: Annotated[
        Path | None,
        typer.Option(help="Optional directory to store per-test debug artifacts."),
    ] = None,
) -> None:
    """Run one skill eval."""
    runtime_command = runtime_command or _default_module_command("skill_eval.runtime.claude")
    judge_command = judge_command or _default_module_command("skill_eval.judges.claude")
    result = run_eval(
        path,
        runtime_command=runtime_command,
        judge_command=judge_command,
        models=models,
        status_stream=sys.stderr,
        verbose=verbose,
        debug_dir=debug_dir,
    )
    if output_json is not None:
        result.write_json(output_json)
    typer.echo(result.format_summary())


@app.command("run-ab")
def run_ab(
    path: Annotated[
        Path,
        typer.Argument(help="Skill directory or eval.yaml path."),
    ],
    runtime_command: Annotated[
        str | None,
        typer.Option(
            help=(
                "Default shell command used to invoke the agent runtime for both "
                "baseline and treatment; defaults to the built-in Claude runtime."
            ),
        ),
    ] = None,
    baseline_runtime_command: Annotated[
        str | None,
        typer.Option(help="Optional baseline-only runtime command; defaults to --runtime-command."),
    ] = None,
    treatment_runtime_command: Annotated[
        str | None,
        typer.Option(
            help="Optional treatment-only runtime command; defaults to --runtime-command."
        ),
    ] = None,
    judge_command: Annotated[
        str | None,
        typer.Option(
            help=(
                "Optional shell command used to invoke an LLM judge; "
                "defaults to the built-in Claude judge."
            ),
        ),
    ] = None,
    models: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            help="Override models from eval.yaml; may be passed multiple times.",
        ),
    ] = None,
    output_json: Annotated[
        Path | None,
        typer.Option(help="Optional path to write the machine-readable A/B result bundle."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(help="Emit per-test runtime details to stderr while the eval runs."),
    ] = False,
    debug_dir: Annotated[
        Path | None,
        typer.Option(help="Optional directory to store per-test debug artifacts."),
    ] = None,
) -> None:
    """Run one skill eval as baseline vs treatment."""
    runtime_command = runtime_command or _default_module_command("skill_eval.runtime.claude")
    judge_command = judge_command or _default_module_command("skill_eval.judges.claude")
    result = run_ab_eval(
        path,
        runtime_command=runtime_command,
        baseline_runtime_command=baseline_runtime_command,
        treatment_runtime_command=treatment_runtime_command,
        judge_command=judge_command,
        models=models,
        status_stream=sys.stderr,
        verbose=verbose,
        debug_dir=debug_dir,
    )
    if output_json is not None:
        result.write_json(output_json)
    typer.echo(result.format_summary())


@app.command("run-all")
def run_all(
    runtime_command: Annotated[
        str | None,
        typer.Option(
            help=(
                "Shell command used to invoke the agent runtime; "
                "defaults to the built-in Claude runtime."
            ),
        ),
    ] = None,
    judge_command: Annotated[
        str | None,
        typer.Option(
            help=(
                "Optional shell command used to invoke an LLM judge; "
                "defaults to the built-in Claude judge."
            ),
        ),
    ] = None,
    models: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            help="Override models from eval.yaml; may be passed multiple times.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Optional directory for JSON result bundles."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(help="Emit per-test runtime details to stderr while the eval runs."),
    ] = False,
    debug_dir: Annotated[
        Path | None,
        typer.Option(help="Optional directory to store per-test debug artifacts."),
    ] = None,
) -> None:
    """Run all skill evals in the repo."""
    runtime_command = runtime_command or _default_module_command("skill_eval.runtime.claude")
    judge_command = judge_command or _default_module_command("skill_eval.judges.claude")
    for eval_path in discover_eval_paths([]):
        result = run_eval(
            eval_path,
            runtime_command=runtime_command,
            judge_command=judge_command,
            models=models,
            status_stream=sys.stderr,
            verbose=verbose,
            debug_dir=debug_dir,
        )
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            result.write_json(output_dir / f"{result.skill_name}.json")
        typer.echo(result.format_summary())


def _default_module_command(module_name: str) -> str:
    return f"{shlex.quote(sys.executable)} -m {module_name}"


def main() -> None:
    app(prog_name="python3 -m skill_eval")
