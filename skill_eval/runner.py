# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eval discovery and execution."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TextIO

from skill_eval.checker_loader import run_checker
from skill_eval.checks import run_deterministic_checks
from skill_eval.llm_judge import run_llm_judge
from skill_eval.results import (
    CheckerResult,
    DeterministicCheckResult,
    EvalABResult,
    EvalRunResult,
    JudgeResult,
    TestCaseRunResult,
    compare_model_summaries,
    compute_case_score,
    summarize_results,
)
from skill_eval.schema import EvalSpec, load_eval_spec


def discover_eval_paths(paths: list[str]) -> list[Path]:
    """Resolve skill directories or eval.yaml files to concrete eval paths."""
    if not paths:
        return sorted(Path("skills").glob("*/evals/eval.yaml"))

    resolved: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            if (path / "evals" / "eval.yaml").is_file():
                resolved.append(path / "evals" / "eval.yaml")
            elif path.name == "evals" and (path / "eval.yaml").is_file():
                resolved.append(path / "eval.yaml")
            else:
                raise FileNotFoundError(f"Could not find eval.yaml under {path}")
        elif path.is_file():
            resolved.append(path)
        else:
            raise FileNotFoundError(f"Path does not exist: {path}")
    return resolved


def run_eval(
    path: Path,
    *,
    runtime_command: str,
    judge_command: str | None = None,
    models: list[str] | None = None,
    variant: str = "single",
    enable_skill: bool | None = None,
    status_stream: TextIO | None = None,
    verbose: bool = False,
    debug_dir: Path | None = None,
) -> EvalRunResult:
    """Run all test cases for one eval spec across the configured models."""
    eval_path = discover_eval_paths([str(path)])[0]
    spec = load_eval_spec(eval_path)
    selected_models = models or spec.models
    if not selected_models:
        raise ValueError("No models configured; provide them in eval.yaml or via --model")

    all_results: list[TestCaseRunResult] = []
    model_summaries = []
    for model in selected_models:
        _emit_status(
            status_stream,
            f"[{variant}] model {model} starting ({len(spec.test_cases)} test case(s))",
        )
        model_results = [
            _run_test_case(
                spec,
                test_case,
                model=model,
                runtime_command=runtime_command,
                judge_command=judge_command,
                variant=variant,
                enable_skill=enable_skill,
                index=index,
                total=len(spec.test_cases),
                status_stream=status_stream,
                verbose=verbose,
                debug_dir=debug_dir,
            )
            for index, test_case in enumerate(spec.test_cases, start=1)
        ]
        all_results.extend(model_results)
        model_summaries.append(summarize_results(model, model_results))
        summary = model_summaries[-1]
        _emit_status(
            status_stream,
            (
                f"[{variant}] model {model} finished: "
                f"{summary.passed_tests}/{summary.total_tests} passed, "
                f"avg score {summary.average_score:.2f}"
            ),
        )

    return EvalRunResult(
        variant=variant,
        skill_name=spec.skill_name,
        eval_version=spec.eval_version,
        model_summaries=model_summaries,
        test_results=all_results,
        runtime_command=runtime_command,
        judge_command=judge_command,
    )


def run_ab_eval(
    path: Path,
    *,
    runtime_command: str,
    judge_command: str | None = None,
    models: list[str] | None = None,
    baseline_runtime_command: str | None = None,
    treatment_runtime_command: str | None = None,
    status_stream: TextIO | None = None,
    verbose: bool = False,
    debug_dir: Path | None = None,
) -> EvalABResult:
    """Run an eval twice, once without the skill and once with it."""
    baseline_result = run_eval(
        path,
        runtime_command=baseline_runtime_command or runtime_command,
        judge_command=judge_command,
        models=models,
        variant="baseline",
        enable_skill=False,
        status_stream=status_stream,
        verbose=verbose,
        debug_dir=debug_dir,
    )
    treatment_result = run_eval(
        path,
        runtime_command=treatment_runtime_command or runtime_command,
        judge_command=judge_command,
        models=models,
        variant="treatment",
        enable_skill=True,
        status_stream=status_stream,
        verbose=verbose,
        debug_dir=debug_dir,
    )
    return EvalABResult(
        skill_name=treatment_result.skill_name,
        eval_version=treatment_result.eval_version,
        baseline=baseline_result,
        treatment=treatment_result,
        comparisons=compare_model_summaries(baseline_result, treatment_result),
    )


def _run_test_case(
    spec: EvalSpec,
    test_case,
    *,
    model: str,
    runtime_command: str,
    judge_command: str | None,
    variant: str,
    enable_skill: bool | None,
    index: int,
    total: int,
    status_stream: TextIO | None,
    verbose: bool,
    debug_dir: Path | None,
) -> TestCaseRunResult:
    eval_dir = spec.path.parent
    repo_root = spec.path.resolve().parents[3]
    artifact_dir = _build_artifact_dir(
        debug_dir,
        variant=variant,
        model=model,
        test_case_name=test_case.name,
    )
    _emit_status(
        status_stream,
        f"[{variant}] {model} [{index}/{total}] {test_case.name} starting",
    )
    with tempfile.TemporaryDirectory(
        prefix=f"skill-eval-{spec.skill_name}-{test_case.name}-"
    ) as tmpdir:
        workdir = Path(tmpdir)
        if test_case.fixture is not None:
            _copy_fixture(eval_dir / test_case.fixture, workdir)

        for command in test_case.setup:
            subprocess.run(command, shell=True, cwd=workdir, check=True)

        env = os.environ.copy()
        env.update(
            {
                "SKILL_EVAL_MODEL": model,
                "SKILL_EVAL_SKILL_NAME": spec.skill_name,
                "SKILL_EVAL_SKILL_PATH": str(repo_root / spec.skill_path),
                "SKILL_EVAL_SKILL_PATH_REPO_REL": spec.skill_path,
                "SKILL_EVAL_EVAL_DIR": str(eval_dir),
                "SKILL_EVAL_WORKDIR": str(workdir),
                "SKILL_EVAL_PROMPT": test_case.user_message,
                "SKILL_EVAL_TEST_CASE": test_case.name,
                "SKILL_EVAL_VARIANT": variant,
                "SKILL_EVAL_REPO_ROOT": str(repo_root),
                "PYTHONPATH": _prepend_pythonpath(str(repo_root), env.get("PYTHONPATH")),
            },
        )
        if enable_skill is not None:
            env["SKILL_EVAL_ENABLE_SKILL"] = "1" if enable_skill else "0"
        if verbose:
            env["SKILL_EVAL_VERBOSE"] = "1"
            _emit_status(status_stream, f"[{variant}] runtime command: {runtime_command}")
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            env["SKILL_EVAL_ARTIFACT_DIR"] = str(artifact_dir)
            _emit_status(status_stream, f"[{variant}] artifacts: {artifact_dir}")
        started = time.perf_counter()
        completed = subprocess.run(
            runtime_command,
            shell=True,
            cwd=workdir,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        duration_seconds = time.perf_counter() - started
        agent_output, tools_invoked = _parse_runtime_output(completed.stdout)
        _write_runner_artifacts(
            artifact_dir,
            runtime_command=runtime_command,
            completed=completed,
            duration_seconds=duration_seconds,
            variant=variant,
            model=model,
            test_case_name=test_case.name,
        )
        if verbose and completed.stderr.strip():
            _emit_status(
                status_stream,
                (
                    f"[{variant}] {model} [{index}/{total}] runtime stderr follows:\n"
                    f"{completed.stderr.rstrip()}"
                ),
            )

        deterministic_results = run_deterministic_checks(
            test_case.deterministic_checks,
            workdir=workdir,
            agent_output=agent_output,
            tools_invoked=tools_invoked,
        )
        checker_result = None
        if test_case.checker is not None:
            checker_result = run_checker(
                test_case.checker,
                eval_dir=eval_dir,
                workdir=workdir,
                agent_output=agent_output,
                model=model,
                test_case_name=test_case.name,
            )

        judge_result = None
        if test_case.llm_judge is not None and completed.returncode == 0 and judge_command:
            judge_result = run_llm_judge(
                test_case.llm_judge,
                judge_command=judge_command,
                workdir=workdir,
                model=model,
                agent_output=agent_output,
                user_message=test_case.user_message,
            )

        passed, score = compute_case_score(
            runtime_exit_code=completed.returncode,
            deterministic_checks=deterministic_results,
            checker_result=checker_result,
            judge_result=judge_result,
        )
        _emit_status(
            status_stream,
            (
                f"[{variant}] {model} [{index}/{total}] {test_case.name} "
                f"{'passed' if passed else 'failed'} in {duration_seconds:.2f}s"
            ),
        )
        if verbose:
            _emit_verbose_details(
                status_stream,
                variant=variant,
                agent_output=agent_output,
                tools_invoked=tools_invoked,
                deterministic_results=deterministic_results,
                checker_result=checker_result,
                judge_result=judge_result,
                score=score,
            )
        return TestCaseRunResult(
            variant=variant,
            test_case_name=test_case.name,
            model=model,
            passed=passed,
            score=score,
            duration_seconds=duration_seconds,
            exit_code=completed.returncode,
            agent_output=agent_output,
            agent_error=completed.stderr,
            deterministic_checks=deterministic_results,
            checker_result=checker_result,
            judge_result=judge_result,
        )


def _copy_fixture(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Fixture path does not exist: {source}")
    if source.is_file():
        shutil.copy2(source, destination / source.name)
        return
    for child in source.iterdir():
        child_destination = destination / child.name
        if child.is_dir():
            shutil.copytree(child, child_destination, dirs_exist_ok=True)
        else:
            shutil.copy2(child, child_destination)


def _prepend_pythonpath(repo_root: str, existing_pythonpath: str | None) -> str:
    if existing_pythonpath:
        return f"{repo_root}{os.pathsep}{existing_pythonpath}"
    return repo_root


def _emit_status(status_stream: TextIO | None, message: str) -> None:
    if status_stream is None:
        return
    print(message, file=status_stream, flush=True)


def _emit_verbose_details(
    status_stream: TextIO | None,
    *,
    variant: str,
    agent_output: str,
    tools_invoked: list[dict],
    deterministic_results: list[DeterministicCheckResult],
    checker_result: CheckerResult | None,
    judge_result: JudgeResult | None,
    score: float,
) -> None:
    prefix = f"[{variant}]"
    preview = agent_output.strip()[:500] or "(empty)"
    _emit_status(status_stream, f"{prefix} agent output: {preview}")
    if tools_invoked:
        tool_names = [
            f"Skill({t['skill']})"
            if t.get("tool") == "Skill" and t.get("skill")
            else t.get("tool", "?")
            for t in tools_invoked
        ]
        _emit_status(status_stream, f"{prefix} tools invoked: {', '.join(tool_names)}")
    if deterministic_results:
        for check in deterministic_results:
            status = "PASS" if check.passed else "FAIL"
            _emit_status(
                status_stream,
                f"{prefix}   {check.check_type}: {status} - {check.details}",
            )
    if checker_result is not None:
        status = "PASS" if checker_result.passed else "FAIL"
        detail = f" - {checker_result.details}" if checker_result.details else ""
        _emit_status(status_stream, f"{prefix}   checker {checker_result.name}: {status}{detail}")
    if judge_result is not None:
        for req in judge_result.critical_requirements:
            status = "PASS" if req.passed else "FAIL"
            evidence = f" ({req.evidence})" if req.evidence else ""
            _emit_status(
                status_stream,
                f"{prefix}   requirement {req.requirement!r}: {status}{evidence}",
            )
        for factor in judge_result.quality_factors:
            _emit_status(
                status_stream,
                f"{prefix}   quality {factor.factor!r}: {factor.score:.2f}",
            )
    _emit_status(status_stream, f"{prefix} score: {score:.2f}")


def _parse_runtime_output(stdout: str) -> tuple[str, list[dict]]:
    """Parse runtime stdout, handling both envelope JSON and plain text."""
    text = stdout.strip()
    if not text:
        return "", []
    try:
        envelope = json.loads(text)
        if isinstance(envelope, dict) and "result_text" in envelope:
            return envelope["result_text"], envelope.get("tools_invoked", [])
    except json.JSONDecodeError:
        pass
    return text, []


def _build_artifact_dir(
    debug_dir: Path | None,
    *,
    variant: str,
    model: str,
    test_case_name: str,
) -> Path | None:
    if debug_dir is None:
        return None
    return (
        debug_dir
        / _sanitize_path_component(variant)
        / _sanitize_path_component(model)
        / _sanitize_path_component(test_case_name)
    )


def _sanitize_path_component(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


def _write_runner_artifacts(
    artifact_dir: Path | None,
    *,
    runtime_command: str,
    completed: subprocess.CompletedProcess[str],
    duration_seconds: float,
    variant: str,
    model: str,
    test_case_name: str,
) -> None:
    if artifact_dir is None:
        return

    artifact_dir.joinpath("runtime_stdout.txt").write_text(completed.stdout)
    artifact_dir.joinpath("runtime_stderr.txt").write_text(completed.stderr)
    artifact_dir.joinpath("runner_metadata.json").write_text(
        json.dumps(
            {
                "variant": variant,
                "model": model,
                "test_case_name": test_case_name,
                "runtime_command": runtime_command,
                "exit_code": completed.returncode,
                "duration_seconds": duration_seconds,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
