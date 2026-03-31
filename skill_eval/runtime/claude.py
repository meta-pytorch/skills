# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Claude Code CLI runtime for skill evals."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def main() -> int:
    try:
        prompt = os.environ["SKILL_EVAL_PROMPT"]
        model = os.environ["SKILL_EVAL_MODEL"]
        workdir = Path(os.environ["SKILL_EVAL_WORKDIR"])
        skill_enabled = os.environ.get("SKILL_EVAL_ENABLE_SKILL", "1") != "0"
        repo_root = Path(os.environ["SKILL_EVAL_REPO_ROOT"])
        verbose = os.environ.get("SKILL_EVAL_VERBOSE") == "1"
        artifact_dir = os.environ.get("SKILL_EVAL_ARTIFACT_DIR")
    except KeyError as exc:
        print(f"Missing required environment variable: {exc.args[0]}", file=sys.stderr)
        return 2

    if shutil.which("claude") is None:
        print("Claude runtime failed: 'claude' binary not found on PATH", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(
        prefix="skill-eval-claude-home-",
        ignore_cleanup_errors=True,
    ) as tmp_home:
        env = build_runtime_env(tmp_home)
        command = build_claude_command(
            prompt=prompt,
            model=model,
            skill_enabled=skill_enabled,
            repo_root=repo_root,
        )
        if verbose:
            print(f"Claude command: {' '.join(command)}", file=sys.stderr)
        completed = subprocess.run(
            command,
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    write_debug_artifacts(
        Path(artifact_dir) if artifact_dir else None,
        command=command,
        completed=completed,
    )

    if completed.returncode != 0:
        print(
            f"Claude runtime failed: {completed.stderr.strip() or completed.stdout.strip()}",
            file=sys.stderr,
        )
        return completed.returncode or 1

    result_text, tools_invoked = parse_stream_output(completed.stdout)
    print(json.dumps({"result_text": result_text, "tools_invoked": tools_invoked}))
    return 0


def build_claude_command(
    *,
    prompt: str,
    model: str,
    skill_enabled: bool,
    repo_root: Path,
) -> list[str]:
    command = [
        "claude",
        "-p",
        prompt,
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "bypassPermissions",
        "--no-session-persistence",
    ]
    # Only add macOS sandbox flag on macOS
    if sys.platform == "darwin":
        command.insert(1, "--dangerously-disable-osx-sandbox")
    if skill_enabled:
        command.extend(["--plugin-dir", str(repo_root)])
    return command


def build_runtime_env(tmp_home: str) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = tmp_home
    env["XDG_CONFIG_HOME"] = str(Path(tmp_home) / ".config")
    env["XDG_CACHE_HOME"] = str(Path(tmp_home) / ".cache")
    env["XDG_DATA_HOME"] = str(Path(tmp_home) / ".local" / "share")
    return env


def parse_stream_output(stdout: str) -> tuple[str, list[dict[str, Any]]]:
    """Parse stream-json output to extract result text and tool invocations."""
    result_text = ""
    tools_invoked: list[dict[str, Any]] = []

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        if event.get("type") == "result":
            result_text = event.get("result", "")

        if event.get("type") == "assistant":
            message = event.get("message", {})
            for block in message.get("content", []):
                if block.get("type") == "tool_use":
                    tool_record = {
                        "tool": block.get("name", ""),
                        "id": block.get("id", ""),
                    }
                    tool_input = block.get("input", {})
                    if block.get("name") == "Skill":
                        tool_record["skill"] = tool_input.get("skill", "")
                    tools_invoked.append(tool_record)

    return result_text, tools_invoked


def write_debug_artifacts(
    artifact_dir: Path | None,
    *,
    command: list[str],
    completed: subprocess.CompletedProcess[str],
) -> None:
    if artifact_dir is None:
        return
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.joinpath("claude_command.txt").write_text(" ".join(command) + "\n")
    artifact_dir.joinpath("claude_stdout.txt").write_text(completed.stdout)
    artifact_dir.joinpath("claude_stderr.txt").write_text(completed.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
