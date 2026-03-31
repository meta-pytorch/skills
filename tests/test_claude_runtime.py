# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess

from skill_eval.runtime import claude


class ClaudeRuntimeTest(unittest.TestCase):
    def test_build_claude_command_adds_plugin_dir_for_treatment(self) -> None:
        command = claude.build_claude_command(
            prompt="/agents-are-coming",
            model="claude-opus-4-6",
            skill_enabled=True,
            repo_root=Path("/tmp/repo"),
        )
        self.assertIn("--plugin-dir", command)
        self.assertIn("/tmp/repo", command)
        self.assertIn("stream-json", command)
        self.assertIn("--verbose", command)

    def test_build_claude_command_omits_plugin_dir_for_baseline(self) -> None:
        command = claude.build_claude_command(
            prompt="/agents-are-coming",
            model="claude-opus-4-6",
            skill_enabled=False,
            repo_root=Path("/tmp/repo"),
        )
        self.assertNotIn("--plugin-dir", command)

    def test_parse_stream_output_extracts_result(self) -> None:
        text, tools = claude.parse_stream_output(
            '{"type":"result","result":"**AGENTS ARE COMING!!!**"}'
        )
        self.assertEqual(text, "**AGENTS ARE COMING!!!**")
        self.assertEqual(tools, [])

    def test_parse_stream_output_extracts_skill_tool_use(self) -> None:
        lines = [
            '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Skill","id":"tu1","input":{"skill":"agents-are-coming"}}]}}',
            '{"type":"result","result":"done"}',
        ]
        text, tools = claude.parse_stream_output("\n".join(lines))
        self.assertEqual(text, "done")
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["tool"], "Skill")
        self.assertEqual(tools[0]["skill"], "agents-are-coming")

    def test_write_debug_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            claude.write_debug_artifacts(
                artifact_dir,
                command=["claude", "-p", "hello"],
                completed=CompletedProcess(
                    args=["claude"],
                    returncode=0,
                    stdout='{"result":"hello"}',
                    stderr="",
                ),
            )
            self.assertEqual(
                artifact_dir.joinpath("claude_command.txt").read_text(), "claude -p hello\n"
            )
            self.assertIn("hello", artifact_dir.joinpath("claude_stdout.txt").read_text())


if __name__ == "__main__":
    unittest.main()
