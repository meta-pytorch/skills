# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from skill_eval.checks import run_deterministic_check
from skill_eval.schema import DeterministicCheckSpec


class DeterministicChecksTest(unittest.TestCase):
    def test_output_equals(self) -> None:
        check = DeterministicCheckSpec(type="output_equals", params={"value": "hello"})
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_deterministic_check(
                check, workdir=Path(tmpdir), tools_invoked=[], agent_output="hello"
            )
        self.assertTrue(result.passed)

    def test_output_contains_ci(self) -> None:
        check = DeterministicCheckSpec(
            type="output_contains_ci", params={"value": "agents are coming"}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_deterministic_check(
                check,
                workdir=Path(tmpdir),
                tools_invoked=[],
                agent_output="**AGENTS ARE COMING!!!**",
            )
        self.assertTrue(result.passed)

    def test_file_json_equals(self) -> None:
        check = DeterministicCheckSpec(
            type="file_json_equals",
            params={"path": "result.json", "key": "answer", "value": 7},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "result.json").write_text(json.dumps({"answer": 7}))
            result = run_deterministic_check(
                check, workdir=Path(tmpdir), tools_invoked=[], agent_output=""
            )
        self.assertTrue(result.passed)

    def test_file_json_equals_fails_closed_on_missing_key(self) -> None:
        check = DeterministicCheckSpec(
            type="file_json_equals",
            params={"path": "result.json", "key": "answer", "value": 7},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "result.json").write_text(json.dumps({"wrong": 7}))
            result = run_deterministic_check(
                check, workdir=Path(tmpdir), tools_invoked=[], agent_output=""
            )
        self.assertFalse(result.passed)
        self.assertIn("error=", result.details)

    def test_skill_invoked_accepts_namespaced_skill_ids(self) -> None:
        check = DeterministicCheckSpec(
            type="skill_invoked",
            params={"skill": "agents-are-coming"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_deterministic_check(
                check,
                workdir=Path(tmpdir),
                tools_invoked=[{"tool": "Skill", "skill": "meta:agents-are-coming"}],
                agent_output="",
            )
        self.assertTrue(result.passed)


if __name__ == "__main__":
    unittest.main()
