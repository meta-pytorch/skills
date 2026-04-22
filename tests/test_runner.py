from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from skill_eval.runner import run_ab_eval, run_eval

FAKE_AGENT_COMMAND = f"python3 {Path('tests/fake_agent.py').resolve()}"
SAMPLE_SKILL_PATH = Path("skills-with-evals/agents-are-coming")


class RunnerTest(unittest.TestCase):
    def test_sample_eval_runs_with_fake_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "result.json"
            result = run_eval(
                SAMPLE_SKILL_PATH,
                runtime_command=FAKE_AGENT_COMMAND,
            )
            result.write_json(output_path)
            self.assertEqual(result.skill_name, "agents-are-coming")
            self.assertEqual(len(result.model_summaries), 1)
            self.assertEqual(result.model_summaries[0].passed_tests, 1)
            self.assertTrue(output_path.exists())

    def test_sample_eval_runs_ab(self) -> None:
        result = run_ab_eval(
            SAMPLE_SKILL_PATH,
            runtime_command=FAKE_AGENT_COMMAND,
        )
        self.assertEqual(result.baseline.model_summaries[0].passed_tests, 0)
        self.assertEqual(result.treatment.model_summaries[0].passed_tests, 1)
        self.assertGreater(result.comparisons[0].delta_average_score, 0.0)

    def test_non_zero_runtime_exit_fails_the_test_case(self) -> None:
        result = run_eval(
            SAMPLE_SKILL_PATH,
            runtime_command=("python3 -c \"import sys; print('agents are coming'); sys.exit(1)\""),
        )
        self.assertEqual(result.model_summaries[0].passed_tests, 0)
        self.assertEqual(result.test_results[0].exit_code, 1)
        self.assertFalse(result.test_results[0].passed)

    def test_run_eval_emits_status_updates(self) -> None:
        status = io.StringIO()
        run_eval(
            SAMPLE_SKILL_PATH,
            runtime_command=FAKE_AGENT_COMMAND,
            status_stream=status,
        )
        output = status.getvalue()
        self.assertIn("[single] model claude-opus-4-6 starting", output)
        self.assertIn("[single] claude-opus-4-6 [1/1] natural_prompt starting", output)
        self.assertIn("[single] claude-opus-4-6 [1/1] natural_prompt passed", output)

    def test_run_eval_writes_debug_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            debug_dir = Path(tmpdir) / "debug"
            run_eval(
                SAMPLE_SKILL_PATH,
                runtime_command=FAKE_AGENT_COMMAND,
                debug_dir=debug_dir,
            )
            artifact_dir = debug_dir / "single" / "claude-opus-4-6" / "natural_prompt"
            self.assertEqual(
                json.loads(artifact_dir.joinpath("runner_metadata.json").read_text())["exit_code"],
                0,
            )
            self.assertIn(
                "AGENTS ARE COMING",
                artifact_dir.joinpath("runtime_stdout.txt").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
