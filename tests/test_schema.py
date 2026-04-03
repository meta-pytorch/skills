from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from skill_eval.schema import SkillEvalValidationError, load_eval_spec


class SchemaTest(unittest.TestCase):
    def test_sample_eval_loads(self) -> None:
        spec = load_eval_spec(Path("skills/agents-are-coming/evals/eval.yaml"))
        self.assertEqual(spec.skill_name, "agents-are-coming")
        self.assertEqual(spec.models, ["claude-opus-4-6"])
        self.assertEqual(len(spec.test_cases), 1)

    def test_duplicate_test_case_names_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            eval_path = Path(tmpdir) / "eval.yaml"
            eval_path.write_text(
                "\n".join(
                    [
                        'eval_version: "1"',
                        'skill_name: "agents-are-coming"',
                        'skill_path: "skills/agents-are-coming/SKILL.md"',
                        "models:",
                        '  - "claude-opus-4-6"',
                        "test_cases:",
                        '  - name: "same"',
                        '    user_message: "first"',
                        "    deterministic_checks:",
                        '      - type: "output_contains_ci"',
                        '        value: "first"',
                        '  - name: "same"',
                        '    user_message: "second"',
                        "    deterministic_checks:",
                        '      - type: "output_contains_ci"',
                        '        value: "second"',
                        "",
                    ]
                )
            )
            with self.assertRaisesRegex(SkillEvalValidationError, "duplicate test case names"):
                load_eval_spec(eval_path)


if __name__ == "__main__":
    unittest.main()
