from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import new_skill


class NewSkillTest(unittest.TestCase):
    def test_create_skill_scaffold_writes_minimal_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            skills_dir = root / "skills"
            skills_dir.mkdir()
            with (
                mock.patch.object(new_skill, "ROOT", root),
                mock.patch.object(new_skill, "SKILLS_DIR", skills_dir),
            ):
                skill_dir = new_skill.create_skill_scaffold("graph-break-finder")

            self.assertTrue((skill_dir / "SKILL.md").is_file())
            self.assertTrue((skill_dir / "evals" / "eval.yaml").is_file())
            self.assertTrue((skill_dir / "evals" / "checkers.py").is_file())
            self.assertTrue((skill_dir / "evals" / "fixtures" / ".gitkeep").is_file())

    def test_create_skill_scaffold_rejects_bad_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "skill name"):
            new_skill.create_skill_scaffold("Bad Name")


if __name__ == "__main__":
    unittest.main()
