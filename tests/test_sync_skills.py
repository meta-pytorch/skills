from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import sync_skills


class SyncSkillsTest(unittest.TestCase):
    def test_rebuild_shipped_skills_strips_evals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_skills_dir = root / "skills-with-evals"
            skill_dir = source_skills_dir / "foo"
            eval_dir = skill_dir / "evals" / "fixtures"
            nested_eval_dir = skill_dir / "references" / "evals"
            eval_dir.mkdir(parents=True)
            nested_eval_dir.mkdir(parents=True)
            skill_dir.joinpath("SKILL.md").write_text("---\nname: foo\ndescription: hi\n---\n")
            skill_dir.joinpath("notes.txt").write_text("keep me")
            eval_dir.joinpath("data.txt").write_text("drop me")
            nested_eval_dir.joinpath("nested.txt").write_text("drop me too")

            with (
                mock.patch.object(sync_skills, "ROOT", root),
                mock.patch.object(sync_skills, "SOURCE_SKILLS_DIR", source_skills_dir),
                mock.patch.object(sync_skills, "SHIPPED_SKILLS_DIR", root / "skills"),
            ):
                shipped_skills = sync_skills.rebuild_shipped_skills()
                self.assertEqual([path.name for path in shipped_skills], ["foo"])
                shipped_skill_dir = root / "skills" / "foo"
                self.assertTrue((root / "skills" / "README.md").exists())
                self.assertTrue(shipped_skill_dir.joinpath("SKILL.md").exists())
                self.assertTrue(shipped_skill_dir.joinpath("notes.txt").exists())
                self.assertFalse(shipped_skill_dir.joinpath("evals").exists())
                self.assertFalse(shipped_skill_dir.joinpath("references", "evals").exists())

    def test_rebuild_shipped_skills_replaces_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_skills_dir = root / "skills-with-evals"
            skill_dir = source_skills_dir / "foo"
            skill_dir.mkdir(parents=True)
            skill_dir.joinpath("SKILL.md").write_text("---\nname: foo\ndescription: hi\n---\n")
            shipped_dir = root / "skills"
            shipped_dir.mkdir()
            shipped_dir.joinpath("stale.txt").write_text("remove me")

            with (
                mock.patch.object(sync_skills, "ROOT", root),
                mock.patch.object(sync_skills, "SOURCE_SKILLS_DIR", source_skills_dir),
                mock.patch.object(sync_skills, "SHIPPED_SKILLS_DIR", shipped_dir),
            ):
                sync_skills.rebuild_shipped_skills()
                self.assertFalse((root / "skills" / "stale.txt").exists())


if __name__ == "__main__":
    unittest.main()
