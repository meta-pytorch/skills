from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import validate_plugin


class ValidatePluginTest(unittest.TestCase):
    def test_validate_marketplace_reports_non_object_plugin_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "marketplace.json"
            path.write_text(json.dumps({"name": "x", "owner": {}, "plugins": ["bad-entry"]}))
            errors = validate_plugin.validate_marketplace(path)
        self.assertIn("marketplace.json plugins[0]: plugin entry must be an object", errors)

    def test_validate_plugin_json_requires_existing_skills_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "plugin.json"
            path.write_text(json.dumps({"name": "x", "description": "y", "skills": ["./missing"]}))
            with mock.patch.object(validate_plugin, "ROOT", root):
                errors = validate_plugin.validate_plugin_json(path)
        self.assertIn("plugin.json skills[0]: skill path './missing' does not exist", errors)

    def test_validate_skills_consistency_requires_name_frontmatter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plugin_dir = root / ".claude-plugin"
            plugin_dir.mkdir()
            source_skills_dir = root / "skills-with-evals" / "foo"
            source_skills_dir.mkdir(parents=True)
            (source_skills_dir / "SKILL.md").write_text("---\ndescription: hi\n---\n\n# Foo\n")
            marketplace_path = plugin_dir / "marketplace.json"
            marketplace_path.write_text(
                json.dumps(
                    {
                        "name": "skills",
                        "owner": {},
                        "plugins": [{"name": "pytorch-skills", "description": "d", "source": "./"}],
                    }
                )
            )
            plugin_path = plugin_dir / "plugin.json"
            plugin_path.write_text(
                json.dumps(
                    {
                        "name": "pytorch-skills",
                        "description": "d",
                        "skills": ["./skills"],
                    }
                )
            )
            with (
                mock.patch.object(validate_plugin, "ROOT", root),
                mock.patch.object(validate_plugin, "PLUGIN_DIR", plugin_dir),
                mock.patch.object(validate_plugin, "SOURCE_SKILLS_DIR", root / "skills-with-evals"),
                mock.patch.object(validate_plugin, "SHIPPED_SKILLS_DIR", root / "skills"),
            ):
                errors = validate_plugin.validate_skills_consistency(marketplace_path, plugin_path)
        self.assertIn("skills-with-evals/foo/SKILL.md: missing 'name' field", errors)

    def test_validate_skills_consistency_requires_skill_to_be_shipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plugin_dir = root / ".claude-plugin"
            plugin_dir.mkdir()
            source_skills_dir = root / "skills-with-evals" / "foo"
            source_skills_dir.mkdir(parents=True)
            (source_skills_dir / "SKILL.md").write_text(
                "---\nname: foo\ndescription: hi\n---\n\n# Foo\n"
            )
            (root / "skills").mkdir()
            marketplace_path = plugin_dir / "marketplace.json"
            marketplace_path.write_text(
                json.dumps(
                    {
                        "name": "skills",
                        "owner": {},
                        "plugins": [{"name": "pytorch-skills", "description": "d", "source": "./"}],
                    }
                )
            )
            plugin_path = plugin_dir / "plugin.json"
            plugin_path.write_text(
                json.dumps(
                    {
                        "name": "pytorch-skills",
                        "description": "d",
                        "skills": ["./skills"],
                    }
                )
            )
            with (
                mock.patch.object(validate_plugin, "ROOT", root),
                mock.patch.object(validate_plugin, "PLUGIN_DIR", plugin_dir),
                mock.patch.object(validate_plugin, "SOURCE_SKILLS_DIR", root / "skills-with-evals"),
                mock.patch.object(validate_plugin, "SHIPPED_SKILLS_DIR", root / "skills"),
            ):
                errors = validate_plugin.validate_skills_consistency(marketplace_path, plugin_path)
        self.assertIn("skills/foo/SKILL.md is missing", errors)

    def test_validate_skills_consistency_rejects_shipped_evals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plugin_dir = root / ".claude-plugin"
            plugin_dir.mkdir()
            source_skills_dir = root / "skills-with-evals" / "foo"
            source_skills_dir.mkdir(parents=True)
            (source_skills_dir / "SKILL.md").write_text(
                "---\nname: foo\ndescription: hi\n---\n\n# Foo\n"
            )
            shipped_skill_dir = root / "skills" / "foo" / "evals"
            shipped_skill_dir.mkdir(parents=True)
            shipped_skill_dir.parent.joinpath("SKILL.md").write_text(
                "---\nname: foo\ndescription: hi\n---\n\n# Foo\n"
            )
            marketplace_path = plugin_dir / "marketplace.json"
            marketplace_path.write_text(
                json.dumps(
                    {
                        "name": "skills",
                        "owner": {},
                        "plugins": [{"name": "pytorch-skills", "description": "d", "source": "./"}],
                    }
                )
            )
            plugin_path = plugin_dir / "plugin.json"
            plugin_path.write_text(
                json.dumps(
                    {
                        "name": "pytorch-skills",
                        "description": "d",
                        "skills": ["./skills"],
                    }
                )
            )
            with (
                mock.patch.object(validate_plugin, "ROOT", root),
                mock.patch.object(validate_plugin, "PLUGIN_DIR", plugin_dir),
                mock.patch.object(validate_plugin, "SOURCE_SKILLS_DIR", root / "skills-with-evals"),
                mock.patch.object(validate_plugin, "SHIPPED_SKILLS_DIR", root / "skills"),
            ):
                errors = validate_plugin.validate_skills_consistency(marketplace_path, plugin_path)
        self.assertIn("skills/foo must not contain an evals directory", errors)

    def test_validate_generated_plugin_dist_reports_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plugin_dir = root / ".claude-plugin"
            plugin_dir.mkdir()
            plugin_path = plugin_dir / "plugin.json"
            plugin_path.write_text(
                json.dumps(
                    {
                        "name": "pytorch-skills",
                        "description": "d",
                        "skills": ["./skills"],
                    }
                )
            )

            shipped_skill_dir = root / "skills" / "foo"
            shipped_skill_dir.mkdir(parents=True)
            shipped_skill_dir.joinpath("SKILL.md").write_text(
                "---\nname: foo\ndescription: hi\n---\n\n# Foo\n"
            )

            generated_plugin_dir = root / "claude-plugin-dist"
            generated_skills_dir = generated_plugin_dir / "skills" / "foo"
            generated_skills_dir.mkdir(parents=True)
            (generated_plugin_dir / "marketplace.json").write_text(
                json.dumps(
                    {
                        "name": "pytorch-skills",
                        "description": "different",
                        "skills": ["./skills"],
                    }
                )
            )
            generated_skills_dir.joinpath("SKILL.md").write_text(
                "---\nname: foo\ndescription: changed\n---\n\n# Foo\n"
            )

            with (
                mock.patch.object(validate_plugin, "SHIPPED_SKILLS_DIR", root / "skills"),
                mock.patch.object(validate_plugin, "GENERATED_PLUGIN_DIR", generated_plugin_dir),
                mock.patch.object(
                    validate_plugin,
                    "GENERATED_PLUGIN_SKILLS_DIR",
                    generated_plugin_dir / "skills",
                ),
            ):
                errors = validate_plugin.validate_generated_plugin_dist(
                    plugin_path,
                    generated_plugin_dir / "marketplace.json",
                )

        self.assertIn(
            "claude-plugin-dist/marketplace.json does not match .claude-plugin/plugin.json",
            errors,
        )
        self.assertIn(
            "claude-plugin-dist/skills/foo/SKILL.md does not match skills/foo/SKILL.md",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
