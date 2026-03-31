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
            skills_dir = root / "skills" / "foo"
            skills_dir.mkdir(parents=True)
            (skills_dir / "SKILL.md").write_text("---\ndescription: hi\n---\n\n# Foo\n")
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
                mock.patch.object(validate_plugin, "SKILLS_DIR", root / "skills"),
            ):
                errors = validate_plugin.validate_skills_consistency(marketplace_path, plugin_path)
        self.assertIn("skills/foo/SKILL.md: missing 'name' field", errors)

    def test_validate_skills_consistency_requires_skill_to_be_shipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            plugin_dir = root / ".claude-plugin"
            plugin_dir.mkdir()
            skills_dir = root / "skills" / "foo"
            skills_dir.mkdir(parents=True)
            (skills_dir / "SKILL.md").write_text("---\nname: foo\ndescription: hi\n---\n\n# Foo\n")
            (root / "published-skills").mkdir()
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
                        "skills": ["./published-skills"],
                    }
                )
            )
            with (
                mock.patch.object(validate_plugin, "ROOT", root),
                mock.patch.object(validate_plugin, "PLUGIN_DIR", plugin_dir),
                mock.patch.object(validate_plugin, "SKILLS_DIR", root / "skills"),
            ):
                errors = validate_plugin.validate_skills_consistency(marketplace_path, plugin_path)
        self.assertIn("skills/foo is not covered by any plugin.json skills root", errors)


if __name__ == "__main__":
    unittest.main()
