"""Validate .claude-plugin/ manifests and cross-check with skills/ directory."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLUGIN_DIR = ROOT / ".claude-plugin"
SKILLS_DIR = ROOT / "skills"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def validate_marketplace(path: Path) -> list[str]:
    errors: list[str] = []
    data = load_json(path)

    for field in ("name", "owner", "plugins"):
        if field not in data:
            errors.append(f"marketplace.json: missing required field '{field}'")

    if not isinstance(data.get("plugins"), list):
        errors.append("marketplace.json: 'plugins' must be an array")
        return errors

    ALLOWED_PLUGIN_FIELDS = {
        "name",
        "description",
        "version",
        "author",
        "source",
        "category",
        "homepage",
        "strict",
        "tags",
        "lspServers",
    }

    for i, plugin in enumerate(data["plugins"]):
        prefix = f"marketplace.json plugins[{i}]"
        for field in ("name", "description", "source"):
            if field not in plugin:
                errors.append(f"{prefix}: missing required field '{field}'")

        unknown = set(plugin.keys()) - ALLOWED_PLUGIN_FIELDS
        if unknown:
            errors.append(f"{prefix}: unknown fields {unknown} (will fail schema validation)")

    return errors


def validate_plugin_json(path: Path) -> list[str]:
    errors: list[str] = []
    data = load_json(path)

    for field in ("name", "description"):
        if field not in data:
            errors.append(f"plugin.json: missing required field '{field}'")

    if "skills" in data and not isinstance(data["skills"], list):
        errors.append("plugin.json: 'skills' must be an array of paths")

    return errors


def validate_skills_consistency(marketplace_path: Path, plugin_path: Path) -> list[str]:
    errors: list[str] = []

    marketplace = load_json(marketplace_path)
    plugin = load_json(plugin_path)

    marketplace_plugins = {p["name"] for p in marketplace.get("plugins", [])}
    plugin_name = plugin.get("name", "")
    if plugin_name and plugin_name not in marketplace_plugins:
        errors.append(f"plugin.json name '{plugin_name}' not found in marketplace.json plugins")

    skill_dirs = {d.name for d in SKILLS_DIR.iterdir() if d.is_dir() and (d / "SKILL.md").exists()}

    for skill_name in skill_dirs:
        skill_md = SKILLS_DIR / skill_name / "SKILL.md"
        content = skill_md.read_text()

        in_frontmatter = False
        name_value = None
        has_description = False

        for line in content.splitlines():
            if line.strip() == "---":
                if in_frontmatter:
                    break
                in_frontmatter = True
                continue
            if in_frontmatter:
                if line.startswith("name:"):
                    name_value = line.split(":", 1)[1].strip()
                if line.startswith("description:"):
                    has_description = True

        if name_value and name_value != skill_name:
            errors.append(
                f"skills/{skill_name}/SKILL.md: name '{name_value}' != directory '{skill_name}'"
            )
        if not has_description:
            errors.append(f"skills/{skill_name}/SKILL.md: missing 'description' field")

    return errors


def main() -> int:
    errors: list[str] = []

    marketplace_path = PLUGIN_DIR / "marketplace.json"
    plugin_path = PLUGIN_DIR / "plugin.json"

    if not marketplace_path.exists():
        errors.append("Missing .claude-plugin/marketplace.json")
    else:
        errors.extend(validate_marketplace(marketplace_path))

    if not plugin_path.exists():
        errors.append("Missing .claude-plugin/plugin.json")
    else:
        errors.extend(validate_plugin_json(plugin_path))

    if marketplace_path.exists() and plugin_path.exists():
        errors.extend(validate_skills_consistency(marketplace_path, plugin_path))

    if errors:
        print(f"Found {len(errors)} error(s):\n")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("All plugin validations passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
