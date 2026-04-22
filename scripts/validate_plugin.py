"""Validate .claude-plugin/ manifests and cross-check source and shipped skill trees."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

ROOT = Path(__file__).resolve().parents[1]
PLUGIN_DIR = ROOT / ".claude-plugin"
SOURCE_SKILLS_DIR = ROOT / "skills-with-evals"
SHIPPED_SKILLS_DIR = ROOT / "skills"


def load_json(path: Path) -> object:
    with open(path) as f:
        return json.load(f)


def validate_marketplace(path: Path) -> list[str]:
    errors: list[str] = []
    data = load_json(path)
    if not isinstance(data, dict):
        return ["marketplace.json: top-level document must be an object"]

    for field in ("name", "owner", "plugins"):
        if field not in data:
            errors.append(f"marketplace.json: missing required field '{field}'")

    if not isinstance(data.get("plugins"), list):
        errors.append("marketplace.json: 'plugins' must be an array")
        return errors

    allowed_plugin_fields = {
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
        if not isinstance(plugin, dict):
            errors.append(f"{prefix}: plugin entry must be an object")
            continue
        for field in ("name", "description", "source"):
            if field not in plugin:
                errors.append(f"{prefix}: missing required field '{field}'")

        unknown = set(plugin.keys()) - allowed_plugin_fields
        if unknown:
            errors.append(f"{prefix}: unknown fields {unknown} (will fail schema validation)")

        source = plugin.get("source")
        if isinstance(source, str) and not (ROOT / source).resolve().exists():
            errors.append(f"{prefix}: source path {source!r} does not exist")

    return errors


def validate_plugin_json(path: Path) -> list[str]:
    errors: list[str] = []
    data = load_json(path)
    if not isinstance(data, dict):
        return ["plugin.json: top-level document must be an object"]

    for field in ("name", "description", "skills"):
        if field not in data:
            errors.append(f"plugin.json: missing required field '{field}'")

    skills = data.get("skills")
    if not isinstance(skills, list):
        errors.append("plugin.json: 'skills' must be an array of paths")
        return errors

    if not skills:
        errors.append("plugin.json: 'skills' must not be empty")
        return errors

    for index, skill_path in enumerate(skills):
        prefix = f"plugin.json skills[{index}]"
        if not isinstance(skill_path, str) or not skill_path.strip():
            errors.append(f"{prefix}: skill path must be a non-empty string")
            continue

        resolved_skill_path = resolve_skill_root(skill_path)
        if not resolved_skill_path.exists():
            errors.append(f"{prefix}: skill path {skill_path!r} does not exist")
            continue
        if not resolved_skill_path.is_dir():
            errors.append(f"{prefix}: skill path {skill_path!r} must point to a directory")

    return errors


def validate_skills_consistency(marketplace_path: Path, plugin_path: Path) -> list[str]:
    errors: list[str] = []

    marketplace = load_json(marketplace_path)
    plugin = load_json(plugin_path)

    if not isinstance(marketplace, dict) or not isinstance(plugin, dict):
        return ["Cannot validate manifest consistency until both manifests parse as objects"]

    marketplace_plugins = set()
    for index, entry in enumerate(marketplace.get("plugins", [])):
        if not isinstance(entry, dict):
            errors.append(f"marketplace.json plugins[{index}]: plugin entry must be an object")
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"marketplace.json plugins[{index}]: missing required field 'name'")
            continue
        marketplace_plugins.add(name)

    plugin_name = plugin.get("name", "")
    if plugin_name and plugin_name not in marketplace_plugins:
        errors.append(f"plugin.json name '{plugin_name}' not found in marketplace.json plugins")

    skill_roots = resolve_plugin_skill_roots(plugin)
    source_skill_dirs = discover_skill_dirs(SOURCE_SKILLS_DIR)
    shipped_skill_dirs = {path.name: path for path in discover_skill_dirs(SHIPPED_SKILLS_DIR)}

    for skill_dir in source_skill_dirs:
        skill_name = skill_dir.name
        skill_md = skill_dir / "SKILL.md"
        frontmatter = load_frontmatter(skill_md)
        name_value = frontmatter.get("name")
        if not isinstance(name_value, str) or not name_value:
            errors.append(f"skills-with-evals/{skill_name}/SKILL.md: missing 'name' field")
        elif name_value != skill_name:
            errors.append(
                "skills-with-evals/"
                f"{skill_name}/SKILL.md: name '{name_value}' != directory '{skill_name}'"
            )
        if not isinstance(frontmatter.get("description"), str):
            errors.append(f"skills-with-evals/{skill_name}/SKILL.md: missing 'description' field")

        shipped_skill_dir = shipped_skill_dirs.get(skill_name)
        if shipped_skill_dir is None:
            errors.append(f"skills/{skill_name}/SKILL.md is missing")
            continue
        if skill_roots and not any(
            shipped_skill_dir.resolve().is_relative_to(root) for root in skill_roots
        ):
            errors.append(f"skills/{skill_name} is not covered by any plugin.json skills root")
        shipped_eval_dirs = [
            path for path in shipped_skill_dir.rglob("*") if path.is_dir() and path.name == "evals"
        ]
        if shipped_eval_dirs:
            errors.append(f"skills/{skill_name} must not contain an evals directory")

    source_skill_names = {path.name for path in source_skill_dirs}
    unexpected_shipped_skills = sorted(set(shipped_skill_dirs) - source_skill_names)
    for skill_name in unexpected_shipped_skills:
        errors.append(f"skills/{skill_name} has no matching source skill")

    return errors


def discover_skill_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if d.is_dir() and (d / "SKILL.md").exists())


def resolve_plugin_skill_roots(plugin: dict) -> list[Path]:
    skills = plugin.get("skills", [])
    if not isinstance(skills, list):
        return []
    return [
        resolve_skill_root(skill_path)
        for skill_path in skills
        if isinstance(skill_path, str) and skill_path.strip()
    ]


def resolve_skill_root(skill_path: str) -> Path:
    return (ROOT / skill_path).resolve()


def load_frontmatter(path: Path) -> dict:
    lines = path.read_text().splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path}: missing YAML frontmatter")

    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break
    if end_index is None:
        raise ValueError(f"{path}: unterminated YAML frontmatter block")

    payload = yaml.safe_load("\n".join(lines[1:end_index]))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: frontmatter must be a mapping")
    return payload


def main() -> None:
    errors: list[str] = []

    marketplace_path = PLUGIN_DIR / "marketplace.json"
    plugin_path = PLUGIN_DIR / "plugin.json"
    manifest_errors = False

    if not marketplace_path.exists():
        errors.append("Missing .claude-plugin/marketplace.json")
        manifest_errors = True
    else:
        marketplace_errors = validate_marketplace(marketplace_path)
        errors.extend(marketplace_errors)
        manifest_errors = manifest_errors or bool(marketplace_errors)

    if not plugin_path.exists():
        errors.append("Missing .claude-plugin/plugin.json")
        manifest_errors = True
    else:
        plugin_errors = validate_plugin_json(plugin_path)
        errors.extend(plugin_errors)
        manifest_errors = manifest_errors or bool(plugin_errors)

    if marketplace_path.exists() and plugin_path.exists() and not manifest_errors:
        try:
            errors.extend(validate_skills_consistency(marketplace_path, plugin_path))
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        typer.secho(f"Found {len(errors)} error(s):", err=True, fg=typer.colors.RED)
        typer.echo("", err=True)
        for err in errors:
            typer.echo(f"  - {err}", err=True)
        raise typer.Exit(1)

    typer.secho("All plugin validations passed.", fg=typer.colors.GREEN)


def cli() -> None:
    typer.run(main)


if __name__ == "__main__":
    cli()
