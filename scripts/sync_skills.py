"""Rebuild generated shipped artifacts from source skills-with-evals/."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SKILLS_DIR = ROOT / "skills-with-evals"
SHIPPED_SKILLS_DIR = ROOT / "skills"
SOURCE_PLUGIN_MANIFEST = ROOT / ".claude-plugin" / "plugin.json"
GENERATED_PLUGIN_DIR = ROOT / "claude-plugin-dist"
GENERATED_PLUGIN_MANIFEST = GENERATED_PLUGIN_DIR / "marketplace.json"
SHIPPED_SKILLS_README = """# Skills

This directory is generated from `skills-with-evals/`.
Do not edit files here directly; your changes will be overwritten.

To update shipped skills, edit `skills-with-evals/` and run:

```bash
uv run sync-skills
```
"""


def should_skip(path: Path) -> bool:
    return path.name == "evals"


def copy_skill_tree(source: Path, destination: Path) -> None:
    for child in source.iterdir():
        if should_skip(child):
            continue
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, ignore=shutil.ignore_patterns("evals"))
            continue
        shutil.copy2(child, target)


def write_shipped_skills_readme() -> None:
    SHIPPED_SKILLS_DIR.joinpath("README.md").write_text(SHIPPED_SKILLS_README)


def discover_source_skill_dirs() -> list[Path]:
    if not SOURCE_SKILLS_DIR.exists():
        return []
    return sorted(path for path in SOURCE_SKILLS_DIR.iterdir() if (path / "SKILL.md").is_file())


def rebuild_shipped_skills() -> list[Path]:
    if SHIPPED_SKILLS_DIR.exists():
        shutil.rmtree(SHIPPED_SKILLS_DIR)
    SHIPPED_SKILLS_DIR.mkdir()
    write_shipped_skills_readme()

    shipped_skills: list[Path] = []
    for skill_dir in discover_source_skill_dirs():
        shipped_skill_dir = SHIPPED_SKILLS_DIR / skill_dir.name
        shipped_skill_dir.mkdir()
        copy_skill_tree(skill_dir, shipped_skill_dir)
        if not shipped_skill_dir.joinpath("SKILL.md").is_file():
            raise ValueError(f"Shipped skill missing SKILL.md: {shipped_skill_dir}")
        shipped_skills.append(shipped_skill_dir)
    return shipped_skills


def load_source_plugin_manifest() -> dict:
    if not SOURCE_PLUGIN_MANIFEST.is_file():
        raise ValueError(f"Missing source plugin manifest: {SOURCE_PLUGIN_MANIFEST}")
    payload = json.loads(SOURCE_PLUGIN_MANIFEST.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Plugin manifest must be a JSON object: {SOURCE_PLUGIN_MANIFEST}")
    return payload


def rebuild_claude_plugin_dist() -> Path:
    if GENERATED_PLUGIN_DIR.exists():
        shutil.rmtree(GENERATED_PLUGIN_DIR)
    GENERATED_PLUGIN_DIR.mkdir()

    manifest = load_source_plugin_manifest()
    GENERATED_PLUGIN_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copytree(SHIPPED_SKILLS_DIR, GENERATED_PLUGIN_DIR / "skills")
    return GENERATED_PLUGIN_DIR


def rebuild_generated_artifacts() -> tuple[list[Path], Path]:
    shipped_skills = rebuild_shipped_skills()
    plugin_dir = rebuild_claude_plugin_dist()
    return shipped_skills, plugin_dir


def main() -> None:
    shipped_skills, plugin_dir = rebuild_generated_artifacts()
    message = (
        f"Rebuilt {SHIPPED_SKILLS_DIR} and {plugin_dir} from {SOURCE_SKILLS_DIR} "
        f"with {len(shipped_skills)} shipped skill(s)."
    )
    typer.secho(message, fg=typer.colors.GREEN)


if __name__ == "__main__":
    main()
