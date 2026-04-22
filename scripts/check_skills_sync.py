"""Pre-commit guard for generated artifact sync drift."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.sync_skills import (
    GENERATED_PLUGIN_DIR,
    SHIPPED_SKILLS_DIR,
    rebuild_generated_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATED_OUTPUT_DIRS = [SHIPPED_SKILLS_DIR, GENERATED_PLUGIN_DIR]


def generated_outputs_dirty() -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", *(str(path) for path in GENERATED_OUTPUT_DIRS)],
        cwd=ROOT,
        check=False,
    )
    return result.returncode != 0


def main() -> None:
    rebuild_generated_artifacts()
    if not generated_outputs_dirty():
        return

    print("Generated skill artifacts were out of date and have been regenerated.", file=sys.stderr)
    print("Review the changes, then run:", file=sys.stderr)
    print("  uv run sync-skills", file=sys.stderr)
    print("  git add skills claude-plugin-dist", file=sys.stderr)
    print("After that, re-run your commit.", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
