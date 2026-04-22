"""Pre-commit guard for shipped skills sync drift."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.sync_skills import rebuild_shipped_skills

ROOT = Path(__file__).resolve().parents[1]
SHIPPED_SKILLS_DIR = ROOT / "skills"


def shipped_skills_dirty() -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", str(SHIPPED_SKILLS_DIR)],
        cwd=ROOT,
        check=False,
    )
    return result.returncode != 0


def main() -> None:
    rebuild_shipped_skills()
    if not shipped_skills_dirty():
        return

    print("skills/ was out of date and has been regenerated.", file=sys.stderr)
    print("Review the changes, then run:", file=sys.stderr)
    print("  uv run sync-skills", file=sys.stderr)
    print("  git add skills", file=sys.stderr)
    print("After that, re-run your commit.", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
