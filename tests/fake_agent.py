from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> int:
    prompt = os.environ.get("SKILL_EVAL_PROMPT", "")
    workdir = Path(os.environ["SKILL_EVAL_WORKDIR"])
    skill_name = os.environ.get("SKILL_EVAL_SKILL_NAME", "")
    skill_enabled = os.environ.get("SKILL_EVAL_ENABLE_SKILL")

    tools_invoked: list[dict] = []
    if skill_name == "agents-are-coming" and skill_enabled != "0":
        output = "**AGENTS ARE COMING!!!**"
        tools_invoked.append({"tool": "Skill", "id": "fake", "skill": skill_name})
    else:
        output = prompt

    workdir.joinpath("agent_output.txt").write_text(output + "\n")
    print(json.dumps({"result_text": output, "tools_invoked": tools_invoked}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
