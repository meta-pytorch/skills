# pt_skills

AI coding skills for PyTorch development. Skills are structured prompts that teach AI agents (Claude Code, Cursor, Copilot) how to perform PyTorch-specific tasks like debugging `torch.compile`, authoring operators, profiling, and triaging issues.

## Install

```bash
npx skills add meta-pytorch/pt_skills
```

## Structure

```
pt_skills/
├── .claude-plugin/
│   └── marketplace.json      # Skill registry metadata
├── skills/
│   └── my-skill/
│       ├── SKILL.md           # Required — skill definition
│       ├── references/        # Optional — detailed docs
│       ├── scripts/           # Optional — helper scripts
│       └── assets/            # Optional — templates, resources
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## SKILL.md Format

Each skill is a markdown file with YAML frontmatter following the [Agent Skills spec](https://agentskills.io/specification):

```yaml
---
name: my-skill                    # Required. Lowercase, hyphens, digits. Must match directory name.
description: >-                   # Required. What it does AND when to use it. Max 1024 chars.
  Debug torch.compile graph breaks
  and guard failures. Use when
  encountering recompilation issues.
license: BSD-3-Clause             # Optional.
metadata:                         # Optional.
  author: pytorch
  version: "1.0"
---

# My Skill

Instructions for the agent...
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

BSD-3-Clause
