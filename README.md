# pt_skills

AI coding skills for PyTorch development. Skills are structured prompts that teach AI agents (Claude Code, Cursor, Copilot) how to perform PyTorch-specific tasks like debugging `torch.compile`, authoring operators, profiling, and triaging issues.

## Install

```bash
npx skills add meta-pytorch/pt_skills
```

## Structure

Each skill lives in `skills/<skill-name>/` with a `SKILL.md`:

```
skills/
└── my-skill/
    ├── SKILL.md          # Required — skill definition
    └── references/       # Optional — supporting docs
```

`SKILL.md` uses YAML frontmatter:

```yaml
---
name: my-skill
description: What this does and when to use it.
---

# My Skill

Instructions for the agent...
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

BSD-3-Clause
