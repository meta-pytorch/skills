# Contributing to skills

## Adding a Skill

1. Create a directory under `skills/` with a lowercase, hyphenated name:
   ```
   skills/my-new-skill/SKILL.md
   ```

2. Write `SKILL.md` with YAML frontmatter:
   ```yaml
   ---
   name: my-new-skill
   description: >-
     Brief description of what this does and when to use it.
     Include trigger keywords so agents can match intent.
   license: BSD-3-Clause
   metadata:
     author: your-name
     version: "1.0"
   ---

   # My New Skill

   ## When to use this skill

   - Scenario 1
   - Scenario 2

   ## Instructions

   Step-by-step guidance for the agent.
   ```

3. Optional: add supporting files in the same directory:
   - `references/` — detailed docs, API references
   - `scripts/` — helper scripts the agent can run
   - `assets/` — templates, config files

4. Add the skill path to `.claude-plugin/marketplace.json` in the `skills` array.

## Naming Rules

- Lowercase letters, numbers, and hyphens only
- Max 64 characters
- Cannot start or end with a hyphen, no consecutive hyphens
- Directory name must match the `name` field in frontmatter

## Description Tips

Include **what** the skill does and **when** to use it. Mention specific PyTorch APIs, error messages, or workflows so agents can match the skill to user intent. Max 1024 characters.

## Size Guidelines

Keep `SKILL.md` under 500 lines (~5000 tokens). Move detailed references to separate files in `references/` for progressive disclosure — agents load these on demand.

## Testing Locally

```bash
npx skills add ./path/to/skills
```

Or symlink directly:

```bash
ln -s $(pwd)/skills/my-new-skill ~/.claude/skills/my-new-skill
```

Restart your agent and test with a relevant prompt.

## Pull Requests

- One skill per PR
- Include a test plan describing how you verified the skill works
