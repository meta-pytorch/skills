# Contributing to pt_skills

## Adding a Skill

1. Create a directory under `skills/` with a lowercase, hyphenated name:
   ```
   skills/my-new-skill/SKILL.md
   ```

2. Write `SKILL.md` with YAML frontmatter:
   ```yaml
   ---
   name: my-new-skill
   description: Brief description of what this does and when to use it.
   ---

   # My New Skill

   ## When to use this skill

   - Scenario 1
   - Scenario 2

   ## Instructions

   Step-by-step guidance for the agent.
   ```

3. Optional: add supporting files in the same directory (references, scripts, templates).

## Naming Rules

- Lowercase letters, numbers, and hyphens only
- Max 64 characters
- Directory name must match the `name` field in frontmatter

## Description Tips

Include **what** the skill does and **when** to use it. Mention specific APIs, error messages, or workflows so agents can match the skill to user intent.

## Testing Locally

Clone and symlink into your Claude Code skills directory:

```bash
ln -s $(pwd)/skills/my-new-skill ~/.claude/skills/my-new-skill
```

Restart Claude Code and test with a relevant prompt.

## Pull Requests

- One skill per PR
- Include a test plan describing how you verified the skill works
