# CLAUDE.md

AI coding skills for PyTorch development. Each skill is a structured prompt
(`SKILL.md`) that teaches agents a PyTorch-specific task, and every skill must
ship with an eval (`evals/eval.yaml`) that measures whether it works. The whole
`skills/` tree is published as a plugin via `npx skills add meta-pytorch/skills`.

## Commands

```bash
uv sync --group dev                                  # set up dev environment
uv run prek run -a                                   # lint / pre-commit checks
uv run python -m unittest discover -s tests          # unit tests
uv run validate-plugin                               # validate plugin manifests
uv run python scripts/new_skill.py <skill-name>      # scaffold a new skill
uv run python scripts/validate_evals.py skills/<skill-name>   # validate an eval contract
uv run python -m skill_eval run skills/<skill-name>          # run one eval
uv run python -m skill_eval run-ab skills/<skill-name>       # A/B eval (baseline vs skill)
```

CI runs `prek`, `validate-plugin`, unit tests, and (on PRs) `validate-skill-pr`.
Run `prek` and the unit tests before pushing.

## Layout

- `skills/<name>/SKILL.md` — required skill definition with YAML frontmatter
- `skills/<name>/evals/eval.yaml` — required benchmark contract
- `skills/<name>/evals/fixtures/`, `evals/checkers.py` — optional task inputs and custom checks
- `skills/<name>/references/`, `scripts/`, `assets/` — optional supporting material
- `skill_eval/` — the eval harness (runner, checks, LLM judge, Claude runtime)
- `scripts/` — scaffolding and validation scripts
- `tests/` — unit tests for the harness and scripts
- `.claude-plugin/` — plugin/marketplace manifests; rarely needs editing, since
  the plugin ships the whole `skills/` tree and `validate-plugin` enforces coverage

## Rules for skills

- **No skill without an eval.** A new or changed skill is not done until
  `evals/eval.yaml` exists, validates, and describes an outcome-oriented
  benchmark — not just a prompt example. See CONTRIBUTING.md for the review rubric.
- Skill names: lowercase letters, digits, hyphens; max 64 chars; must match the
  directory name and the `name:` field in frontmatter.
- Frontmatter `description` must say what the skill does **and** when to use it,
  with trigger keywords (APIs, error messages, workflows); max 1024 chars.
- Keep `SKILL.md` under ~500 lines; move deep detail into `references/` so
  agents load the core workflow first.
- Eval `user_message` should be a natural request, not a slash command — the
  A/B baseline arm runs without the skill loaded, and checks should only pass
  when the skill adds value.
- Prefer deterministic checks (`file_exists`, `output_contains_ci`,
  `command_exit_code`, `skill_invoked`, …); reserve LLM judges for what cannot
  be checked mechanically. Check types are documented in README.md.
- Keep fixtures small and stable while preserving the real problem.

## Pull requests

- One skill per PR.
- Include a test plan: the exact prompts tried, which eval cases were exercised
  manually, and why the eval is a believable measure of the skill's value.
- Follow `.github/pull_request_template.md`.
