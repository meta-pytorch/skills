# Contributing to skills

## Definition of done

A new or modified skill is ready to merge when all of the following are true:

- `skills-with-evals/<skill-name>/SKILL.md` exists and has valid frontmatter
- `skills-with-evals/<skill-name>/evals/eval.yaml` exists and validates
- The eval is outcome-oriented and believable enough that a future score drop would matter
- The PR includes manual verification notes against the authored benchmark contract
- The skill stays focused, specific, and small enough for agents to load quickly

The goal is simple: no skill without a way to evaluate whether it produces the
right outcomes.

## Adding a skill

1. Start from the scaffold command:

   ```bash
   uv run python scripts/new_skill.py my-new-skill
   ```

   This creates the minimum reviewable structure:

   ```text
   skills-with-evals/my-new-skill/
   ├── SKILL.md
   └── evals/
       ├── eval.yaml
       ├── fixtures/
       └── checkers.py
   ```

2. Fill in `SKILL.md` with YAML frontmatter:

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

3. Replace the scaffolded `evals/eval.yaml` with a real benchmark contract.

4. Optional: add supporting files in the same directory:
   - `references/` for detailed docs and API references
   - `scripts/` for helper scripts the agent can run
   - `assets/` for templates and reusable resources

You usually do not need to edit `.claude-plugin/plugin.json` or
`.claude-plugin/marketplace.json` for each new skill because this repo ships
generated install surfaces automatically. After changing a source skill,
regenerate them with:

```bash
uv run sync-skills
```

The sync step mirrors `skills-with-evals/<skill-name>/` into:

- `skills/<skill-name>/` for `npx skills`
- `claude-plugin-dist/skills/<skill-name>/` for Claude marketplace installs

Both generated outputs strip any directory named `evals/`. If a source skill is
deleted, the next sync rebuild removes both generated copies. The validator will
fail if a source skill is missing from generated output or if an `evals/`
directory appears in `skills/` or `claude-plugin-dist/`.

## Evaluation workflow

Contributors are expected to define how the skill should be judged before
opening a PR.

### What goes in `eval.yaml`

Each eval file should describe:

- The skill under test
- Which models to run
- The user task or scenario
- Any fixture or setup required to reproduce the task
- Deterministic checks for observable outcomes
- Optional custom checker logic
- Optional LLM-judge rubric for semantic quality

Use this shape:

```yaml
eval_version: "1"
skill_name: "my-new-skill"
skill_path: "skills-with-evals/my-new-skill/SKILL.md"
models:
  - "claude-sonnet"

test_cases:
  - name: "happy_path_basic"
    description: "Skill handles the primary use case"
    user_message: "The exact prompt the user would send"
    fixture: "fixtures/happy_path_basic"
    setup:
      - "python3 make_repro.py"
    deterministic_checks:
      - type: "file_json_equals"
        path: "result.json"
        key: "answer"
        value: 7
    checker: "checkers:check_answer"
    llm_judge:
      rubric:
        critical_requirements:
          - "Must explain why the issue occurs"
        quality_factors:
          - factor: "Explanation is actionable"
            weight: 1.0
```

### Benchmark design guidance

The benchmark contract should help a reviewer answer: "if this score regresses
later, will we care?"

Strong evals usually include a mix of:

- happy-path tasks
- edge or missing-context tasks
- adversarial or misleading tasks
- at least one task with a real outcome, not just formatting
- Deterministic checks are preferred whenever possible
- LLM judgment should be reserved for what cannot be checked mechanically

There is intentionally no rigid required taxonomy yet. PyTorch skill types are
still emerging, so authors should optimize for realistic tasks and stable
verification rather than forcing their evals into a template too early.

### Fixtures

Fixtures are the concrete task substrate. Good fixtures are:

- small PyTorch programs
- failing tests
- repro scripts
- logs, traces, or stack dumps
- tiny workspace snapshots
- operator stubs or profiling artifacts

Keep fixtures as small and stable as possible while still preserving the real
problem the skill is supposed to help with.

### Review rubric

Reviewers are evaluating the benchmark contract, not hand-grading the harness
score. A good benchmark:

- reflects the real job the skill claims to improve
- has observable success criteria
- is hard to game with generic boilerplate
- uses LLM judgment only where deterministic checks are insufficient
- would produce a meaningful signal if rerun later on a different model

## Local validation and runs

Sync the repo environment first:

```bash
uv sync --group dev
uv run prek run -a
uv run sync-skills
```

The local pre-commit config also checks that generated artifacts match the
source skills. If it regenerates files and fails, run:

```bash
uv run sync-skills
git add skills claude-plugin-dist
```

Then retry your commit.

Scaffold a new skill:

```bash
uv run python scripts/new_skill.py my-new-skill
```

Validate the contract:

```bash
uv run python scripts/validate_evals.py skills-with-evals/my-new-skill
```

Run the benchmark:

```bash
uv run python -m skill_eval run skills-with-evals/my-new-skill
```

Run the benchmark as A/B baseline vs treatment:

```bash
uv run python -m skill_eval run-ab skills-with-evals/my-new-skill
```

By default, both commands use the built-in Claude Code CLI runtime and the
built-in Claude judge. Make sure:

- `claude` is on `PATH`
- Claude Code auth is working in your shell environment
- the `models:` values in `eval.yaml` are valid Claude model IDs

If you want a different runtime or judge, override them explicitly:

```bash
uv run python -m skill_eval run skills-with-evals/my-new-skill \
  --runtime-command "uv run python -m your_agent_runtime" \
  --judge-command "uv run python -m your_judge_runtime"
```

The harness computes scores per model from the authored eval. Reviewers do not
need to audit the numeric score itself as part of normal PR review.

When run in A/B mode, the harness sets:

- `SKILL_EVAL_ENABLE_SKILL=0` for the baseline arm
- `SKILL_EVAL_ENABLE_SKILL=1` for the treatment arm

Runtime adapters can use that to include or exclude the skill while keeping the
benchmark contract unchanged.

## Manual verification before opening a PR

Before opening the PR:

1. Validate `evals/eval.yaml`.
2. Run the benchmark locally for at least one target model.
3. Exercise the main tasks the eval is claiming to measure.
4. Record what you tested in the PR, including gaps or known limitations.

### Local install

For `npx skills` testing, sync the shipped tree first and install the repo root.
Because top-level `skills/` is the generated shipped tree, repo-root installs use
the right artifact without pulling in eval files:

```bash
uv run sync-skills
npx skills add ./
```

Or symlink the shipped skill directly:

```bash
uv run sync-skills
ln -s $(pwd)/skills/my-new-skill ~/.claude/skills/my-new-skill
```

For Claude marketplace testing, `.claude-plugin/marketplace.json` points at the
generated `claude-plugin-dist/` payload, so marketplace installs avoid copying
the whole repo into the installed plugin payload.

If you want to inspect the exact shipped files, read `skills/README.md` before
editing anything there.

Restart your agent and test with prompts that map to your eval cases.

## Naming rules

- Lowercase letters, numbers, and hyphens only
- Max 64 characters
- Cannot start or end with a hyphen
- No consecutive hyphens
- Directory name must match the `name` field in frontmatter

## Description tips

Include both what the skill does and when to use it. Mention specific PyTorch
APIs, error messages, workflows, or failure modes so agents can match the skill
to user intent. Keep descriptions under 1024 characters.

## Size guidelines

Keep `SKILL.md` under 500 lines when possible. Move detailed references into
`references/` for progressive disclosure so agents load the core workflow first
and pull deeper material only when needed.

## Pull requests

- Prefer one skill per PR
- Include a test plan with the exact prompts you tried
- Call out which eval cases were exercised manually
- Explain why the eval is a believable measure of the skill's value
- Note any behavior you expect to improve in follow-up changes
