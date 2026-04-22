# Skills

AI coding skills for PyTorch development. Skills are structured prompts that teach AI agents (Claude Code, Cursor, Copilot) how to perform PyTorch-specific tasks like debugging `torch.compile`, authoring operators, profiling, and triaging issues.

## Install

```bash
npx skills add meta-pytorch/skills
```

## Dev Setup

```bash
uv sync --group dev
uv run prek run -a
uv run sync-skills
```

The local pre-commit config checks that generated artifacts are up to date. If it
fails, run:

```bash
uv run sync-skills
git add skills claude-plugin-dist
```

## Create A Skill

```bash
uv run python scripts/new_skill.py my-new-skill
```

This creates a minimal `SKILL.md`, `evals/eval.yaml`, `evals/checkers.py`, and
`evals/fixtures/` scaffold under `skills-with-evals/`. After editing a skill,
regenerate the shipped tree with:

```bash
uv run sync-skills
```

New skills normally do not require a manual `.claude-plugin` edit. The repo
ships two generated install surfaces:

- `./skills` for `npx skills add meta-pytorch/skills`
- `./claude-plugin-dist` for Claude marketplace installs

`scripts/validate_plugin.py` checks that every source skill stays represented in
both generated outputs.

## Structure

```text
repo-root/
├── .claude-plugin/
│   ├── marketplace.json       # Root marketplace; points Claude installs at claude-plugin-dist/
│   └── plugin.json            # Root plugin manifest for local --plugin-dir flows
├── skills-with-evals/         # Source skills, including eval authoring files
│   └── my-skill/
│       ├── SKILL.md           # Required — skill definition
│       ├── evals/
│       │   ├── eval.yaml      # Required — benchmark contract
│       │   ├── fixtures/      # Optional — task inputs, repros, traces
│       │   └── checkers.py    # Optional — custom outcome checks
│       ├── references/        # Optional — detailed docs
│       ├── scripts/           # Optional — helper scripts
│       └── assets/            # Optional — templates, resources
├── skills/                    # Generated shipped tree discovered by npx skills
│   ├── README.md
│   └── my-skill/
│       ├── SKILL.md
│       ├── references/
│       ├── scripts/
│       └── assets/
├── claude-plugin-dist/        # Generated Claude marketplace payload
│   ├── marketplace.json
│   └── skills/
│       └── my-skill/
│           ├── SKILL.md
│           ├── references/
│           ├── scripts/
│           └── assets/
├── skill_eval/                # Eval harness, scoring, reporting
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

`skills/` and `claude-plugin-dist/` are generated from `skills-with-evals/` and
strip every directory named `evals/`. External installs like
`npx skills add meta-pytorch/skills` consume the shipped top-level `skills/`
tree, while Claude marketplace installs consume the generated
`claude-plugin-dist/` payload. Skill authors continue to work in
`skills-with-evals/`. If a source skill is deleted, the next sync rebuild
removes both generated mirrors automatically.

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

## Quality Bar

Every skill is expected to ship with evaluation coverage. Today that means:

- `evals/eval.yaml` lives next to `SKILL.md`
- The eval defines a benchmark contract, not just a prompt example
- Reviewers approve the eval's credibility; the harness computes the score
- The harness emits a score per model and preserves raw run evidence for later comparisons

Each test case can combine:

- Deterministic outcome checks like file contents, JSON values, or exact output
- Optional Python checkers for PyTorch-specific verification
- Optional LLM judgment for semantic quality that is hard to check deterministically

## Eval Harness

### Quickstart

```bash
# 1. Validate the eval contract (syntax, required fields)
uv run python -m skill_eval validate skills-with-evals/my-skill

# 2. Run a single eval
uv run python -m skill_eval run skills-with-evals/my-skill

# 3. Run an A/B eval (baseline without skill vs treatment with skill)
uv run python -m skill_eval run-ab skills-with-evals/my-skill

# 4. Run all evals in the repo
uv run python -m skill_eval run-all
```

### Verbose and Debug Output

Use `--verbose` to see per-test details inline:

```bash
uv run python -m skill_eval run-ab skills-with-evals/my-skill --verbose
```

Verbose output includes:
- The agent's response text (first 500 chars)
- Which tools and skills were invoked (e.g. `Skill(my-skill)`)
- Each deterministic check result with PASS/FAIL and details
- Checker and LLM judge results when present
- Final score per test case

Use `--debug-dir` to save full artifacts to disk for post-mortem inspection:

```bash
uv run python -m skill_eval run-ab skills-with-evals/my-skill --debug-dir ./debug
```

This writes per-test-case directories under `debug/<variant>/<model>/<test_case>/` containing:
- `claude_command.txt` — the exact CLI command that was run
- `claude_stdout.txt` — full stream-json output from the Claude CLI
- `claude_stderr.txt` — stderr from the Claude CLI
- `runner_metadata.json` — exit code, duration, variant, model

Use `--output-json` to write the machine-readable result bundle:

```bash
uv run python -m skill_eval run-ab skills-with-evals/my-skill --output-json results.json
```

### Writing eval.yaml

Each skill needs an `evals/eval.yaml` next to `SKILL.md`. Minimal example:

```yaml
eval_version: "1"
skill_name: "my-skill"
skill_path: "skills-with-evals/my-skill/SKILL.md"

models:
  - "claude-opus-4-6"

test_cases:
  - name: "basic_usage"
    description: "The skill produces the expected output."
    user_message: "Do the thing my skill does"
    deterministic_checks:
      - type: "output_contains_ci"
        value: "expected phrase"
      - type: "skill_invoked"
        skill: "my-skill"
```

**Key fields:**

| Field | Required | Description |
|---|---|---|
| `user_message` | yes | The prompt sent to the agent. Use a natural question, not a slash command — the baseline arm has no skill loaded. |
| `deterministic_checks` | at least one of checks/checker/judge | Built-in pass/fail checks (see table below). |
| `checker` | optional | Path to a Python checker module for custom verification. |
| `llm_judge` | optional | LLM judge rubric for semantic quality scoring. |
| `fixture` | optional | Directory copied into the workdir before the agent runs. |
| `setup` | optional | Shell commands to run in the workdir before the agent. |
| `tags` | optional | Labels for filtering test cases. |

### Deterministic Check Types

| Type | Params | Description |
|---|---|---|
| `output_equals` | `value` | Agent output exactly equals value (whitespace-trimmed). |
| `output_contains` | `value` | Agent output contains value (case-sensitive). |
| `output_contains_ci` | `value` | Agent output contains value (case-insensitive). |
| `file_exists` | `path` | File exists in the workdir after the agent runs. |
| `file_contains` | `path`, `value` | File in workdir contains the value string. |
| `file_json_equals` | `path`, `key`, `value` | JSON file's dot-path key equals value. |
| `command_exit_code` | `command`, `exit_code` (default 0) | Shell command exits with expected code. |
| `skill_invoked` | `skill` | The agent invoked the named skill via the Skill tool. |

### How A/B Evals Work

In `run-ab` mode, the harness runs every test case twice:

- **baseline** (`SKILL_EVAL_ENABLE_SKILL=0`): no `--plugin-dir`, skill is not available
- **treatment** (`SKILL_EVAL_ENABLE_SKILL=1`): `--plugin-dir <repo-root>`, skill is loaded

A good eval should:
1. Use a natural `user_message` that the agent can attempt without the skill
2. Have checks that only pass when the skill adds value (e.g. specific output, correct file state)
3. Expect the baseline to fail and the treatment to pass

### Requirements

The built-in Claude runtime requires:
- `claude` CLI on `PATH`
- Working Claude auth in the environment
- Model names in `eval.yaml` that are valid Claude model IDs

### Custom Runtimes

Override the runtime or judge with `--runtime-command` and `--judge-command`:

```bash
uv run python -m skill_eval run-ab skills-with-evals/my-skill \
  --runtime-command "uv run python -m your_runtime" \
  --judge-command "uv run python -m your_judge"
```

Per-arm overrides for A/B mode: `--baseline-runtime-command`, `--treatment-runtime-command`.

The runtime receives these environment variables:

| Variable | Description |
|---|---|
| `SKILL_EVAL_PROMPT` | The `user_message` from the test case |
| `SKILL_EVAL_MODEL` | Model ID to use |
| `SKILL_EVAL_SKILL_NAME` | Skill name from `eval.yaml` |
| `SKILL_EVAL_SKILL_PATH` | Absolute path to the skill's `SKILL.md` |
| `SKILL_EVAL_SKILL_PATH_REPO_REL` | Repo-relative path to `SKILL.md` |
| `SKILL_EVAL_EVAL_DIR` | Directory containing `eval.yaml` |
| `SKILL_EVAL_WORKDIR` | Temporary working directory for the agent |
| `SKILL_EVAL_TEST_CASE` | Test case name |
| `SKILL_EVAL_VARIANT` | `baseline`, `treatment`, or `single` |
| `SKILL_EVAL_ENABLE_SKILL` | `0` or `1` |
| `SKILL_EVAL_REPO_ROOT` | Repository root path |
| `SKILL_EVAL_VERBOSE` | `1` if `--verbose` was passed |

The runtime should print a JSON envelope to stdout:

```json
{"result_text": "agent response", "tools_invoked": [{"tool": "Skill", "id": "...", "skill": "my-skill"}]}
```

Plain text stdout is also accepted (tool invocation tracking won't be available).

## License

BSD-3-Clause
