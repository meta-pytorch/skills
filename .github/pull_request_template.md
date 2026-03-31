## What

<!-- Brief description of the skill or change -->

## Skill Checklist

<!-- If this PR adds/modifies a skill, complete this checklist -->

- [ ] `uv run python scripts/new_skill.py <name>` was used or the same structure was created intentionally
- [ ] `SKILL.md` has YAML frontmatter with `name` and `description`
- [ ] Directory name matches `name` field in frontmatter
- [ ] `evals/eval.yaml` exists for each added or modified skill
- [ ] The eval measures a real task outcome, not only formatting or style
- [ ] Deterministic checks are used where possible
- [ ] `uv run python scripts/validate_plugin.py` passes
- [ ] Manual verification was run against the authored benchmark contract below

## Test Plan

<!-- How did you verify the skill works? Include specific prompts you tested. -->

## Eval Coverage

<!-- List the eval cases you added or updated, plus which ones you exercised manually. -->

## Why This Eval Is Credible

<!-- Explain why this benchmark reflects the real value of the skill and why future score changes should matter. -->
