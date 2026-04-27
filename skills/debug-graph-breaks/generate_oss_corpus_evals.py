#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Generate a single eval.yaml from the oss-model-graph-break-corpus.

This script reads corpus.json from the oss-model-graph-break-corpus repository
and generates a single eval YAML with one test case per model that has graph
breaks. The generated YAML follows the eval_version: "1" format, providing
minimal hints about the actual root cause so the skill is evaluated on its
independent ability to detect, diagnose, and fix graph breaks.

Usage:
    # Clone the corpus first:
    git clone https://github.com/penguinwu/oss-model-graph-break-corpus.git /tmp/corpus

    # Generate eval.yaml (written to evals/eval.yaml by default):
    python3 generate_oss_corpus_evals.py /tmp/corpus/corpus/corpus.json

    # Or specify a custom output path:
    python3 generate_oss_corpus_evals.py /tmp/corpus/corpus/corpus.json --output /path/to/eval.yaml

    # Generate only for specific root causes:
    python3 generate_oss_corpus_evals.py /tmp/corpus/corpus/corpus.json --root-cause "copy.deepcopy"

    # Limit to N models per root cause (for a smaller eval suite):
    python3 generate_oss_corpus_evals.py /tmp/corpus/corpus/corpus.json --max-per-cause 3

    # Dry run (print what would be generated):
    python3 generate_oss_corpus_evals.py /tmp/corpus/corpus/corpus.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "evals", "eval.yaml")

SKILL_NAME = "debug-graph-breaks"
SKILL_PATH = "skills/debug-graph-breaks/SKILL.md"
MODELS = ["claude-opus-4-6"]

GRAPH_BREAK_DOCS_URL = "https://meta-pytorch.org/compile-graph-break-site/"

USER_MESSAGE_TEMPLATE = """\
/pt2:debug-graph-breaks
I want to compile the following model with torch.compile, but it's hitting graph breaks.
Please detect the graph breaks, diagnose them, and write a fixed version of the script
that compiles successfully with fullgraph=True or has as few graph breaks as possible.

```python
{code}
```

The graph break documentation website is at {docs_url}.
When you encounter a specific graph break (e.g., GB0179), fetch the corresponding page
(e.g., {docs_url}gb/gb0179.html) for detailed context, examples, and fix suggestions.

To find the transformers library source code, run:
  python3 -c "import transformers, os; print(os.path.dirname(transformers.__file__))"
You can then read source files from that directory to inspect the code causing graph breaks.

When fixing graph breaks that originate in the transformers library, create a subclass
of the model that overrides the problematic method(s) with compile-compatible
implementations.

Fix the graph breaks and save the fixed script as `fixed_model.py` in the current
working directory. The script must be self-contained and runnable with
`python3 fixed_model.py`.
"""

CRITICAL_REQUIREMENTS = [
    (
        "Agent runs the script with TORCH_LOGS='graph_breaks' or fullgraph=True"
        " to detect graph breaks"
    ),
    (
        "Agent identifies the source of at least one graph break"
        " in the user code or transformers library"
    ),
    (
        "Agent explains the root cause of the graph break"
        " and why Dynamo cannot trace it"
    ),
    (
        "Agent produces a fixed_model.py script that either compiles"
        " with fullgraph=True or has fewer graph breaks than the original"
    ),
]

QUALITY_FACTORS = [
    {
        "factor": (
            "Accuracy — correctly identifies the cause of the graph break"
            " and explains why Dynamo cannot trace it"
        ),
        "weight": 0.2,
    },
    {
        "factor": "Actionability — provides concrete fix strategies with working code examples",
        "weight": 0.2,
    },
    {
        "factor": (
            "Graph break reduction — the fixed script has fewer graph breaks"
            " or achieves fullgraph compilation"
        ),
        "weight": 0.5,
    },
    {
        "factor": (
            "Documentation use — references the graph break documentation website"
            " or GB type pages when explaining the break"
        ),
        "weight": 0.1,
    },
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    name: str
    description: str
    user_message: str
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# YAML serialization helpers
# ---------------------------------------------------------------------------


class _LiteralStr(str):
    pass


class _FoldedStr(str):
    pass


def _literal_representer(dumper: yaml.Dumper, data: str) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


def _folded_representer(dumper: yaml.Dumper, data: str) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=">")


class _EvalDumper(yaml.Dumper):
    """Custom YAML dumper that preserves key order and uses block scalars."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        return super().increase_indent(flow, False)


_EvalDumper.add_representer(_LiteralStr, _literal_representer)
_EvalDumper.add_representer(_FoldedStr, _folded_representer)


def _test_case_to_dict(tc: TestCase) -> dict[str, Any]:
    return {
        "name": tc.name,
        "description": tc.description,
        "user_message": _LiteralStr(tc.user_message),
        "tags": tc.tags,
        "deterministic_checks": [
            {"type": "skill_invoked", "skill": SKILL_NAME},
        ],
        "llm_judge": {
            "rubric": {
                "critical_requirements": CRITICAL_REQUIREMENTS,
                "quality_factors": QUALITY_FACTORS,
            }
        },
    }


def serialize_eval_suite(test_cases: list[TestCase]) -> str:
    """Serialize a list of TestCases to a single eval.yaml."""
    doc: dict[str, Any] = {
        "eval_version": "1",
        "skill_name": SKILL_NAME,
        "skill_path": SKILL_PATH,
        "models": MODELS,
        "test_cases": [_test_case_to_dict(tc) for tc in test_cases],
    }
    return yaml.dump(
        doc,
        Dumper=_EvalDumper,
        default_flow_style=False,
        sort_keys=False,
        width=200,
        allow_unicode=True,
    )


# ---------------------------------------------------------------------------
# Eval generation
# ---------------------------------------------------------------------------


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "-", name.lower()).strip("-")


def build_test_case(model: dict[str, Any], mode: str = "eval") -> TestCase:
    """Build a TestCase from a corpus model entry."""
    name = model["name"]
    source = model.get("source", "hf")
    mode_data = model.get(mode, {})
    root_cause = mode_data.get("root_cause", "unknown")

    safe_name = sanitize_name(name)
    safe_root_cause = sanitize_name(root_cause)
    source_label = "HuggingFace Diffusers" if source == "diffusers" else "HuggingFace"
    mode_call = "model.eval()" if mode == "eval" else "model.train()"

    code = (
        "import torch\n"
        "from transformers import AutoModel\n"
        "\n"
        f'model = AutoModel.from_pretrained("{name}", trust_remote_code=True)\n'
        f"{mode_call}\n"
        "\n"
        'compiled_model = torch.compile(model, backend="eager", fullgraph=True)'
    )

    return TestCase(
        name=f"{safe_name}-{safe_root_cause}",
        description=(
            f"Tests that the agent detects, diagnoses, and fixes graph breaks "
            f"in a {source_label} {name} compiled with torch.compile."
        ),
        user_message=USER_MESSAGE_TEMPLATE.format(
            code=code,
            docs_url=GRAPH_BREAK_DOCS_URL,
        ),
        tags=["smoke", "transformers", "fullgraph"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a single eval.yaml from oss-model-graph-break-corpus"
    )
    parser.add_argument(
        "corpus_json",
        help="Path to corpus.json from oss-model-graph-break-corpus",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output path for the generated eval.yaml (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--root-cause",
        help="Only generate evals for this root cause (e.g., 'copy.deepcopy')",
    )
    parser.add_argument(
        "--max-per-cause",
        type=int,
        default=0,
        help="Max test cases per root cause category (0 = unlimited)",
    )
    parser.add_argument(
        "--mode",
        default="eval",
        choices=["eval", "train"],
        help="Which mode to generate evals for (default: eval)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files",
    )
    args = parser.parse_args()

    with open(args.corpus_json) as f:
        corpus = json.load(f)

    models: list[dict[str, Any]] = corpus["models"]
    gb_models = [
        m
        for m in models
        if m.get("has_graph_break")
        and m.get(args.mode, {}).get("status") == "graph_break"
    ]

    if args.root_cause:
        gb_models = [
            m
            for m in gb_models
            if m.get(args.mode, {}).get("root_cause") == args.root_cause
        ]

    if args.max_per_cause > 0:
        cause_counts: dict[str, int] = defaultdict(int)
        filtered = []
        for m in gb_models:
            rc = m.get(args.mode, {}).get("root_cause", "unknown")
            if cause_counts[rc] < args.max_per_cause:
                filtered.append(m)
                cause_counts[rc] += 1
        gb_models = filtered

    root_cause_counts: Counter[str] = Counter()
    test_cases = []
    for model in sorted(gb_models, key=lambda m: m["name"]):
        rc = model.get(args.mode, {}).get("root_cause", "unknown")
        root_cause_counts[rc] += 1
        tc = build_test_case(model, args.mode)
        test_cases.append(tc)
        print(f"  {'Would include' if args.dry_run else 'Including'}: {tc.name} (root_cause={rc})")

    if not args.dry_run:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            f.write(serialize_eval_suite(test_cases))
        print(f"\nWrote {len(test_cases)} test cases to {args.output}")
    else:
        print(f"\nWould write {len(test_cases)} test cases to {args.output}")

    print("\nBy root cause:")
    for rc, count in root_cause_counts.most_common():
        print(f"  {rc}: {count}")


if __name__ == "__main__":
    main()
