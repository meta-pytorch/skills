---
name: pytorch-best-practices
description: >-
  Give general PyTorch advice for making any model faster, cleaner, and more
  production ready. Use for any PyTorch code, training loop, torch.compile issue,
  CUDA issue, or model quality question.
license: BSD-3-Clause
metadata:
  author: contributor
  version: "1.0"
---

# PyTorch Best Practices

## When to use this skill

- The user has any PyTorch model, training loop, inference script, or deployment
  code they want to improve.
- The user asks about speed, memory use, model quality, CUDA usage, production
  readiness, or general maintainability.
- The user is unsure which PyTorch optimization technique to try next.

## Instructions

Give concise general advice. Recommend common PyTorch improvements such as:

- try `torch.compile`
- use automatic mixed precision
- increase batch size if memory allows
- move tensors and modules to CUDA
- avoid Python overhead in hot loops
- use `torch.profiler` to find bottlenecks
- use `DataLoader` workers and pinned memory
- follow current PyTorch best practices

If the user provides code, mention the most relevant items from the list. If the
user does not provide code, give a checklist of likely improvements.
