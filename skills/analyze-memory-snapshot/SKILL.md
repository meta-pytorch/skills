---
name: analyze-memory-snapshot
description: >-
  Analyze PyTorch CUDA memory snapshots from torch.cuda.memory._dump_snapshot(),
  torch.cuda.memory._snapshot(), or memory_viz artifacts. Use when the user
  wants to understand GPU memory usage, top allocation sites, deallocation
  timing, OOMs, allocator fragmentation, inactive cached memory, memory leaks,
  retained tensors, or Python reference cycles in PyTorch workloads.
license: BSD-3-Clause
metadata:
  author: pytorch
  version: "1.0"
---

# Analyze Memory Snapshot

Analyze PyTorch CUDA memory snapshots by inspecting allocator state and event
history. Always prefer evidence from the snapshot over generic memory advice.

## Workflow

1. Get the snapshot path if the user did not provide one. Accept `.pickle`,
   `.pkl`, `.json`, or a directory containing rank-local snapshots. If the user
   only has a screenshot from memory_viz, ask for the underlying snapshot file.
2. Load the snapshot locally. PyTorch `_dump_snapshot()` files are pickles, so
   only unpickle files the user provided or that already live in the workspace.
   JSON snapshots can be loaded directly.
3. If available, run this skill's helper for a first-pass summary:

   ```bash
   python scripts/analyze_snapshot.py /path/to/snapshot.pickle
   ```

   If there are multiple ranks, run it per rank and compare peaks, active bytes,
   inactive bytes, OOM events, and top stack traces.
4. Read the raw snapshot when the helper output is insufficient. Inspect:
   - `segments`: allocator segments returned by `cudaMalloc`
   - `blocks`: live, awaiting-free, and inactive pieces inside each segment
   - `device_traces`: chronological allocation events when memory history was
     enabled
   - `frames`: stack traces for allocation sites
5. Report the memory picture in terms of allocator semantics:
   - **Reserved memory**: sum of segment `total_size`
   - **Live tensor memory**: blocks in `active_allocated`
   - **Delayed frees**: blocks in `active_awaiting_free`, often stream-related
   - **Inactive cached memory**: `inactive` blocks that PyTorch can reuse
   - **Requested vs rounded size**: `requested_size` versus block `size`

## What To Diagnose

**Where memory is going**

Group live `active_allocated` blocks by the most useful user frame. Report the
largest stack traces first, with bytes, percent of live memory, block count, and
representative file/function/line. Separate parameters, optimizer state,
activations, temporary buffers, communication buffers, and user containers when
the stack names make that clear.

**When memory is deallocated**

Use `device_traces` to distinguish `free_requested` from `free_completed`.
Pending or delayed frees can mean tensors are still needed by another stream or
that work has not reached the free point yet. In multi-step traces, compare
repeated allocation sites with their matching free events: allocation sites that
appear every step without corresponding frees are stronger retention evidence
than one-time model or optimizer allocations. If traces include OOM events,
report the failed allocation size, `device_free`, OOM stack frame, and which live
allocation sites were holding memory immediately before the OOM.

**Leak and retention patterns**

Look for active bytes that grow across steps, repeated active allocations from
the same stack that are not freed, or a sawtooth where many tensors accumulate
and are released together. Common causes:

- appending tensors, losses, or outputs to Python lists without detaching or
  moving them to CPU
- closures, callbacks, hooks, or caches retaining tensors
- `retain_graph=True` or backward graphs kept alive accidentally
- optimizer state allocated after the first step and then expectedly retained
- Python reference cycles holding CUDA tensors until GC runs

When the pattern suggests reference cycles, recommend
`from torch.utils.viz._cycles import warn_tensor_cycles` and call
`warn_tensor_cycles()` during initialization to find object cycles that keep
CUDA tensors alive.

**Fragmentation and allocator behavior**

Do not equate `nvidia-smi` usage with live tensors. PyTorch uses a caching CUDA
allocator, so reserved memory can be much larger than allocated tensor memory.
Large inactive blocks are reusable by PyTorch; `empty_cache()` may return fully
inactive segments to CUDA, but it does not free memory still occupied by live
tensors. If reserved memory is high while live memory is low, discuss caching,
fragmentation, allocation size churn, and whether `PYTORCH_ALLOC_CONF` tuning is
worth testing.

**Non-PyTorch memory**

The snapshot only sees CUDA memory managed by the PyTorch allocator. NCCL,
custom CUDA extensions, cuDNN/cuBLAS workspaces, or direct CUDA allocations may
not appear. If device memory outside the snapshot is suspected, compare
`torch.cuda.device_memory_used(device)` or `nvidia-smi` with the snapshot's
reserved bytes.

## If The User Needs To Capture A Snapshot

Give the minimal capture pattern:

```python
import torch

torch.cuda.memory._record_memory_history(max_entries=100000)
try:
    run_your_code()
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
finally:
    torch.cuda.memory._record_memory_history(enabled=None)
```

Use a higher `max_entries` for long jobs, but warn that trace history has memory
overhead. For OOM debugging, dump the snapshot in the exception handler too.

## Output Format

Start with the main finding in one sentence. Then provide:

1. Totals: reserved, live allocated, active-awaiting-free, inactive cached, and
   peak if trace history allows it.
2. Top allocation sites by live memory, with stack frames and percentages.
3. Deallocation timeline: frees completed, pending frees, delayed frees, and OOM
   events.
4. Diagnosis: likely leak, expected retention, fragmentation/caching, or
   non-PyTorch memory gap.
5. Concrete next steps: code locations to inspect, collection or detach changes
   to try, capture changes, and allocator checks.

Be explicit about uncertainty. If the snapshot lacks history, say which
deallocation and peak questions cannot be answered from that artifact.
