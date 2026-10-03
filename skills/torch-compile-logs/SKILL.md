---
name: torch-compile-logs
description: Collect logging information from torch.compile.  This is the first step in debugging any compile issues.  Also used to interpret user-provided logs
---

# Collect torch.compile logs

Collecting logs from torch.compile is the first step in debugging any issues.  There are two ways to capture this information - `tlparse` (more comprehensive), or the TORCH_LOGS environment variable (if tlparse is not installed).  If the user provides logs rather than a script, the below information is still useful for interpreting them.

## tlparse

`tlparse` can be installed with `pip install tlparse` or `cargo install tlparse`. Capture a structured trace with `TORCH_TRACE` into a fresh directory, then parse it:
```bash
TRACE_DIR=$(mktemp -d /tmp/torch_trace.XXXXXX)
TORCH_TRACE=$TRACE_DIR python repro.py
ls $TRACE_DIR    # one dedicated_log_*.log per process/rank
tlparse $TRACE_DIR/<one dedicated_log file> -o $TRACE_DIR/tlparse_out --overwrite --no-browser
```

- `tlparse` takes exactly one log file. Don't pass a glob: if it matches more than one file, `tlparse` errors out.
- `tlparse` won't write into an existing output directory. Use a fresh `-o` directory or pass `--overwrite`.
- `--no-browser` stops `tlparse` from trying to open the report.
- If there are several logs (multiple ranks or subprocesses), parse each one into its own output directory. Run `tlparse --help` first: newer versions also accept the trace directory with `--latest` (most recent log) or `--all-ranks-html` (combined multi-rank report). Use those only if the installed version lists them.

Note that `TORCH_TRACE` output can contain model source code.  Treat it as sensitive and warn the user before sharing or uploading.

Throughout the report compilation attempts are identified by a compile id like `[0/0]` (frame 0, first compile) or `[1/0]` (frame 1, first compile — often a resume after graph break).  Compile ids may additionally have a `_1` suffix, indicating a restart.

Key files under the output dir:
| File                                                      | Contents                                                                          |
|-----------------------------------------------------------|-----------------------------------------------------------------------------------|
| `index.html`                                              | entry point: stack trie, compile list, timing, status — read first                |
| `failures_and_restarts.html`                              | graph breaks and Dynamo restart events + reasons                                  |
| `chromium_events.json`                                    | Perfetto trace of the compile timeline (open in `chrome://tracing`)               |
| `-_<frame>_<compile>_.../`                                | per-graph frame folders                                                           |
| `.../recompile_reasons_*.json`                            | why a frame recompiled (the failing guard)                                        |
| `.../compilation_metrics*.html`                           | per-frame metrics + dynamic-shape specialization section                          |
| `.../dynamo_output_graph_*.txt`                           | FX graph Dynamo captured                                                          |
| `.../aot_forward_graph_*.txt`, `aot_backward_graph_*.txt` | AOTAutograd graphs                                                                |
| `.../output_code_*.py`                                    | Inductor-generated Triton/C++ code                                                |
| `.../put_local_code_state_*.txt`                          | dynamic-shape state (which dims are dynamic).  Only generated when a job id is set (`TORCH_COMPILE_JOB_ID=<name>` or `torch.compiler.config.job_id`) |

## TORCH_LOGS (targeted logging without a full trace)

Run the script with the `TORCH_LOGS` env variable set to collect targeted logging information.  Below are some common options, or use `TORCH_LOGS=+help` to see additional options.

```bash
TORCH_LOGS="recompiles,graph_breaks"        # why it recompiles / breaks
TORCH_LOGS="+dynamic"                        # symbolic-shape guards & specialization
TORCH_LOGS="+torch._inductor.codecache"      # FxGraph / code cache hits vs misses
TORCH_LOGS="guards,recompiles_verbose"       # full guard text
TORCH_LOGS="+inductor,output_code"           # Inductor codegen details
TORCH_LOGS="+cudagraphs"                      # CUDA graph tree operations
```
