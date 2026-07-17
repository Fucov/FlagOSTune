# Reliable SGLang Nsight Server-Steps Design

## Scope

This increment completes the independent SGLang Nsight Systems workflow for
single-node tensor-parallel inference. It supersedes the earlier decision to
defer `server-steps`, while preserving the existing `full-offline` interface
and Torch Profiler isolation.

Development happens directly on the current `main` branch. Local validation
uses fixture CSV/SQLite data, fake HTTP/process endpoints, shell dry runs, and
unit tests. Real SGLang, CUDA, multi-GPU, and Nsight Systems 2025.3.1 behavior
is verified on the server at `/data/yangkw/src/FlagOSTune` after the local
implementation is ready.

The current modular parser is the implementation baseline. It must not be
replaced by the older, simple public parser. Before server execution, the
server copy of `scripts/tools/parse_nsys.py` is compared with the working copy
and `origin/main` by commit and SHA256. Any server-only parser enhancement is
reviewed and incorporated rather than overwritten.

## Root Cause

The current `full-offline` command runs
`sglang.bench_offline_throughput` through `cuda_profiler_launcher.py` and calls
`cudaProfilerStart()` before the benchmark module starts. The capture therefore
includes model loading, NCCL initialization, allocator warmup, possible
DeepGEMM JIT, and early scheduler work. It cannot reliably select steady-state
worker decode steps, and must not be described as decode-only.

The parser has four related compatibility gaps:

1. Its default report set requests `:base` variants that Nsight Systems 2025.3.1
   does not necessarily expose.
2. Event dependency and communication analysis depends on a native
   `cuda_gpu_trace:base` CSV instead of falling back to SQLite event tables.
3. Kernel classification puts generic communication, attention, MoE, FP8, and
   memory rules in an order that misclassifies known kernels.
4. A single integrity Boolean cannot distinguish a valid raw report from a
   report whose requested analysis is incomplete.

## Architecture

The implementation uses a hybrid boundary:

```text
sglang-nsys-workflow.sh
├── configuration and CLI validation
├── full-offline command construction
├── server-steps command construction
└── parser invocation

sglang_server_steps.py
├── HTTP readiness and profile requests
├── warmup and benchmark client lifecycle
├── log-driven decode detection
├── child-process supervision and cleanup
└── capture/benchmark metadata

scripts/tools/nsys/
├── native report capability and fallback selection
├── SQLite schema introspection and event normalization
├── classification, device, phase, communication, dependency analysis
├── completeness and sanity evaluation
└── Markdown/CSV rendering
```

The shell entry point remains backward compatible and remains responsible for
reading YAML with `yq`. A focused Python helper owns stateful server capture,
because HTTP JSON handling, simultaneous child processes, incremental log
monitoring, timeouts, and signal-safe cleanup are more reliable and directly
testable in Python than in shell.

The parser facade remains `scripts/tools/parse_nsys.py`. Focused analysis code
stays under `scripts/tools/nsys/`; SQLite extraction and completeness logic are
separate modules so neither the facade nor the existing device analyzer becomes
an unbounded collection of responsibilities.

## Command-Line Contract

The workflow accepts:

```text
--capture-mode full-offline|server-steps
--profile-phase startup|prefill|decode|full
--profile-start-step N
--profile-num-steps N
--profile-warmup-prompts N
--profile-concurrency N
--profile-ready-timeout N
--cuda-graph-trace graph|node|none
--layerwise-nvtx auto|true|false
```

Positive-integer validation is applied to step count, warmup prompt count,
concurrency, and timeout. `profile-start-step` accepts zero. Phase values and
CUDA Graph trace values are enums. Parser-related workflow options are only
legal with `--parse`.

`full-offline` accepts the new common metadata options but always records:

```text
capture_scope=startup_and_full_process
profile_phase=full_process
steady_state_guaranteed=false
```

It never labels itself as decode. Phase-specific step controls are rejected or
ignored with an explicit diagnostic rather than silently promising selective
capture.

`server-steps` defaults to a phase-oriented capture and launches:

```text
nsys profile
  --trace-fork-before-exec=true
  --trace=cuda,nvtx,osrt
  --sample=none
  --cpuctxsw=none
  --capture-range=cudaProfilerApi
  --capture-range-end=stop
  --cuda-graph-trace=<graph|node|none>
  --output <prefix>
  python -m sglang.launch_server ...
```

The server receives model/tokenizer, TP size, host/port, dtype, memory fraction,
context length, quantization/load format, trust-remote-code, and compatible
`sglang.extra_args`. The workflow extracts a configured `--port` from extra
arguments when present; otherwise it uses `benchmark.port_base`. Duplicate
server-only and client-only options are filtered rather than forwarded to the
wrong command.

The benchmark client uses `python -m sglang.bench_serving` and reuses dataset,
dataset path, input length, output length, prompt count, concurrency,
model/tokenizer, host, and port. `--profile-concurrency` overrides scenario
concurrency only for this capture.

## Server-Steps Lifecycle

Each scenario uses a deterministic state machine:

```text
STARTING_SERVER
→ WAITING_READY
→ WARMING_UP
→ WAITING_PHASE or STARTING_PROFILE
→ PROFILING
→ WAITING_BENCHMARK
→ STOPPING_SERVER
→ FINALIZING_NSYS
→ VALIDATING_REPORT
→ COMPLETE
```

The Nsight parent process is started in its own process group. Nsight
stdout/stderr goes to `<prefix>.nsys.log`; server stdout/stderr goes to
`<prefix>.server.log`. When Nsight cannot directly separate the wrapped server
streams, the helper passes explicit server log redirection through a small
launcher boundary while retaining Nsight's own stream in the Nsight log.

Readiness polling first tries the SGLang health endpoint supported by the
installed version. If it is absent, it falls back to `/v1/models`. A refused
connection or a non-success response is retried until
`--profile-ready-timeout`; server or Nsight exit is checked on every iteration
and fails immediately.

After readiness, warmup requests run outside the CUDA Profiler API capture
range. The warmup uses the selected workload shape and
`--profile-warmup-prompts`. It must complete successfully before the measured
benchmark starts. This moves model initialization, DeepGEMM JIT, NCCL group
initialization, and allocator convergence out of the intended capture window.

Phase behavior is:

- `prefill`: call `/start_profile` before starting the measured benchmark.
- `decode`: start the measured benchmark in the background, then incrementally
  monitor the server log for stable decode evidence such as `Decode batch` and
  a positive running-request count. Call `/start_profile` only after evidence
  is observed.
- `full`: call `/start_profile` immediately before the measured benchmark.
- `startup`: reserved for deliberate server-start capture and explicitly
  documented as startup-oriented, not steady state.

The `/start_profile` request is:

```json
{
  "start_step": 0,
  "num_steps": PROFILE_NUM_STEPS,
  "activities": ["CUDA_PROFILER"]
}
```

The configured `profile-start-step` is represented in workflow behavior and
metadata. Where the installed SGLang endpoint supports only a relative
`start_step`, phase gating occurs before the request and the body uses the
server-compatible relative value. The exact request and response are logged
without treating a successful TCP connection as a successful profile request.

After the server automatically stops the profiler range, the helper waits for
the benchmark client. It then sends an interrupt to the server process group,
allows a bounded graceful-exit period, escalates only remaining child
processes, waits for Nsight finalization, and verifies that the `.nsys-rep`
exists and is non-empty. Signal handlers apply the same cleanup path on Ctrl+C.

No success metadata is written if readiness, warmup, decode detection,
`/start_profile`, benchmark execution, server health, Nsight finalization, or
report validation fails.

## Layerwise NVTX and CUDA Graphs

`--layerwise-nvtx auto` enables `--enable-layerwise-nvtx-marker` only when the
server configuration disables CUDA Graphs. When CUDA Graphs are enabled, auto
mode emits a warning and leaves layerwise markers disabled. Explicit `true`
with CUDA Graphs also emits a prominent compatibility warning and records that
the combination was user-forced. Explicit `false` always disables it.

Metadata separately records whether CUDA Graph execution is enabled, the
selected Nsight CUDA Graph trace mode, whether layerwise NVTX is enabled, and
whether its use was automatic or explicit.

## Capture Metadata

Capture metadata is written atomically next to the report and includes:

- Git commit and dirty state;
- SHA256 of `sglang-nsys-workflow.sh` and `parse_nsys.py`;
- Nsight Systems version;
- capture mode, scope, requested phase, detected phase, confidence, and
  evidence;
- capture start/end wall time and duration;
- benchmark start/end wall time and duration;
- model, tokenizer, scenario, dataset, prompt count, input/output tokens,
  concurrency, TP size, visible devices, host, and port;
- benchmark result and parsed throughput when available;
- CUDA Graph and layerwise NVTX state;
- DeepGEMM JIT and MoE config fallback evidence;
- commands and all log/report paths.

For decode capture, log-driven `Decode batch` evidence produces HIGH phase
confidence. Metadata timestamps are ISO-8601 with timezone and numeric duration
fields are seconds.

## Nsight 2025.3.1 Native Report Fallback

The default report set no longer assumes these variants exist:

```text
cuda_gpu_kern_sum:base
cuda_kern_exec_sum:base
cuda_gpu_trace:base
cuda_kern_exec_trace:base
nvtx_kern_sum:base
```

The event-level fallback chains are:

```text
cuda_gpu_trace:nvtx-name
→ cuda_gpu_trace
→ SQLite kernel/API/NVTX/memory extraction

cuda_kern_exec_trace:nvtx-name
→ cuda_kern_exec_trace
→ SQLite correlation extraction

nvtx_kern_sum
→ SQLite NVTX attribution
```

Native summary reports that are supported remain useful inputs. The core
kernel summary may itself be derived from normalized SQLite kernel events if a
compatible native report is unavailable, with its source clearly recorded.

`nsys stats --help-reports` merges stdout and stderr. Exit code 1 is accepted
when the body contains a parseable report list. If help is unavailable, direct
report probes continue. Fallback candidates are attempted in order and each
failure, unsupported candidate, and selected source is retained in metadata.

## SQLite Schema Introspection and Event Normalization

The SQLite reader opens the database read-only and begins with:

```sql
SELECT name FROM sqlite_master WHERE type='table';
PRAGMA table_info(<quoted table>);
```

Table and column lookup uses normalized names and aliases. It does not assume
one Nsight schema. It discovers available sources for:

- CUDA kernel start/end, device, context, stream, process, thread, correlation,
  name string ID, and demangled name;
- StringIds and kernel symbol/name tables;
- CUDA runtime/driver API correlation;
- NVTX ranges, domains, process/thread identity, and text;
- CUDA memory copies and memory sets;
- GPU device metadata.

Normalized kernel events include event ID, timestamps, device/context/stream,
process/thread, correlation ID, resolved name, NVTX attribution, classification
fields, and source table. Missing optional fields become `N/A`; missing kernel
identity or timestamps makes event analysis PARTIAL and names the missing table
or column.

SQLite-derived outputs include:

- `kernel_events.csv`;
- per-device kernel summary;
- per-stream timeline;
- runtime communication events;
- same-stream adjacency;
- cross-stream overlap;
- captured GPU list;
- schema inventory and missing-capability diagnostics.

The reader never invents unavailable PID/rank/NVTX/correlation information.

## Kernel Classification

Classification is ordered and preserves `classification_rule` and
`classification_confidence`.

1. Memory/Layout Transform has highest priority for `transpose`, `permute`,
   `copy`, `memcpy`, `memset`, `gather`, `scatter`, `cat`, and `layout`.
   Therefore `deep_gemm::transpose_fp32` is a memory/layout transform.
2. Communication is split into NCCL AllReduce, NCCL AllGather, NCCL
   ReduceScatter, NCCL AllToAll, Custom AllReduce, Custom AllGather,
   Communication Init, and P2P Send/Recv. `ncclCommInitRank` is initialization,
   never runtime communication. `all_reduce_two_shot_kernel` and
   `one_shot_push_kernel` are Custom AllReduce.
3. GEMM matches `deep_gemm`, `sm90_fp8_gemm`, `gemm`, `matmul`, `cutlass`,
   `cublas`, `mma`, and `grouped_gemm` before quantization. NVTX/module evidence
   refines Dense, MoE, or Attention GEMM; otherwise the result is
   `GEMM (unattributed)`.
4. Quant/Dequant requires explicit `quant`, `dequant`, `scaled_quant`,
   `per_token_quant`, `cast_fp8`, `fp8_quant`, or `int8_quant`. Bare `fp8` is
   insufficient.
5. Normalization, Attention, MoE, Elementwise, KV cache, Sampling, and known
   runtime families use explicit lower-priority rules.

All unmatched kernels remain Unknown and all, not only the first 30, are
written to `unknown_kernels.csv`.

## CSV Normalization and Markdown

Every native `nsys stats --format csv` reader uses `csv.reader`, skips preamble
lines such as `Processing [...] with [...]`, locates the real header by aliases,
and converts fields by header name and units. No parser assumes fixed column
positions.

The common reader is used by kernel grid/block, CUDA API, NVTX, GPU memory time,
and GPU memory size sections. Markdown escaping converts sequences and mappings
to stable scalar text or JSON rather than Python list representations.

## Communication and Dependency Analysis

Only runtime collectives contribute to communication metrics. Initialization
ranges and kernels are retained as evidence but excluded from runtime totals.

Outputs include:

- per-device runtime communication duration;
- per-collective count, average, P50, and P95;
- custom versus NCCL totals;
- compute overlap, exposed communication, and exposed ratio;
- same-stream predecessor and successor;
- per-rank arrival skew when rank mapping is supported by direct evidence;
- communication-compute candidate chains.

Cross-stream overlap uses interval intersection and union, not summed duration.
The report states that summed GPU duration is not wall-clock, adjacency is not
a Tensor dependency, overlap is not proof of a fused kernel, and heuristic
scores are not predicted speedups.

## Integrity, Completeness, and Sanity

The parser reports two independent states:

```text
raw_report_integrity = PASS|PARTIAL|FAIL
analysis_completeness = PASS|PARTIAL|FAIL
```

Raw integrity concerns readable, non-empty report/SQLite data and internally
valid kernel timestamps. Analysis completeness concerns whether requested
phase, device, dependency, and communication conclusions can be supported.

Overall PASS is forbidden when:

- dependency analysis was requested without an event trace;
- communication analysis was requested without runtime communication events;
- TP is greater than one but GPU/device/rank information is absent;
- requested phase attribution is UNKNOWN;
- kernel events are initialization-only;
- capture and benchmark durations materially disagree;
- runtime collective count is anomalously near zero.

Sanity checks additionally verify expected TP GPU count, prevent
`ncclCommInitRank` from dominating decode attribution, require nonzero decode
runtime collectives, flag model-loading H2D dominance, flag kernel accumulated
time that is suspiciously small relative to the capture window, and mark
full-offline captures containing DeepGEMM JIT as
`startup_contaminated=true`.

Missing optional evidence results in PARTIAL with a concrete reason. Corrupt,
empty, or unusable core event data results in FAIL. The parser never produces a
fabricated PASS.

## Error Handling

The workflow returns nonzero for invalid CLI combinations, unavailable
dependencies, missing configuration, incompatible TP/model family, readiness
timeout, failed warmup, missing decode evidence, unsuccessful profile HTTP
response, benchmark failure, unexpected server/Nsight exit, finalization
timeout, or empty report.

Cleanup is idempotent and scoped to process groups created by the workflow. It
does not use broad process-name matching and does not delete prior reports.
Temporary metadata is atomically promoted only after capture success.

The parser distinguishes unsupported reports, absent SQLite capabilities,
empty-but-valid event categories, and corrupt data. Optional failures are
warnings plus PARTIAL; core report/export failures are nonzero.

## Testing

Tests cover:

1. workflow CLI validation and help;
2. full-offline dry run and scope metadata;
3. server-steps prefill dry run;
4. server-steps decode dry run;
5. readiness fallback and timeout;
6. exact `/start_profile` JSON body;
7. incremental log-driven decode detection;
8. graceful cleanup and escalation boundaries;
9. valid help body with exit code 1;
10. CSV preamble stripping and header discovery;
11. native report fallback order;
12. SQLite table/column introspection and event normalization;
13. DeepGEMM GEMM classification;
14. transpose memory/layout priority;
15. custom AllReduce versus NCCL and initialization;
16. requested missing analysis produces PARTIAL;
17. TP4 with absent devices cannot PASS;
18. Markdown has no Python list-valued cells.

Local completion runs:

```bash
python -m pytest tests -q
bash -n scripts/sglang-nsys-workflow.sh
python scripts/tools/parse_nsys.py --help
./scripts/sglang-nsys-workflow.sh --help
```

If the local machine exposes only `python3`, equivalent `python3` commands are
also recorded. Existing unrelated test failures are reported separately and
are not silently treated as success.

Server validation first compares repository/parser identity, then runs prefill
and decode captures against the selected Qwen TP4 and DeepSeek TP8 configs. It
verifies `.nsys-rep` size, metadata hashes/version, captured GPUs, phase
evidence, runtime collectives, parser fallbacks, integrity/completeness, and
the generated Markdown/CSV artifacts.
