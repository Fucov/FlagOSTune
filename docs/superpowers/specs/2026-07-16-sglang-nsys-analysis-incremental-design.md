# SGLang Nsight Systems Analysis Incremental Design

## Scope and Constraints

This increment turns the existing SGLang full-offline Nsight capture into a
reproducible analysis pipeline:

```text
full SGLang workload
→ .nsys-rep
→ one SQLite export
→ native Nsight statistics
→ normalized kernel/device/dependency analysis
→ mentor-style Markdown
```

Development happens directly on `main`, as explicitly authorized. Commits are
allowed; pushing is not. Local verification uses fixture CSV/SQLite data, mock
subprocesses, unit tests, shell syntax checks, and Python compilation only.
There is no requirement to run a real GPU, Nsight Systems, NCU, or model
workload locally.

The existing Torch Profiler capture and reporting files are protected and are
not modified. Existing reports, traces, logs, and uncommitted user work are
never removed.

The primary capture and analysis scope is:

```text
capture_mode = full-offline
profile_phase = full
```

Prefill and decode are auxiliary attribution labels inside a full trace. The
complex `server-steps` independent capture mode is deferred. Metadata and CLI
boundaries remain extensible for a future implementation, but the deferred mode
does not block full-run parsing, Top Kernel, device, communication, or
dependency analysis.

## Architecture

`scripts/tools/parse_nsys.py` remains the backward-compatible CLI facade.
Analysis logic moves into a focused standard-library package:

```text
scripts/tools/nsys/
├── __init__.py
├── models.py
├── progress.py
├── export_report.py
├── collect_stats.py
├── normalize_stats.py
├── classify_kernels.py
├── analyze_devices.py
├── analyze_dependencies.py
├── analyze_communication.py
├── analyze_phases.py
├── render_markdown.py
└── utils.py
```

Responsibilities:

- `models.py`: immutable records for reports, kernels, devices, adjacency,
  communication events/chains, fusion candidates, warnings, and metadata.
- `progress.py`: numbered stages, timestamps, elapsed time, commands, file
  sizes, stderr forwarding, heartbeat, logging, subprocess exit propagation,
  and interruption cleanup.
- `export_report.py`: validate input, select or export SQLite, atomic
  `.sqlite.tmp` promotion, reuse policy, disk-space preflight, and export log.
- `collect_stats.py`: detect supported reports, run each native
  `nsys stats` command against SQLite, stream stderr, and write each CSV
  immediately.
- `normalize_stats.py`: tolerate minor header/unit changes and stream native
  CSV into normalized records without requiring pandas.
- `classify_kernels.py`: ordered, documented kernel classification rules.
- `analyze_devices.py`: SQLite device/process/kernel counts, TP validation,
  per-device time, communication split, top family, and imbalance.
- `analyze_dependencies.py`: same-stream predecessor/successor records, gaps,
  NVTX/phase evidence, and relationship confidence.
- `analyze_communication.py`: communication interval overlap, exposed-time
  derived metrics, communication chains, adjacency denominators, shape
  stability proxies, and transparent fusion-candidate scoring.
- `analyze_phases.py`: PREFILL/DECODE/MIXED/UNKNOWN attribution using explicit
  evidence only.
- `render_markdown.py`: fixed 16-section mentor report and data-scope notes.
- `utils.py`: CSV/JSON atomic writes, size/time formatting, header lookup, and
  SQLite schema inspection.

Native Nsight reports remain the statistical source of truth. SQLite is used
for device metadata and streaming event-level analysis, where native summary
reports do not provide the required relationships.

## Input, Export, and Cache Semantics

The CLI accepts either `.nsys-rep` or `.sqlite`.

For `.nsys-rep`:

- The default database is the adjacent same-name `.sqlite`.
- A missing database is exported once.
- A database older than the report is considered stale and is re-exported.
- A current database is reused by default.
- `--force-export` always exports again.
- `--reuse-sqlite` is the default.
- `--no-reuse-sqlite` explicitly disables cache reuse and therefore requests
  a fresh export, equivalent to a cache-bypass policy.
- Export writes `target.sqlite.tmp`; only a successful process promotes it to
  `target.sqlite`.
- Interrupted or failed temporary databases are clearly marked in progress
  metadata and removed; the original `.nsys-rep` and successful CSV files are
  never deleted.

For direct `.sqlite` input, no export is run. Export-only flags are rejected
when they would have no meaningful effect.

The export command is:

```text
nsys export --type sqlite --output <target.sqlite.tmp> <input.nsys-rep>
```

When replacing an existing database, the command uses the installed Nsight
version's supported force-overwrite form.

Before export, available disk space is compared with the input size. Clearly
insufficient space fails before launch. ENOSPC or any other export failure
returns nonzero with the command and log path.

## Progress and Process Control

Progress always goes to stderr. The final Markdown goes to stdout, preserving:

```bash
python3 scripts/tools/parse_nsys.py REPORT | tee summary.txt
```

The total stage count is computed from validation, version/report detection,
export/reuse, selected reports, analyses, metadata, and rendering. Each stage
prints:

- `[stage/total] name`;
- start time and completion time;
- elapsed time;
- complete shell-escaped command, when applicable;
- input and output paths and sizes;
- RUNNING/SUCCESS/WARNING/FAILED status.

Long export and stats commands use `subprocess.Popen`. Their stderr is
forwarded live to the parser's stderr and to the appropriate log. A heartbeat
is printed every ten seconds with elapsed time and current output size. No
internal percentage is invented.

Ctrl+C terminates and then kills only the active subprocess if needed, returns
exit code 130, preserves the original report and completed CSVs, and prevents a
partial SQLite from being reused.

## Reports and Output Files

The default summary directory is `<input-directory>/summary`; it can be
changed with `--output-dir`.

The base report set is:

- `cuda_gpu_kern_sum` (core)
- `cuda_gpu_kern_sum:base`
- `cuda_gpu_kern_gb_sum`
- `cuda_kern_exec_sum:base`
- `cuda_api_sum`
- `nvtx_sum`
- `nvtx_gpu_proj_sum`
- `cuda_gpu_mem_time_sum`
- `cuda_gpu_mem_size_sum`

Dependency or communication analysis adds:

- `cuda_gpu_trace:base`
- `cuda_kern_exec_trace:base`
- `nvtx_kern_sum:base`
- `nvtx_gpu_proj_trace`

`nsys stats --help-reports` is queried first. `--reports` accepts a
comma-separated explicit selection; the core kernel report is always included.
Unsupported optional reports generate visible warnings and metadata entries.
An unsupported or failed core kernel report terminates nonzero. Every successful
CSV is flushed and atomically promoted immediately after its report finishes.

Output includes:

```text
summary/
├── metadata.json
├── progress.log
├── export_sqlite.log
├── cuda_gpu_kern_sum.csv
├── cuda_gpu_kern_sum_base.csv
├── cuda_gpu_kern_gb_sum.csv
├── cuda_gpu_trace_base.csv
├── cuda_kern_exec_sum_base.csv
├── cuda_kern_exec_trace_base.csv
├── cuda_api_sum.csv
├── nvtx_sum.csv
├── nvtx_kern_sum_base.csv
├── nvtx_gpu_proj_sum.csv
├── nvtx_gpu_proj_trace.csv
├── cuda_gpu_mem_time_sum.csv
├── cuda_gpu_mem_size_sum.csv
├── kernel_classification.csv
├── device_summary.csv
├── kernel_adjacency.csv
├── communication_events.csv
├── communication_chains.csv
├── fusion_candidates.csv
├── unknown_kernels.csv
└── nsys_analysis.md
```

Files for analyses that were not requested are still represented in metadata as
not generated; they are not fabricated as successful empty analyses.

## Normalization and Kernel Analysis

Header lookup is normalized for case, punctuation, whitespace, and known
version aliases. Units are recognized from headers and converted to
nanoseconds/bytes without assuming a fixed Nsight version. Missing optional
columns become `N/A`; missing identity/time/count columns make that report a
warning or error according to report criticality.

The full kernel and base-family tables preserve native total time, instances,
average, median, min, max, standard deviation, and Time (%). Kernel time share
uses the sum of all kernel durations as its denominator. Top-N affects display
only, never the denominator.

Classification uses ordered rules with name, regex, category, priority,
description, and source. Categories include Attention/MLA, MoE
Routing/Permute, MoE Expert GEMM, MoE Combine/Unpermute, Mamba/SSM, Dense GEMM,
Norm/Activation, Quant/Dequant, KV Cache, NCCL Communication, Memory/Copy,
Sampling, Other, and Unknown.

Priority guarantees include:

- NCCL before generic memory/copy;
- MoE expert GEMM before Dense GEMM;
- attention GEMM before Dense GEMM.

Unknown kernels remain Unknown. The report includes unknown time share,
instance share, and Top 30 unknown kernels.

Grid/block analysis preserves dimensions and available register/shared-memory
fields. Grid/block is explicitly documented as a kernel launch-shape proxy, not
a Tensor shape.

## Device and TP Analysis

SQLite schema inspection tolerates missing optional columns/tables. Device
analysis streams `CUPTI_ACTIVITY_KIND_KERNEL` and joins available
`TARGET_INFO_GPU`/StringIds metadata.

`device_summary.csv` records device ID, processes, inferred TP rank and its
source, kernel events, summed kernel time, compute and communication time,
communication share, top family, relative time, and imbalance.

Expected GPU count comes only from explicit workflow/config metadata. Qwen TP4
expects four and DeepSeek TP8 expects eight. A mismatch marks data integrity as
failed, recommends checking `--trace-fork-before-exec=true`, and suppresses
trusted multi-GPU comparison conclusions.

## Dependency, Communication, and Fusion Analysis

Event order is grouped by device/context/stream and sorted by GPU start time.
Adjacency rows record previous/next same-stream kernels and gaps. Time adjacency
alone is never called a data dependency.

Communication classification covers NCCL, all-reduce, all-gather,
reduce-scatter, all-to-all, broadcast, send, recv, and barrier forms.

For each communication interval, overlapping compute intervals on the same GPU
are unioned before intersection. Derived values satisfy:

```text
0 <= exposed_communication_ns
   = communication_duration_ns - overlap_compute_ns
   <= communication_duration_ns
```

The report explicitly identifies exposed communication as a FlagOSTune-derived
timeline metric, not an Nsight critical-path metric. Overlapping communication
events are not summed and presented as wall-clock.

Communication chains aggregate Compute→Communication,
Communication→Compute, and Compute→Communication→Compute by phase, NVTX/module,
families, and device. Adjacency rate uses the communication family count in the
same phase/module as denominator. Relation type, evidence, and confidence are
always emitted.

Fusion candidate scoring is a transparent heuristic composed from importance,
exposed communication, frequency, adjacency stability, shape stability, module
attribution, and feasibility. Component scores are retained. The score is only
for candidate screening and is never described as a theoretical speedup.
Distributed collectives are marked as requiring a distributed primitive;
ordinary Triton is never claimed to replace NCCL, and unchecked TLE feasibility
is UNKNOWN.

## Phase and NVTX Attribution

Phase evidence priority is explicit NVTX marker, timestamped SGLang scheduler
log, workflow/profile metadata, then conservative heuristic. Insufficient
evidence remains UNKNOWN. No fixed step or kernel name forces a phase label.

The facade reserves `--phase-log` and `--phase-metadata`. A full trace
without reliable phase evidence still produces all full-run reports and states
that phase attribution is unavailable.

NVTX attribution uses native NVTX kernel/projection reports. Source files and
symbols are only emitted when supported by direct evidence; vague kernel names
never create invented SGLang source paths.

## Workflow Increment

`scripts/sglang-nsys-workflow.sh` remains the current full-offline workflow.
Its Nsight command adds:

```text
--trace-fork-before-exec=true
--capture-range-end=stop
```

and retains:

```text
--trace=cuda,nvtx,osrt
--sample=none
--cpuctxsw=none
--capture-range=cudaProfilerApi
```

Before launch it prints model/config/scenario/TP/visible devices/output and the
complete command. Capture stdout/stderr is saved to a workflow log. Completion
prints report size.

New optional arguments are:

- `--parse`
- `--parse-top`
- `--parse-output-dir`
- `--force-parse-export`
- `--analyze-dependencies`
- `--analyze-communication`
- `--dependency-trace`

`--dependency-trace` alone enables expensive CUDA event tracing. It is off by
default. Parse options are forwarded to the facade. Existing commands and
output naming remain compatible.

Torch profiler flags in config or CLI are rejected with a clear NSys/Torch
mutual-exclusion error. Existing Torch profiler implementation files are not
edited.

The workflow writes adjacent capture metadata containing model, scenario,
workload, TP, visible devices, capture mode/phase, command, paths, config, and
git commit. The parser prefers this metadata and never guesses missing fields.

The deferred `server-steps` mode is documented as a future limitation, not
silently accepted as working behavior.

## Markdown Contract

`nsys_analysis.md` uses the fixed sections:

1. Execution Summary
2. Experiment Environment
3. Workload and Benchmark
4. Data Integrity
5. Top Kernel Families
6. Top Kernel Variants
7. Kernel Classification
8. Multi-GPU and TP Rank
9. CUDA API and Launch/Execution
10. NVTX and Module Attribution
11. Communication Analysis
12. Communication–Compute Candidate Chains
13. Triton/TLE Fusion Candidates
14. Prefill/Decode/Mixed Auxiliary Attribution
15. Torch Profiler Cross-Validation Interface
16. Data Scope and Limitations

Unavailable values are `N/A`, never zero. Required scope statements include:

- kernel Time (%) divides by summed kernel duration;
- TP multi-GPU summed kernel time is not wall-clock;
- CUDA API Time (%) divides by summed CUDA API duration;
- NCCL summed time is not critical-path communication overhead;
- temporal adjacency is not Tensor data dependency;
- grid/block is not Tensor shape;
- exposed communication is a FlagOSTune-derived metric;
- full-run is not decode-only;
- CUDA-graph-disabled results do not represent graph-enabled production;
- conclusions apply to the captured workload only.

The final Markdown is atomically written to `nsys_analysis.md` and then emitted
unchanged to stdout.

## Error Handling

Errors are never silently skipped.

Fatal errors include missing input, missing `nsys`, invalid SQLite, export
failure, insufficient disk space, core kernel report failure, unusable core
columns, and write failure.

Warnings include unsupported/empty optional reports, missing NVTX, missing
optional SQLite tables/columns, incomplete phase attribution, unknown kernels,
and missing workload metadata. Warnings appear in stderr, metadata, and the
Markdown limitations section.

## Testing and Static Verification

Tests use temporary fixture CSV/SQLite databases and fake executables; they do
not require real Nsight or GPUs.

Coverage includes:

- one export per report and atomic promotion;
- reuse of current SQLite, refresh of stale SQLite, and forced export;
- interrupted export does not leave reusable SQLite;
- progress/heartbeat on stderr and Markdown on stdout;
- supported-report detection;
- immediate CSV persistence;
- optional warnings and core failure;
- header/unit compatibility;
- classification priority and Unknown preservation;
- TP4/TP8 validation and per-device aggregation;
- adjacency, gap, interval union, exposed-time bounds, and denominators;
- confidence cannot be HIGH for temporal adjacency alone;
- Markdown section and denominator language;
- backward-compatible workflow arguments and new forwarding;
- protected Torch profiler files remain unchanged.

Verification commands are limited to available local tooling:

```bash
python3 -m unittest discover -s tests -v
bash -n scripts/sglang-nsys-workflow.sh
python3 -m compileall scripts/tools
git diff --check
```

If pytest or shellcheck is unavailable, that limitation is reported and no
system package is installed.
