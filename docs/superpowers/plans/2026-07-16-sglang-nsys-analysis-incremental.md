# SGLang Nsight Systems Analysis Incremental Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the existing SGLang full-offline Nsight Systems workflow observable and reproducible by exporting each `.nsys-rep` once, collecting native reports from SQLite, and producing device, communication, dependency, and Markdown analysis without changing Torch Profiler behavior.

**Architecture:** Keep `scripts/tools/parse_nsys.py` as the compatible command-line facade and move implementation into a standard-library `scripts/tools/nsys/` package. Use streamed `Popen` execution and atomic files for external commands, native `nsys stats` CSV as the statistical source of truth, and read-only SQLite queries for event/device relationships. Extend the existing shell workflow incrementally with capture metadata and optional parsing.

**Tech Stack:** Python 3 standard library (`argparse`, `csv`, `dataclasses`, `json`, `sqlite3`, `subprocess`, `threading`), POSIX/Bash, `unittest`, temporary fixture databases and fake `nsys` executables.

---

## Task 1: Add the analysis package, shared models, and normalization primitives

**Files:**
- Create: `scripts/tools/nsys/__init__.py`
- Create: `scripts/tools/nsys/models.py`
- Create: `scripts/tools/nsys/utils.py`
- Create: `scripts/tools/nsys/normalize_stats.py`
- Create: `tests/test_nsys_normalize_stats.py`

**Step 1: Write failing normalization tests**

Cover punctuation/case-insensitive headers, known duration/count/name aliases,
time-unit conversion, missing optional values as `None`, and a fatal error when
the core identity/duration/count columns cannot be found. Include two fixture
header styles representing different Nsight releases.

**Step 2: Run the focused test and confirm RED**

Run: `python3 -m unittest tests.test_nsys_normalize_stats -v`

Expected: import failure because `scripts.tools.nsys` does not exist.

**Step 3: Implement the minimum shared types and helpers**

Add immutable records for warnings, normalized kernel rows, device summaries,
adjacency rows, communication events/chains, fusion candidates, report results,
and capture metadata. Add these public helpers:

```python
def normalize_header(value): ...
def find_column(fieldnames, aliases, required=False): ...
def parse_duration_ns(value, header): ...
def atomic_write_text(path, text): ...
def atomic_write_json(path, value): ...
def format_bytes(value): ...
def load_kernel_summary(path): ...
```

`load_kernel_summary()` must stream CSV rows, preserve the complete kernel
name, distinguish `N/A` from zero, and return a denominator based on summed
kernel duration rather than truncated Top-N rows.

**Step 4: Run the focused test and confirm GREEN**

Run: `python3 -m unittest tests.test_nsys_normalize_stats -v`

Expected: all normalization tests pass.

**Step 5: Commit**

```bash
git add scripts/tools/nsys tests/test_nsys_normalize_stats.py
git commit -m "feat: add Nsight statistics normalization"
```

## Task 2: Implement observable subprocess execution and atomic SQLite export

**Files:**
- Create: `scripts/tools/nsys/progress.py`
- Create: `scripts/tools/nsys/export_report.py`
- Create: `tests/test_nsys_export_report.py`

**Step 1: Write failing export and progress tests**

Use a temporary fake `nsys` executable and cover:

- a missing SQLite invokes exactly one `nsys export` command;
- a current SQLite is reused without launching export;
- a stale SQLite is refreshed;
- `--force-export` and cache bypass refresh it;
- successful `.sqlite.tmp` is atomically promoted;
- failure or `KeyboardInterrupt` never replaces the original report/database;
- progress and forwarded child stderr go to stderr and `progress.log`;
- a heartbeat reports elapsed time and current output size;
- clearly insufficient free space fails before process launch.

**Step 2: Run the focused test and confirm RED**

Run: `python3 -m unittest tests.test_nsys_export_report -v`

Expected: import failure for the new modules.

**Step 3: Implement progress and the streaming runner**

Provide a numbered `ProgressReporter` and a runner with this contract:

```python
def run_streaming_command(command, stdout_path, stderr_log_path,
                          progress, heartbeat_seconds=10.0,
                          popen_factory=subprocess.Popen): ...
```

It must print the shell-escaped command, timestamps, input/output sizes, status,
and elapsed time; forward stderr live using a reader thread; flush log data;
heartbeat while the child is alive; return/raise with the exact exit status;
and terminate then kill the active child on interruption.

**Step 4: Implement export/cache policy**

Expose:

```python
def resolve_sqlite(input_path, output_dir, nsys_path,
                   force_export=False, reuse_sqlite=True,
                   progress=None): ...
```

Validate `.nsys-rep`/`.sqlite`, check the tool and free space, reuse only a
valid database whose mtime is not older than the report, export to
`<target>.sqlite.tmp`, and atomically replace the target after success. Reject
meaningless export flags for direct SQLite input.

**Step 5: Run the focused test and confirm GREEN**

Run: `python3 -m unittest tests.test_nsys_export_report -v`

Expected: all tests pass without a real Nsight installation.

**Step 6: Commit**

```bash
git add scripts/tools/nsys/progress.py scripts/tools/nsys/export_report.py tests/test_nsys_export_report.py
git commit -m "feat: stream Nsight export progress"
```

## Task 3: Detect and collect native reports from SQLite

**Files:**
- Create: `scripts/tools/nsys/collect_stats.py`
- Create: `tests/test_nsys_collect_stats.py`

**Step 1: Write failing report collection tests**

Cover supported-report parsing from `nsys stats --help-reports`, report-name to
file-name mapping, commands using only the SQLite input, absence of
`--force-export=true`, immediate CSV promotion after each success, warning on
unsupported/failed optional reports, and nonzero failure for
`cuda_gpu_kern_sum`.

**Step 2: Run and confirm RED**

Run: `python3 -m unittest tests.test_nsys_collect_stats -v`

Expected: module import failure.

**Step 3: Implement native report selection and collection**

Define an ordered report registry with output names and core/optional status.
The base set includes kernel, grid/block, CUDA API, NVTX, and memory summaries;
dependency/communication flags add trace reports. `--reports` accepts a
comma-separated override while always retaining the core kernel report.

Each report command follows this shape and writes through a temporary CSV:

```text
nsys stats --report <report> --format csv --output <csv.tmp> <input.sqlite>
```

If the installed version instead emits CSV to stdout, the runner redirects
stdout to the temporary path. Promote and flush each CSV before starting the
next report. Record supported, unsupported, failed, empty, and successful
reports for metadata and Markdown.

**Step 4: Run and confirm GREEN**

Run: `python3 -m unittest tests.test_nsys_collect_stats -v`

Expected: all tests pass.

**Step 5: Commit**

```bash
git add scripts/tools/nsys/collect_stats.py tests/test_nsys_collect_stats.py
git commit -m "feat: collect Nsight reports from SQLite"
```

## Task 4: Add kernel classification and multi-GPU integrity analysis

**Files:**
- Create: `scripts/tools/nsys/classify_kernels.py`
- Create: `scripts/tools/nsys/analyze_devices.py`
- Create: `tests/test_nsys_kernel_analysis.py`
- Create: `tests/test_nsys_device_analysis.py`

**Step 1: Write failing classification tests**

Assert ordered precedence: NCCL over generic copy, MoE expert GEMM over dense
GEMM, attention/MLA over dense GEMM, and unmatched names remain `Unknown`.
Verify classification CSV contains full name, base family, category, matched
rule/source, total time, instances, and shares; unknown output includes the Top
30 and full unknown time/instance denominator.

**Step 2: Write failing SQLite/device tests**

Build minimal temporary SQLite schemas for four and eight devices, with
alternate column names. Verify captured GPU/process counts, per-device event
counts and summed time, compute/communication split, top family, imbalance,
and TP mismatch warnings that cite `--trace-fork-before-exec=true`. Missing
optional GPU metadata must produce `N/A`, not crash.

**Step 3: Run and confirm RED**

Run:

```bash
python3 -m unittest tests.test_nsys_kernel_analysis tests.test_nsys_device_analysis -v
```

Expected: import failures.

**Step 4: Implement ordered classification rules**

Implement documented regex rules for Attention/MLA, MoE routing/permute,
expert GEMM, combine/unpermute, Mamba/SSM, dense GEMM, norm/activation,
quant/dequant, KV cache, NCCL, memory/copy, sampling, Other, and Unknown. Keep
rules in data so the emitted CSV can expose the evidence.

**Step 5: Implement schema-tolerant device queries**

Inspect `sqlite_master`/`PRAGMA table_info`, locate CUPTI kernel and optional
GPU/string tables, stream rows grouped by device/process, and validate expected
TP only when authoritative workflow metadata supplies it. Mark integrity
failed rather than presenting trusted rank comparison when counts mismatch.

**Step 6: Run and confirm GREEN**

Run the command from Step 3. Expected: all tests pass.

**Step 7: Commit**

```bash
git add scripts/tools/nsys/classify_kernels.py scripts/tools/nsys/analyze_devices.py tests/test_nsys_kernel_analysis.py tests/test_nsys_device_analysis.py
git commit -m "feat: analyze Nsight kernels and devices"
```

## Task 5: Add event dependency, communication, and fusion-candidate analysis

**Files:**
- Create: `scripts/tools/nsys/analyze_dependencies.py`
- Create: `scripts/tools/nsys/analyze_communication.py`
- Create: `tests/test_nsys_communication_analysis.py`

**Step 1: Write failing event-analysis tests**

Use small event fixtures to verify:

- adjacency is only within the same `(device, context, stream)` and ordered by
  GPU start time;
- gaps are computed without calling adjacency a tensor/data dependency;
- overlapping compute intervals are unioned before communication intersection;
- `0 <= exposed <= duration` for every event;
- Compute→Comm, Comm→Compute, and Compute→Comm→Compute chains retain device,
  phase/module, evidence, relation type, and confidence;
- adjacency denominators are local to the same phase/module;
- temporal adjacency alone can never produce HIGH confidence;
- fusion score exposes each heuristic component and never reports speedup;
- NCCL candidates require distributed primitives and TLE feasibility defaults
  to UNKNOWN.

**Step 2: Run and confirm RED**

Run: `python3 -m unittest tests.test_nsys_communication_analysis -v`

Expected: import failures.

**Step 3: Implement same-stream adjacency and event loading**

Stream required event columns from SQLite/native trace CSV, normalize
timestamps, join available StringIds/NVTX evidence, group and sort per stream,
then emit `kernel_adjacency.csv`. Label the relation `temporal_adjacency` and
retain evidence/confidence fields.

**Step 4: Implement interval and chain analysis**

Classify collective forms, union overlapping compute intervals per GPU,
calculate overlap/exposed time, aggregate chains with correct denominators,
and emit communication event/chain CSVs. Build a transparent fusion-candidate
score from importance, exposed fraction, frequency, stability, attribution,
and feasibility; retain components instead of a single opaque score.

**Step 5: Run and confirm GREEN**

Run the command from Step 2. Expected: all tests pass.

**Step 6: Commit**

```bash
git add scripts/tools/nsys/analyze_dependencies.py scripts/tools/nsys/analyze_communication.py tests/test_nsys_communication_analysis.py
git commit -m "feat: analyze Nsight communication chains"
```

## Task 6: Add conservative phase attribution and the fixed Markdown report

**Files:**
- Create: `scripts/tools/nsys/analyze_phases.py`
- Create: `scripts/tools/nsys/render_markdown.py`
- Create: `tests/test_nsys_render_markdown.py`

**Step 1: Write failing rendering tests**

Require the fixed 16 section headings, Top-N full/base tables, grid/block, CUDA
API/launch/queue/kernel summary, NVTX and GPU projection, memory summaries,
kernel instance count, GPU/device information, per-GPU event counts, warnings,
and every denominator/scope caveat from the approved design. Missing optional
reports must render `N/A` plus a warning, never fabricated zeros.

Also test phase evidence priority: explicit NVTX, timestamped phase log,
workflow metadata, conservative heuristic, then UNKNOWN. A full-run trace must
not be described as decode-only.

**Step 2: Run and confirm RED**

Run: `python3 -m unittest tests.test_nsys_render_markdown -v`

Expected: import failures.

**Step 3: Implement phase analysis and rendering**

Add explicit evidence/source/confidence to each phase attribution. Render the
16 sections in their fixed order, include all available native and derived
tables, data-integrity failures, report warnings, and scope limitations. Write
`nsys_analysis.md` atomically and return its exact text for stdout.

**Step 4: Run and confirm GREEN**

Run the command from Step 2. Expected: all tests pass.

**Step 5: Commit**

```bash
git add scripts/tools/nsys/analyze_phases.py scripts/tools/nsys/render_markdown.py tests/test_nsys_render_markdown.py
git commit -m "feat: render Nsight analysis report"
```

## Task 7: Refactor `parse_nsys.py` into the compatible orchestration facade

**Files:**
- Modify: `scripts/tools/parse_nsys.py`
- Modify: `tests/test_parse_nsys.py`
- Create: `tests/test_parse_nsys_cli.py`

**Step 1: Extend the existing tests before replacing implementation**

Cover old positional input/`--top`/`--nsys` behavior plus:

```text
--output-dir
--force-export
--reuse-sqlite / --no-reuse-sqlite
--reports
--analyze-dependencies
--analyze-communication
--phase-log
--phase-metadata
```

With a fake `nsys`, assert one export, all stats input paths end in `.sqlite`,
progress is stderr-only, final Markdown is stdout-only, each successful CSV is
visible before the next fake report starts, metadata contains success/failure
and provenance fields, and the process returns 130 on simulated interruption.

**Step 2: Run and confirm RED**

Run:

```bash
python3 -m unittest tests.test_parse_nsys tests.test_parse_nsys_cli -v
```

Expected: new facade tests fail against the current repeated-export parser.

**Step 3: Implement orchestration**

Parse and validate arguments, initialize the summary directory and progress
log, resolve one SQLite, detect/collect reports, normalize and classify, run
requested device/dependency/communication/phase analysis, write
`metadata.json`, render Markdown, then write that exact Markdown to stdout.

Metadata must include input/report sizes and mtimes, SQLite path/size, Nsight
version, command line, generated time, report status, model/scenario/workload,
TP/devices, capture mode/phase, git commit, warnings, and analysis artifacts.
Prefer adjacent workflow metadata; leave unavailable values null.

**Step 4: Run and confirm GREEN**

Run the command from Step 2. Expected: all parser tests pass.

**Step 5: Commit**

```bash
git add scripts/tools/parse_nsys.py tests/test_parse_nsys.py tests/test_parse_nsys_cli.py
git commit -m "feat: build SQLite-first Nsight parser"
```

## Task 8: Incrementally extend the full-offline workflow

**Files:**
- Modify: `scripts/sglang-nsys-workflow.sh`
- Modify: `tests/test_sglang_nsys_workflow.py`

**Step 1: Write failing shell-contract tests**

Test help/backward-compatible options and assert the generated command contains
exactly one each of:

```text
--trace-fork-before-exec=true
--capture-range-end=stop
--trace=cuda,nvtx,osrt
--sample=none
--cpuctxsw=none
--capture-range=cudaProfilerApi
```

Cover `--parse`, `--parse-top`, `--parse-output-dir`,
`--force-parse-export`, `--analyze-dependencies`,
`--analyze-communication`, and `--dependency-trace`; ensure expensive CUDA
event tracing is absent unless explicitly enabled. Verify full command/model,
scenario/device/TP/output logging, combined workflow log, report size, adjacent
JSON metadata, parser forwarding, capture exit-code propagation, and cleanup.
Reject Torch profiler flags with a clear mutual-exclusion error.

Accept `--capture-mode full-offline` for explicit metadata. Reject
`server-steps` with a clear deferred/not-implemented message rather than
silently claiming support.

**Step 2: Run and confirm RED**

Run: `python3 -m unittest tests.test_sglang_nsys_workflow -v`

Expected: new assertions fail.

**Step 3: Implement the workflow increment**

Keep the existing launcher/model-config behavior. Add the NSys options, print
and log the complete launch context, run with `pipefail` while preserving the
NSys exit code, print report bytes, and write JSON metadata with Python's
standard library. Invoke the parser only after a successful capture and pass
the selected analysis flags. Validate extra SGLang arguments so they cannot
override profiler ownership or core model/TP settings.

**Step 4: Run and confirm GREEN**

Run the command from Step 2. Expected: all workflow tests pass.

**Step 5: Verify protected Torch files are unchanged**

Run:

```bash
git diff 580688c -- scripts/sglang-auto-workflow.sh scripts/sglang-run-workflow.sh scripts/tools/sglang_profile_runner.py
```

Expected: no output.

**Step 6: Commit**

```bash
git add scripts/sglang-nsys-workflow.sh tests/test_sglang_nsys_workflow.py
git commit -m "feat: make SGLang Nsight workflow observable"
```

## Task 9: Document operation and complete static verification

**Files:**
- Modify: `README.md`
- Modify: tests as required only for discovered implementation defects

**Step 1: Add documentation checks to existing tests**

Assert README contains Qwen TP4 and DeepSeek TP8 full-offline commands, parsing
an existing `.nsys-rep`, prefill/decode auxiliary attribution guidance,
monitoring SQLite growth/parser processes, all output artifact meanings,
stdout/stderr piping behavior, cache/force semantics, and the deferred
`server-steps` limitation.

**Step 2: Update README**

Provide copy-paste commands for:

- Qwen3.6-35B-A3B-FP8-TP4 full-offline capture and analysis;
- DeepSeek-V4-Flash-FP8-TP8 full-offline capture and analysis;
- short prefill/decode attribution using full-run metadata/log evidence;
- standalone parsing, forced export, custom reports, and dependency analysis;
- `watch`, `ls -lh`, and `ps` monitoring of SQLite growth and parser activity.

State that independent `server-steps` capture is intentionally deferred and
that CUDA-graph-disabled short traces are not production graph-path evidence.

**Step 3: Run focused and relevant regression tests**

Run:

```bash
python3 -m unittest \
  tests.test_nsys_normalize_stats \
  tests.test_nsys_export_report \
  tests.test_nsys_collect_stats \
  tests.test_nsys_kernel_analysis \
  tests.test_nsys_device_analysis \
  tests.test_nsys_communication_analysis \
  tests.test_nsys_render_markdown \
  tests.test_parse_nsys \
  tests.test_parse_nsys_cli \
  tests.test_sglang_nsys_workflow -v
```

Then run relevant existing SGLang/Torch tests, without changing protected
implementation files.

**Step 4: Run static verification**

```bash
bash -n scripts/sglang-nsys-workflow.sh
python3 -m compileall scripts/tools tests
git diff --check
git status --short
```

If available, also run `shellcheck scripts/sglang-nsys-workflow.sh`; otherwise
record that it is unavailable. Do not install system dependencies.

**Step 5: Inspect final scope and protected files**

Run:

```bash
git diff --stat 580688c..HEAD
git diff 580688c -- scripts/sglang-auto-workflow.sh scripts/sglang-run-workflow.sh scripts/tools/sglang_profile_runner.py
git log --oneline --decorate -12
```

Expected: only the Nsight workflow/parser/package/tests/docs are changed and
the protected Torch Profiler files have no diff.

**Step 6: Commit documentation**

```bash
git add README.md
git commit -m "docs: explain SGLang Nsight analysis workflow"
```

Do not push. Report fixture/static test evidence and explicitly list the lack
of real Nsight/GPU/model validation plus deferred `server-steps` capture as
remaining limitations.
