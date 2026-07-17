# Reliable SGLang Nsight Server-Steps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reliable phase-selective SGLang server capture and a Nsight 2025.3.1-compatible SQLite-first analysis path while preserving full-offline capture.

**Architecture:** Keep shell configuration and command construction in `sglang-nsys-workflow.sh`, delegate stateful HTTP/process supervision to `sglang_server_steps.py`, and add schema-tolerant SQLite event extraction plus explicit completeness evaluation under `scripts/tools/nsys/`. Native reports remain optional normalized inputs and use ordered fallbacks.

**Tech Stack:** Bash, Python 3 standard library, SQLite, unittest/pytest-compatible tests, Nsight Systems CLI, SGLang HTTP profiling API.

---

### Task 1: Workflow CLI and capture-mode command construction

**Files:**
- Modify: `scripts/sglang-nsys-workflow.sh`
- Modify: `tests/test_sglang_nsys_workflow.py`

- [ ] **Step 1: Write failing CLI and dry-run tests**

Add tests that assert:

```python
def test_server_steps_prefill_dry_run_builds_server_and_client(self):
    result = self.run_workflow(
        suffix, "--nsys", "--dry-run", "--capture-mode", "server-steps",
        "--profile-phase", "prefill", "--profile-num-steps", "4",
        "--profile-warmup-prompts", "2", "--profile-concurrency", "3",
        "--cuda-graph-trace", "node", "--layerwise-nvtx", "auto",
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("sglang.launch_server", result.stdout)
    self.assertIn("sglang.bench_serving", result.stdout)
    self.assertIn("--cuda-graph-trace=node", result.stdout)

def test_full_offline_dry_run_remains_offline(self):
    result = self.run_workflow(suffix, "--nsys", "--dry-run")
    self.assertIn("sglang.bench_offline_throughput", result.stdout)
    self.assertNotIn("sglang.launch_server", result.stdout)
```

Add enum, integer, parser-option, phase/mode, and layerwise/CUDA-Graph validation
tests. Replace the deferred-server test with successful prefill/decode dry runs.

- [ ] **Step 2: Run the focused tests and observe the expected failure**

Run:

```bash
python3 -m unittest tests.test_sglang_nsys_workflow -v
```

Expected: new server-steps tests fail because the mode is deferred and the new
arguments are unknown.

- [ ] **Step 3: Implement workflow options and mode-specific commands**

Add shell variables and `parse_args` cases for every new option. Add helpers:

```bash
validate_nonnegative_int OPTION VALUE
validate_positive_int OPTION VALUE
extract_extra_arg_value OPTION TOKEN_ARRAY
cuda_graph_enabled TOKEN_ARRAY
resolve_layerwise_nvtx GRAPH_ENABLED REQUESTED
build_full_offline_command MODEL_AND_SCENARIO_FIELDS
build_server_steps_command MODEL_AND_SCENARIO_FIELDS
```

`build_full_offline_command` retains `cuda_profiler_launcher.py` and offline
throughput. `build_server_steps_command` invokes the helper with separate
server/client argument delimiters and all required Nsight options.

- [ ] **Step 4: Re-run workflow tests**

Run the focused unittest command and require all workflow tests to pass.

### Task 2: Stateful server-steps lifecycle

**Files:**
- Create: `scripts/tools/sglang_server_steps.py`
- Create: `tests/test_sglang_server_steps.py`
- Modify: `scripts/sglang-nsys-workflow.sh`

- [ ] **Step 1: Write failing unit tests for HTTP and log behavior**

Test pure functions and dependency-injected orchestration:

```python
def test_profile_body_uses_cuda_profiler_steps():
    assert profile_request_body(7) == {
        "start_step": 0,
        "num_steps": 7,
        "activities": ["CUDA_PROFILER"],
    }

def test_decode_detector_requires_decode_and_running_request():
    detector = DecodeDetector()
    assert not detector.feed("Decode batch, running requests: 0")
    assert detector.feed("Decode batch, running requests: 4")

def test_health_probe_falls_back_to_v1_models():
    probe = FakeProbe({"/health": 404, "/health_generate": 404, "/v1/models": 200})
    assert wait_ready("http://127.0.0.1:30001", 1.0, probe.child_alive,
                      request=probe.request) == "/v1/models"
```

Add readiness timeout, failed `/start_profile`, unexpected child exit, cleanup
order, report-empty, and successful lifecycle metadata tests.

- [ ] **Step 2: Run the new test module and observe import failure**

Run:

```bash
python3 -m unittest tests.test_sglang_server_steps -v
```

Expected: module import fails because `sglang_server_steps.py` does not exist.

- [ ] **Step 3: Implement lifecycle primitives**

Provide a frozen `CaptureConfig` carrying command arrays, paths, URL, phase,
step count, timeouts, and metadata. Provide `CaptureError`, stateful
`DecodeDetector`, and the concrete APIs `profile_request_body`, `http_json`,
`wait_ready`, `wait_for_decode`, `terminate_process_group`, and `run_capture`.

Use `urllib.request`, monotonic condition polling, incremental log offsets,
`start_new_session=True`, SIGINT then SIGTERM then SIGKILL for owned children,
and atomic JSON output. Never use process-name matching.

- [ ] **Step 4: Integrate command execution and separated logs**

The helper starts the Nsight-wrapped server and benchmark client, writes
`server.log`, `nsys.log`, `benchmark.log`, records timestamps and HTTP evidence,
waits for automatic profile stop, performs graceful cleanup, waits for Nsight,
and validates a non-empty report before writing success metadata.

- [ ] **Step 5: Re-run lifecycle and workflow tests**

Require both focused modules to pass.

### Task 3: Capture metadata identity and full-offline semantics

**Files:**
- Modify: `scripts/sglang-nsys-workflow.sh`
- Modify: `scripts/tools/sglang_server_steps.py`
- Modify: `tests/test_sglang_nsys_workflow.py`
- Modify: `tests/test_sglang_server_steps.py`

- [ ] **Step 1: Add failing metadata assertions**

Assert capture metadata contains:

```python
for key in (
    "git_commit", "git_dirty", "workflow_sha256", "parser_sha256",
    "nsys_version", "capture_scope", "profile_phase",
    "steady_state_guaranteed",
):
    self.assertIn(key, value)
```

Assert full-offline values are `startup_and_full_process`, `full_process`, and
false. Assert server decode metadata contains requested/detected phase,
confidence/evidence, capture and benchmark times, workload shape, graph state,
and log paths.

- [ ] **Step 2: Observe focused failures**

Run the two workflow/lifecycle test modules.

- [ ] **Step 3: Implement identity and timing metadata**

Use `git rev-parse`, `git status --porcelain`, `hashlib.sha256`, and
`nsys --version`. Store ISO-8601 timestamps and numeric duration seconds.
Parse benchmark throughput conservatively from its log; absent throughput is
`null`, not an invented value.

- [ ] **Step 4: Re-run focused tests**

Require metadata tests to pass.

### Task 4: Nsight report fallback and common CSV normalization

**Files:**
- Modify: `scripts/tools/nsys/collect_stats.py`
- Modify: `scripts/tools/nsys/normalize_stats.py`
- Modify: `scripts/tools/nsys/utils.py`
- Modify: `scripts/tools/parse_nsys.py`
- Modify: `tests/test_nsys_collect_stats.py`
- Modify: `tests/test_nsys_normalize_stats.py`

- [ ] **Step 1: Write failing fallback and preamble tests**

Add:

```python
def test_trace_fallback_prefers_nvtx_name_then_plain():
    assert report_candidates("cuda_gpu_trace") == (
        "cuda_gpu_trace:nvtx-name", "cuda_gpu_trace"
    )

def test_read_csv_rows_skips_processing_preamble():
    path.write_text("Processing [x] with [y]\nTime (%),Name\n100,kernel\n")
    assert read_csv_rows(path) == [{"Time (%)": "100", "Name": "kernel"}]
```

Assert default selection excludes all unsupported `:base` requests listed in
the design and fallback records the selected candidate.

- [ ] **Step 2: Observe focused failures**

Run collect-stats and normalize-stats tests.

- [ ] **Step 3: Implement ordered native fallback**

Represent logical reports separately from native candidate names. Probe
candidates in order, write one normalized logical CSV, and retain candidate
failures plus selected source in `ReportCollection`.

- [ ] **Step 4: Implement common header discovery**

Add a `read_csv_table(path, required_alias_groups=())` helper built on
`csv.reader`. It scans for a header satisfying aliases, then returns normalized
dict rows. Update kernel and native Markdown readers to use it.

- [ ] **Step 5: Re-run focused tests**

Require report and CSV tests to pass.

### Task 5: SQLite schema introspection and event extraction

**Files:**
- Create: `scripts/tools/nsys/sqlite_events.py`
- Create: `tests/test_nsys_sqlite_events.py`
- Modify: `scripts/tools/nsys/models.py`
- Modify: `scripts/tools/nsys/analyze_devices.py`
- Modify: `scripts/tools/parse_nsys.py`

- [ ] **Step 1: Write a version-variant SQLite fixture test**

Create fixture tables for `StringIds`, `TARGET_INFO_GPU`,
`CUPTI_ACTIVITY_KIND_KERNEL`, `CUPTI_ACTIVITY_KIND_RUNTIME`, NVTX, and memcpy
using variant aliases. Assert inventory, resolved names, timestamps, device,
context, stream, PID/TID, correlation, memory events, and captured devices.

Add a missing-kernel-table test that returns a concrete missing capability and
PARTIAL, never fabricated events.

- [ ] **Step 2: Observe module import failure**

Run the new SQLite event test module.

- [ ] **Step 3: Implement introspection and normalized records**

Provide the read-only APIs `inspect_schema(connection) -> SchemaInventory`,
`load_kernel_events(path) -> EventExtraction`, `load_memory_events(path)`,
`load_device_metadata(path)`, and `write_event_artifacts(extraction,
output_dir)`.

Quote identifiers, normalize aliases, join StringIds in Python, and record the
source table/column mapping. Extend `KernelEvent` with optional PID/TID,
correlation, NVTX, rule, confidence, and source fields using defaults that keep
existing tests compatible.

- [ ] **Step 4: Use SQLite when native event trace is unavailable**

In `parse_nsys.py`, prefer a successful native logical trace, otherwise use
SQLite extraction. Always write schema inventory; write event/timeline outputs
when supported.

- [ ] **Step 5: Re-run SQLite, device, dependency, and parser tests**

Require the relevant modules to pass.

### Task 6: Ordered kernel and communication classification

**Files:**
- Modify: `scripts/tools/nsys/classify_kernels.py`
- Modify: `scripts/tools/nsys/models.py`
- Modify: `scripts/tools/nsys/analyze_communication.py`
- Modify: `tests/test_nsys_kernel_analysis.py`
- Modify: `tests/test_nsys_communication_analysis.py`

- [ ] **Step 1: Write failing priority tests**

Add assertions:

```python
self.assertEqual(classify_kernel("deep_gemm::transpose_fp32").category,
                 "Memory/Layout Transform")
self.assertEqual(classify_kernel("sm90_fp8_gemm").category,
                 "GEMM (unattributed)")
self.assertEqual(classify_kernel("all_reduce_two_shot_kernel").category,
                 "Custom AllReduce")
self.assertEqual(classify_kernel("ncclCommInitRank").category,
                 "Communication Init")
```

Also cover all NCCL collective kinds, Custom AllGather, P2P, explicit quant,
bare FP8, normalization, attention, MoE, and elementwise.

- [ ] **Step 2: Observe focused failures**

Run kernel and communication test modules.

- [ ] **Step 3: Implement structured ordered rules**

Return a `Classification` record with category, rule, confidence, and runtime
communication Boolean. Preserve tuple unpacking compatibility only if existing
callers need it. Add optional evidence parameters for NVTX/module attribution.
Write every Unknown row to `unknown_kernels.csv`.

- [ ] **Step 4: Update communication metrics**

Exclude Communication Init. Add collective kind/provider, percentile helper,
per-device/provider summaries, overlap/exposed ratio, predecessor/successor,
and optional arrival skew only when rank evidence exists.

- [ ] **Step 5: Re-run focused tests**

Require classification and communication tests to pass.

### Task 7: Integrity, completeness, phase, and sanity

**Files:**
- Create: `scripts/tools/nsys/evaluate_integrity.py`
- Create: `tests/test_nsys_integrity.py`
- Modify: `scripts/tools/nsys/models.py`
- Modify: `scripts/tools/nsys/analyze_phases.py`
- Modify: `scripts/tools/parse_nsys.py`

- [ ] **Step 1: Write failing status tests**

Cover requested dependency without events, requested communication without a
runtime collective table, TP4 without devices, UNKNOWN requested phase,
initialization-only events, capture/benchmark duration mismatch, decode with
zero runtime collectives, decode dominated by initialization/H2D, suspiciously
small kernel time, and full-offline DeepGEMM JIT contamination.

- [ ] **Step 2: Observe module import failure**

Run the new integrity test module.

- [ ] **Step 3: Implement deterministic evaluation**

Provide `evaluate_integrity(inputs: IntegrityInputs) -> IntegrityResult`.

The result contains `raw_report_integrity`, `analysis_completeness`, reasons,
sanity checks, and flags. Severity combines as FAIL > PARTIAL > PASS. Phase
attribution gives HIGH confidence to metadata carrying log-driven decode
evidence.

- [ ] **Step 4: Integrate parser metadata and exit behavior**

Write both states and reasons to metadata. Corrupt raw data is nonzero. A valid
but PARTIAL report still produces artifacts and returns zero unless the capture
workflow requested strict parse success; it never renders overall PASS.

- [ ] **Step 5: Re-run integrity, phase, parser CLI tests**

Require the relevant tests to pass.

### Task 8: Markdown, documentation, and end-to-end verification

**Files:**
- Modify: `scripts/tools/nsys/render_markdown.py`
- Modify: `tests/test_nsys_render_markdown.py`
- Modify: `tests/test_nsys_documentation.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing Markdown/documentation tests**

Assert both integrity states, full-offline startup caveats, server lifecycle,
fallback order, direct commands, and the absence of Python list strings in
table cells.

- [ ] **Step 2: Observe focused failures**

Run render and documentation tests.

- [ ] **Step 3: Update rendering and README**

Render list/mapping cells as compact JSON or joined scalar text. Document that
summed GPU time is not wall-clock, adjacency is not dependency, overlap is not
fusion, and heuristic score is not speedup. Add directly executable Qwen TP4
prefill/decode and DeepSeek TP8 examples.

- [ ] **Step 4: Run the complete requested verification**

Run:

```bash
python3 -m pytest tests -q
python3 -m unittest discover -s tests -q
bash -n scripts/sglang-nsys-workflow.sh
python3 scripts/tools/parse_nsys.py --help
./scripts/sglang-nsys-workflow.sh --help
git diff --check
```

Record missing local `pytest` separately. Existing unrelated failures must be
identified by test name; all new Nsight-focused tests must pass.

- [ ] **Step 5: Prepare server validation commands**

Provide commands that first compare commit/status/parser/workflow SHA256 and
`nsys --version`, then run one prefill and one decode server-steps capture with
`--parse --analyze-dependencies --analyze-communication`. Do not claim dynamic
success until server output is observed.
