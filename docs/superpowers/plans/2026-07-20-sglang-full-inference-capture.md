# SGLang Full-Inference Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace phase-selective scheduler-step profiling with one warmed-up, complete serving-round capture for the existing Qwen C64 TP4 and DeepSeek C32 TP8 configurations.

**Architecture:** Keep the Bash workflow as configuration/command assembly and use `sglang_server_capture.py` as the only serving lifecycle supervisor. Runs before `benchmark.num_runs` execute outside CUDA profiler collection; the supervisor then flushes request caches, and the final identical run is bracketed by `/start_profile` and `/stop_profile`. The existing parser analyzes the complete prefill-and-decode report.

**Tech Stack:** Bash, Python 3 standard library, SGLang HTTP profiling endpoints, NVIDIA Nsight Systems CUDA Profiler API capture range, `unittest`.

---

## File Map

- `scripts/sglang-nsys-workflow.sh`: expose the reduced CLI, read `benchmark.num_runs`, and build one `server-full` supervisor command.
- `scripts/tools/sglang_server_capture.py`: own readiness, identical warmup runs, one measured run, manual profile stop, cleanup, and capture metadata.
- `scripts/tools/sglang_server_steps.py`: remove the obsolete compatibility entry point.
- `tests/test_sglang_nsys_workflow.py`: verify default mode, removed flags, run/concurrency mapping, and both real target-config dry runs.
- `tests/test_sglang_server_capture.py`: replace phase-step tests with complete-window lifecycle and error tests.
- `tests/test_sglang_server_steps.py`: remove obsolete phase-step tests.
- `tests/test_nsys_documentation.py`: require the two exact full-window examples and reject phase-step documentation.
- `README.md`: document the single-window workflow and exact acceptance commands.

### Task 1: Drive the reduced workflow CLI with failing tests

**Files:**
- Modify: `tests/test_sglang_nsys_workflow.py`

- [ ] **Step 1: Extend the test config helper with run count**

Change the helper signature and benchmark mapping to:

```python
def make_config(model_name, model_path, tp, scenarios=None, num_runs=2):
    return {
        "model": {
            "name": model_name,
            "path": model_path,
            "tokenizer_path": None,
            "tensor_parallel_size": tp,
        },
        "serve": {"trust_remote_code": True},
        "sglang": {
            "dtype": "bfloat16",
            "mem_fraction_static": 0.75,
            "context_length": 4096,
            "load_format": "auto",
            "extra_args": "--disable-cuda-graph --sampling-backend pytorch",
        },
        "benchmark": {
            "dataset_name": "random",
            "dataset_path": "/datasets/local.json",
            "num_runs": num_runs,
            "scenarios": {
                "optimized": scenarios
                or [{
                    "name": "p128d16_c1",
                    "input_len": 128,
                    "output_len": 16,
                    "concurrency": 1,
                }]
            },
        },
    }
```

- [ ] **Step 2: Replace server-step dry-run tests with complete-window tests**

Add tests with these assertions:

```python
def test_server_full_is_default_and_maps_runs_and_concurrency(self):
    scenarios = [{
        "name": "p32768d1024_c64",
        "input_len": 32768,
        "output_len": 1024,
        "concurrency": 64,
    }]
    suffix = self.write_config(make_config(
        "Qwen3.6-35B-A3B-FP8-TP4-Test",
        "/models/Qwen3.6-35B-A3B-FP8",
        4,
        scenarios,
        num_runs=2,
    ))
    result = self.run_workflow(suffix, "--nsys", "--dry-run")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("sglang_server_capture.py", result.stdout)
    self.assertIn("--total-runs 2", result.stdout)
    self.assertIn("--num-prompts 64", result.stdout)
    self.assertIn("--max-concurrency 64", result.stdout)
    self.assertNotIn("sglang_server_steps.py", result.stdout)
    self.assertNotIn("--profile-phase", result.stdout)
    self.assertNotIn("--profile-num-steps", result.stdout)

def test_deepseek_c32_maps_requests_concurrency_and_tp8(self):
    scenarios = [{
        "name": "p8192d256_c32",
        "input_len": 8192,
        "output_len": 256,
        "concurrency": 32,
    }]
    suffix = self.write_config(make_config(
        "DeepSeek-V4-Flash-FP8-TP8-Test",
        "/models/DeepSeek-V4-Flash-FP8",
        8,
        scenarios,
        num_runs=2,
    ))
    result = self.run_workflow(suffix, "--nsys", "--dry-run")
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn("--total-runs 2", result.stdout)
    self.assertIn("--num-prompts 32", result.stdout)
    self.assertIn("--max-concurrency 32", result.stdout)
    self.assertIn("--tp-size 8", result.stdout)

def test_phase_and_step_options_are_rejected(self):
    suffix = self.write_config(make_config(
        "Qwen3.6-35B-A3B-FP8-TP4-Test",
        "/models/Qwen3.6-35B-A3B-FP8",
        4,
    ))
    for option, value in (
        ("--profile-phase", "prefill"),
        ("--profile-start-step", "0"),
        ("--profile-num-steps", "5"),
        ("--profile-warmup-prompts", "2"),
    ):
        with self.subTest(option=option):
            result = self.run_workflow(
                suffix, "--nsys", "--dry-run", option, value
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("未知参数", result.stderr)
```

Update the help test to require `server-full|full-offline` and reject all four
removed phase/step options. Keep validation coverage for graph trace, layerwise
NVTX, readiness timeout, and explicit `full-offline`.

- [ ] **Step 3: Run the focused workflow tests and confirm RED**

Run:

```bash
python3 -m unittest tests.test_sglang_nsys_workflow -v
```

Expected: the new tests fail because the default is `full-offline`, the command
uses `sglang_server_steps.py`, and `--total-runs` is absent.

- [ ] **Step 4: Commit only the failing tests**

```bash
git add tests/test_sglang_nsys_workflow.py
git commit -m "test: require full-inference SGLang workflow"
```

### Task 2: Implement workflow command assembly

**Files:**
- Modify: `scripts/sglang-nsys-workflow.sh`

- [ ] **Step 1: Reduce the CLI state and validation**

Use these defaults:

```bash
CAPTURE_MODE="server-full"
PROFILE_READY_TIMEOUT=3600
CUDA_GRAPH_TRACE="node"
LAYERWISE_NVTX="auto"
```

Remove variables and argument cases for phase, start step, step count, warmup
prompts, and concurrency override. Change help/validation to:

```bash
--capture-mode MODE   server-full|full-offline，默认 server-full
--profile-ready-timeout N server readiness/benchmark 超时秒数，默认 3600
```

```bash
case "$CAPTURE_MODE" in
    server-full|full-offline) ;;
    *)
        log_error "--capture-mode 仅支持 server-full|full-offline，当前值: ${CAPTURE_MODE}"
        exit 2
        ;;
esac
```

- [ ] **Step 2: Read the configured run count and build identical run commands**

In `main`, read and validate:

```bash
benchmark_num_runs=$(yq_read '.benchmark.num_runs // 1')
if [[ ! "$benchmark_num_runs" =~ ^[1-9][0-9]*$ ]]; then
    log_error "benchmark.num_runs 必须是正整数"
    exit 1
fi
```

Pass it into `run_scenario`. Inside `run_scenario`, keep:

```bash
num_prompts=$(yq_read "${base}.num_prompts // ${base}.concurrency // 1")
scenario_concurrency=$(yq_read "${base}.concurrency // 1")
```

Build one client command used for every run:

```bash
benchmark_cmd=(
    "$PYTHON_EXECUTABLE" -m sglang.bench_serving
    "${client_common[@]}"
    --num-prompts "$num_prompts"
    --max-concurrency "$scenario_concurrency"
)
```

Build the supervisor command with:

```bash
"$PYTHON_EXECUTABLE" "${SCRIPT_DIR}/tools/sglang_server_capture.py" run
--total-runs "$benchmark_num_runs"
--concurrency "$scenario_concurrency"
--profile-ready-timeout "$PROFILE_READY_TIMEOUT"
--nsys-command "${nsys_cmd[@]}"
--benchmark-command "${benchmark_cmd[@]}"
```

Remove `--warmup-command` and all phase/step supervisor arguments. Launch
`exec-server` through `sglang_server_capture.py` directly.

- [ ] **Step 3: Run workflow tests and confirm GREEN**

Run:

```bash
python3 -m unittest tests.test_sglang_nsys_workflow -v
```

Expected: all workflow tests pass.

- [ ] **Step 4: Commit workflow implementation**

```bash
git add scripts/sglang-nsys-workflow.sh
git commit -m "feat: build full-inference SGLang captures"
```

### Task 3: Drive and implement the complete capture lifecycle

**Files:**
- Create: `tests/test_sglang_server_capture.py`
- Delete: `tests/test_sglang_server_steps.py`
- Modify: `scripts/tools/sglang_server_capture.py`
- Delete: `scripts/tools/sglang_server_steps.py`

- [ ] **Step 1: Create focused supervisor tests**

Rename the test class to `SGLangServerCaptureTest`. Copy these tests without
changing their assertions: `test_prepare_capture_outputs_removes_stale_pass_metadata`,
`test_log_flags_require_specific_jit_and_moe_fallback_evidence`,
`test_endpoint_metadata_records_host_port_and_visible_devices`,
`test_wait_ready_falls_back_to_v1_models`,
`test_wait_ready_timeout_names_endpoints`,
`test_command_groups_preserve_option_like_child_arguments`, and
`test_terminate_process_group_stops_owned_child`. In the command-group test,
remove the `--warmup-command` group and assert only `nsys` and `benchmark`.
Import directly from:

```python
from scripts.tools.sglang_server_capture import (
    CaptureError,
    detect_log_flags,
    endpoint_metadata,
    parse_command_groups,
    prepare_capture_outputs,
    profile_request_body,
    flush_cache,
    start_profile,
    stop_profile,
    terminate_process_group,
    wait_ready,
)
```

Replace step-body expectations with:

```python
def test_profile_request_body_covers_until_manual_stop(self):
    self.assertEqual(
        profile_request_body(),
        {"activities": ["CUDA_PROFILER"]},
    )

def test_start_profile_rejects_success_false_response(self):
    with mock.patch(
        "scripts.tools.sglang_server_capture.http_json",
        return_value=(200, {"success": False}),
    ):
        with self.assertRaisesRegex(CaptureError, "rejected"):
            start_profile("http://127.0.0.1:30001")

def test_stop_profile_rejects_success_false_response(self):
    with mock.patch(
        "scripts.tools.sglang_server_capture.http_json",
        return_value=(200, {"success": False}),
    ):
        with self.assertRaisesRegex(CaptureError, "rejected"):
            stop_profile("http://127.0.0.1:30001")

def test_flush_cache_rejects_success_false_response(self):
    with mock.patch(
        "scripts.tools.sglang_server_capture.http_json",
        return_value=(200, {"success": False}),
    ):
        with self.assertRaisesRegex(CaptureError, "rejected"):
            flush_cache("http://127.0.0.1:30001")
```

Add a lifecycle test using patched `Popen`, `_run_logged`, `wait_ready`,
`start_profile`, `stop_profile`, and a temporary non-empty report. Record calls
in a list and assert:

```python
self.assertEqual(
    events,
    ["ready", "warmup-1", "flush", "start", "benchmark-2", "stop"],
)
self.assertEqual(metadata["total_runs"], 2)
self.assertEqual(metadata["warmup_runs"], 1)
self.assertEqual(metadata["captured_run"], 2)
self.assertEqual(metadata["capture_scope"], "measured_inference")
self.assertEqual(metadata["inference_scope"], "prefill_and_decode")
```

Also assert that a failed warmup never calls `flush_cache` or `start_profile`, a
failed flush never calls `start_profile`, a failed measured run attempts cleanup
stop but writes no PASS metadata, and a missing/empty report raises
`CaptureError`.

- [ ] **Step 2: Run supervisor tests and confirm RED**

Run:

```bash
python3 -m unittest tests.test_sglang_server_capture -v
```

Expected: failure because `profile_request_body` still requires a step count,
phase-selective lifecycle fields remain, and the new metadata fields are absent.

- [ ] **Step 3: Remove step/decode code and implement full-window profiling**

Change the module description and profile body to:

```python
"""Capture one complete measured SGLang inference window with Nsight Systems."""

def profile_request_body() -> dict:
    return {"activities": ["CUDA_PROFILER"]}

def start_profile(base_url: str, timeout: float = 10.0) -> dict:
    body = profile_request_body()
    status, response = http_json(
        "POST", base_url.rstrip("/") + "/start_profile", body, timeout
    )
    if not 200 <= status < 300:
        raise CaptureError(
            f"/start_profile returned HTTP {status}: {response!r}; body={body!r}"
        )
    if isinstance(response, Mapping) and (
        response.get("error") or response.get("success") is False
    ):
        raise CaptureError(f"/start_profile rejected request: {response!r}")
    return {"request": body, "status": status, "response": response}

def flush_cache(base_url: str, timeout: float = 30.0) -> dict:
    status, response = http_json(
        "POST", base_url.rstrip("/") + "/flush_cache", None, timeout
    )
    if not 200 <= status < 300:
        raise CaptureError(f"/flush_cache returned HTTP {status}: {response!r}")
    if isinstance(response, Mapping) and (
        response.get("error") or response.get("success") is False
    ):
        raise CaptureError(f"/flush_cache rejected request: {response!r}")
    return {"status": status, "response": response}
```

Remove `DecodeDetector`, `wait_for_decode`, and
`wait_for_profile_completion`. Change command groups to only:

```python
GROUP_MARKERS = {
    "--nsys-command": "nsys",
    "--benchmark-command": "benchmark",
}
```

Run `args.total_runs - 1` identical commands before capture:

```python
for run_number in range(1, args.total_runs):
    _run_logged(
        commands["benchmark"],
        benchmark_log,
        f"warmup-{run_number}",
        child_alive=alive,
        timeout=max(args.profile_ready_timeout, 60.0) * 4,
    )

cache_flush_exchange = flush_cache(args.base_url)
capture_start_iso = now_iso()
capture_start_monotonic = time.monotonic()
profile_exchange = start_profile(args.base_url)
_run_logged(
    commands["benchmark"],
    benchmark_log,
    f"benchmark-{args.total_runs}",
    child_alive=alive,
    timeout=max(args.profile_ready_timeout, 60.0) * 4,
)
benchmark_end_iso = now_iso()
stop_exchange = stop_profile(args.base_url)
capture_end_iso = now_iso()
capture_end_monotonic = time.monotonic()
```

Implement `_run_logged` with `subprocess.Popen`, `start_new_session=True`, an
optional `child_alive` callback, and an optional monotonic deadline. Poll every
0.2 seconds; on each poll call `child_alive`, terminate the command's owned
process group if liveness/timeout fails, and raise `CaptureError`. A nonzero
command exit raises `CaptureError(f"{label} command failed with exit {code}")`.

Write these full-window metadata values:

```python
"capture_mode": "server-full",
"capture_scope": "measured_inference",
"inference_scope": "prefill_and_decode",
"steady_state_guaranteed": args.total_runs > 1,
"total_runs": args.total_runs,
"warmup_runs": args.total_runs - 1,
"captured_run": args.total_runs,
"profile_request": profile_exchange,
"profile_stop_response": stop_exchange,
"cache_flush_response": cache_flush_exchange,
"capture_end_source": "stop_profile_after_measured_benchmark",
"profile_ready_timeout_seconds": args.profile_ready_timeout,
"num_prompts": args.num_prompts,
"concurrency": args.concurrency,
```

The run parser accepts `--total-runs` and `--concurrency`, both positive, and
removes all phase/step/warmup arguments. Rename parser program strings from
`sglang_server_steps.py` to `sglang_server_capture.py`.

In `finally`, if profiling started but normal stop did not succeed, attempt
`stop_profile` before terminating the owned Nsight/server process group. Do not
write metadata until after normal stop, report finalization, and a non-empty
report check.

- [ ] **Step 4: Delete the obsolete compatibility entry point**

Delete `scripts/tools/sglang_server_steps.py` after the workflow and tests no
longer reference it.

- [ ] **Step 5: Run supervisor and workflow tests and confirm GREEN**

Run:

```bash
python3 -m unittest \
  tests.test_sglang_server_capture \
  tests.test_sglang_nsys_workflow -v
```

Expected: all focused tests pass.

- [ ] **Step 6: Commit lifecycle implementation**

```bash
git add scripts/tools/sglang_server_capture.py \
  scripts/tools/sglang_server_steps.py \
  tests/test_sglang_server_capture.py \
  tests/test_sglang_server_steps.py
git commit -m "feat: capture complete SGLang inference rounds"
```

### Task 4: Update reports documentation with the two acceptance workloads

**Files:**
- Modify: `README.md`
- Modify: `tests/test_nsys_documentation.py`

- [ ] **Step 1: Write failing documentation assertions**

Require these values in README:

```python
required = (
    "Qwen3.6-35B-A3B-FP8-TP4-P32768D1024C64",
    "DeepSeek-V4-Flash-FP8-TP8-P8192D256C32",
    "--capture-mode server-full",
    "prefill_and_decode",
    "captured_run",
    "operator_hotspots.csv",
    "kernel_classification.csv",
    "fusion_candidates.csv",
    "nsys_analysis.md",
    "/start_profile",
    "/stop_profile",
)
for value in required:
    self.assertIn(value, text)
for removed in (
    "--capture-mode server-steps",
    "--profile-phase",
    "--profile-start-step",
    "--profile-num-steps",
    "sglang_server_steps.py",
):
    self.assertNotIn(removed, text)
```

- [ ] **Step 2: Run the documentation test and confirm RED**

Run:

```bash
python3 -m unittest tests.test_nsys_documentation -v
```

Expected: failure because README still documents separate prefill/decode
server-step commands.

- [ ] **Step 3: Replace the README Nsight examples and lifecycle explanation**

Document the exact Qwen command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/sglang-nsys-workflow.sh \
  --model Qwen3.6-35B-A3B-FP8-TP4-P32768D1024C64 \
  --nsys \
  --capture-mode server-full \
  --cuda-graph-trace node \
  --layerwise-nvtx auto \
  --nsys-output qwen-p32768d1024-c64-full \
  --parse --parse-top 50 \
  --analyze-dependencies --analyze-communication --dependency-trace
```

Document the exact DeepSeek command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./scripts/sglang-nsys-workflow.sh \
  --model DeepSeek-V4-Flash-FP8-TP8-P8192D256C32 \
  --nsys \
  --capture-mode server-full \
  --cuda-graph-trace node \
  --layerwise-nvtx auto \
  --nsys-output deepseek-p8192d256-c32-full \
  --parse --parse-top 50 \
  --analyze-dependencies --analyze-communication --dependency-trace
```

Explain that `num_runs: 2` means run 1 is outside capture, `/flush_cache` removes
warmup KV/prefix entries, and run 2 is bracketed by `/start_profile` and
`/stop_profile`; both prefill and all decode work enter one report. List the
`.nsys-rep`, adjacent capture metadata, separate logs, and `summary/` artifacts.

- [ ] **Step 4: Run documentation and focused tests**

Run:

```bash
python3 -m unittest \
  tests.test_nsys_documentation \
  tests.test_sglang_nsys_workflow \
  tests.test_sglang_server_capture -v
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md tests/test_nsys_documentation.py
git commit -m "docs: document full-inference Nsight reports"
```

### Task 5: Verify the two real configurations and repository regression suite

**Files:**
- No production changes expected.

- [ ] **Step 1: Run shell and Python static checks**

```bash
bash -n scripts/sglang-nsys-workflow.sh
python3 -m compileall -q scripts tests
git diff --check
```

Expected: all commands exit zero with no syntax or whitespace errors.

- [ ] **Step 2: Dry-run the Qwen C64 TP4 acceptance command**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/sglang-nsys-workflow.sh \
  --model Qwen3.6-35B-A3B-FP8-TP4-P32768D1024C64 \
  --nsys --capture-mode server-full --dry-run \
  --nsys-output qwen-p32768d1024-c64-full \
  --parse --parse-top 50 \
  --analyze-dependencies --analyze-communication --dependency-trace
```

Expected: output includes TP4, 32768/1024, 64 prompts, max concurrency 64,
total runs 2, `sglang_server_capture.py`, and the parse command.

- [ ] **Step 3: Dry-run the DeepSeek C32 TP8 acceptance command**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./scripts/sglang-nsys-workflow.sh \
  --model DeepSeek-V4-Flash-FP8-TP8-P8192D256C32 \
  --nsys --capture-mode server-full --dry-run \
  --nsys-output deepseek-p8192d256-c32-full \
  --parse --parse-top 50 \
  --analyze-dependencies --analyze-communication --dependency-trace
```

Expected: output includes TP8, 8192/256, 32 prompts, max concurrency 32,
total runs 2, `sglang_server_capture.py`, and the parse command.

- [ ] **Step 4: Run all unit tests**

```bash
python3 -m unittest discover -s tests -v
```

Expected: all tests related to this workflow pass. Any unrelated pre-existing
failure is recorded with its exact test name and traceback rather than hidden.

- [ ] **Step 5: Inspect final repository state**

```bash
git status --short
git log --oneline -6
```

Expected: only intentional implementation changes exist, all are committed,
and no generated GPU report is claimed on the local non-GPU host.

- [ ] **Step 6: Run GPU acceptance on the Ubuntu Nsight host**

Execute the two non-`--dry-run` README commands on the host with `/data/models`,
the configured datasets, and eight GPUs. Acceptance requires each command to
produce a non-empty `.nsys-rep`, adjacent metadata with `capture_status: PASS`,
and `summary/nsys_analysis.md`. If that host is not accessible from the current
session, hand off the exact commands and report that GPU acceptance remains to
be run; do not represent dry runs as generated reports.
