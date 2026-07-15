# Independent SGLang Nsight Systems Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated SGLang Nsight Systems capture workflow and report parser without modifying the existing SGLang Torch Profiler workflow.

**Architecture:** A Bash entry point resolves existing model YAML, validates the supported Qwen TP4 and DeepSeek TP8 families, maps benchmark scenarios to SGLang offline-throughput arguments, and wraps each scenario with the required `nsys profile` options. A Python launcher owns the CUDA Profiler API range, and a separate Python parser normalizes three `nsys stats` reports.

**Tech Stack:** Bash, Mike Farah `yq` v4, Python 3.9+, PyTorch CUDA runtime, SGLang, NVIDIA Nsight Systems, pytest.

---

## File Map

- Create `scripts/sglang-nsys-workflow.sh`: CLI, YAML resolution, validation, command assembly, output naming, Nsight execution.
- Create `scripts/tools/cuda_profiler_launcher.py`: `cudaProfilerStart/Stop` around `runpy.run_module`.
- Create `scripts/tools/parse_nsys.py`: Nsight report invocation, normalization, percentages, terminal tables.
- Create `tests/test_cuda_profiler_launcher.py`, `tests/test_parse_nsys.py`, and `tests/test_sglang_nsys_workflow.py`.
- Modify `README.md`: separate SGLang Nsight usage.
- Never modify `scripts/sglang-auto-workflow.sh`, `scripts/sglang-run-workflow.sh`, or `scripts/tools/sglang_profile_runner.py`.

### Task 1: CUDA Profiler API Launcher

**Files:**
- Create: `scripts/tools/cuda_profiler_launcher.py`
- Create: `tests/test_cuda_profiler_launcher.py`

- [ ] **Step 1: Write failing lifecycle tests**

Create a fake runtime recording `start` and `stop`. Assert the module runner observes:

```python
["sglang.bench_offline_throughput", "--model-path", "/models/qwen"]
```

as `sys.argv`, assert `cudaProfilerStop` runs after both success and `RuntimeError("boom")`, and assert the original `sys.argv` is restored. Add tests for nonzero start and stop return values.

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest tests/test_cuda_profiler_launcher.py -q`

Expected: import failure because `scripts.tools.cuda_profiler_launcher` does not exist.

- [ ] **Step 3: Implement the minimal launcher**

Implement this interface:

```python
def cuda_call_succeeded(result: object) -> bool:
    try:
        return int(result) == 0
    except (TypeError, ValueError):
        return result is None


def run_module_with_cuda_profiler(
    module: str,
    module_args: list[str],
    *,
    cudart=None,
    module_runner=runpy.run_module,
):
    runtime = cudart or load_torch_cudart()
    old_argv = list(sys.argv)
    sys.argv = [module, *module_args]
    started = False
    try:
        result = runtime.cudaProfilerStart()
        if not cuda_call_succeeded(result):
            raise RuntimeError(f"cudaProfilerStart failed: {result}")
        started = True
        return module_runner(module, run_name="__main__")
    finally:
        try:
            if started:
                result = runtime.cudaProfilerStop()
                if not cuda_call_succeeded(result) and sys.exc_info()[0] is None:
                    raise RuntimeError(f"cudaProfilerStop failed: {result}")
        finally:
            sys.argv = old_argv
```

`load_torch_cudart` must import torch lazily, require `torch.cuda.is_available()`, and return `torch.cuda.cudart()`. CLI accepts `--module` with default `sglang.bench_offline_throughput` plus `argparse.REMAINDER`; remove one leading `--`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python3 -m pytest tests/test_cuda_profiler_launcher.py -q`

Expected: all launcher tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/tools/cuda_profiler_launcher.py tests/test_cuda_profiler_launcher.py
git commit -m "feat: add CUDA profiler range launcher"
```

### Task 2: Nsight Stats Parser

**Files:**
- Create: `scripts/tools/parse_nsys.py`
- Create: `tests/test_parse_nsys.py`

- [ ] **Step 1: Write failing parser tests**

Use this fixture and verify descending order and computed percentages:

```python
CUDA_CSV = """Processing report...
Time (%),Total Time (ns),Instances,Avg (ns),Name
60.0,600,3,200,kernel_a
40.0,400,2,200,kernel_b
"""

rows = parse_summary_csv(CUDA_CSV, name_aliases=("Name",))
kernels = kernel_rows_with_percentage(rows)
assert [row.name for row in kernels] == ["kernel_a", "kernel_b"]
assert [row.time_percentage for row in kernels] == pytest.approx([60.0, 40.0])
```

Also test `Range` for NVTX, `Num Calls` for CUDA API, malformed/empty CSV, zero kernel time, and a fake subprocess runner. Assert `run_nsys_report` calls:

```text
nsys stats --report cuda_gpu_kern_sum --format csv --force-export=true REPORT.nsys-rep
```

and includes report name plus stderr on failure.

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest tests/test_parse_nsys.py -q`

Expected: import failure because `scripts.tools.parse_nsys` does not exist.

- [ ] **Step 3: Implement report execution and CSV normalization**

Define:

```python
@dataclass(frozen=True)
class SummaryRow:
    name: str
    total_ns: float
    instances: int
    avg_ns: float
    time_percentage: float


def run_nsys_report(report_path, report_name, *, nsys="nsys", runner=subprocess.run):
    cmd = [nsys, "stats", "--report", report_name, "--format", "csv",
           "--force-export=true", str(report_path)]
    result = runner(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"nsys stats report {report_name!r} failed: {detail}")
    return result.stdout
```

Normalize headers by lowercasing and removing nonalphanumerics. Scan preamble lines until finding `Total Time (ns)`, `Instances|Num Calls|Calls`, `Avg (ns)`, and an accepted `Name|Range` header. Sort rows by total time. `kernel_rows_with_percentage` divides by the sum of all kernel total times and rejects empty/zero totals.

- [ ] **Step 4: Implement rendering and CLI**

CLI is:

```text
parse_nsys.py REPORT.nsys-rep [--top N] [--nsys PATH]
```

Validate file existence, `.nsys-rep` suffix, and positive top-N. Invoke `cuda_gpu_kern_sum`, `nvtx_sum`, and `cuda_api_sum` separately. Render columns `Time %`, `Total ms`, `Calls`, `Avg us`, and `Name` under headings `Top CUDA Kernels`, `NVTX Range Summary`, and `CUDA API Summary`.

- [ ] **Step 5: Run tests and commit**

Run: `python3 -m pytest tests/test_parse_nsys.py -q`

Expected: all parser tests pass.

```bash
git add scripts/tools/parse_nsys.py tests/test_parse_nsys.py
git commit -m "feat: add Nsight Systems summary parser"
```

### Task 3: Independent SGLang Nsight Workflow

**Files:**
- Create: `scripts/sglang-nsys-workflow.sh`
- Create: `tests/test_sglang_nsys_workflow.py`

- [ ] **Step 1: Write failing shell contract tests**

Use temporary executable shims prepended to `PATH` for `yq` and `nsys`. Cover required `--nsys`, unknown/missing options, both model families, TP mismatch, missing scenario group, output prefix rules, multiple scenarios, and the exact dry-run contract:

```python
assert "--trace=cuda,nvtx,osrt" in result.stdout
assert "--sample=none" in result.stdout
assert "--cpuctxsw=none" in result.stdout
assert "--capture-range=cudaProfilerApi" in result.stdout
assert "--profile" not in result.stdout
assert "SGLANG_TORCH_PROFILER" not in result.stdout
assert "results/Qwen3.6-35B-A3B-FP8-TP4-Test/nsys/capture" in result.stdout
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest tests/test_sglang_nsys_workflow.py -q`

Expected: failure because `scripts/sglang-nsys-workflow.sh` does not exist.

- [ ] **Step 3: Implement CLI and validation**

Use `set -euo pipefail`; accept `--model`, `--scenario`, `--nsys`, `--nsys-output`, `--dry-run`, and `--help`. Resolve `config.yaml.<suffix>` or root `config.yaml`. Require `yq` and Python; require `nsys` outside dry-run.

Read model, benchmark, scenarios, and SGLang fields. Recognize Qwen when suffix/name/path contains `Qwen3.6-35B-A3B-FP8` and enforce TP4. Recognize DeepSeek when it contains `DeepSeek-V4-Flash-FP8` and enforce TP8.

- [ ] **Step 4: Implement SGLang argument mapping**

Build an array beginning with:

```bash
sglang_args=(
  --model-path "$model_path"
  --tokenizer-path "$tokenizer_path"
  --dataset-name "$dataset_name"
  --num-prompts "$num_prompts"
  --tp-size "$tensor_parallel"
)
```

For `random`, add dataset path when present and random input/output lengths. For `sharegpt`, add dataset path and ShareGPT output length. Reject other datasets. Map trust-remote-code and optional SGLang dtype, memory fraction, context, quantization, load format, and `extra_args`; split extra args safely with Python `shlex.split` emitting NUL delimiters. Never append `--profile`.

- [ ] **Step 5: Implement output naming and Nsight invocation**

Default to repository-root `results/<model.name>/nsys/<model.name>-<scenario-name>-<timestamp>`. Strip one `.nsys-rep` from an explicit value. Bare names go in the default directory; relative paths containing `/` are project-root relative; absolute paths remain absolute. Append scenario name to an explicit prefix only when multiple scenarios are selected.

Execute this array per scenario:

```bash
cmd=(
  nsys profile
  --trace=cuda,nvtx,osrt
  --sample=none
  --cpuctxsw=none
  --capture-range=cudaProfilerApi
  --force-overwrite=true
  --output "$output_prefix"
  "$PYTHON_EXECUTABLE" "$SCRIPT_DIR/tools/cuda_profiler_launcher.py"
  --module sglang.bench_offline_throughput --
  "${sglang_args[@]}"
)
```

Dry-run prints a shell-escaped command and expected report. Normal mode creates the parent directory, executes directly, and verifies the `.nsys-rep` exists.

- [ ] **Step 6: Run tests and commit**

Run: `python3 -m pytest tests/test_sglang_nsys_workflow.py -q`

Expected: all workflow tests pass.

```bash
chmod +x scripts/sglang-nsys-workflow.sh scripts/tools/cuda_profiler_launcher.py scripts/tools/parse_nsys.py
git add scripts/sglang-nsys-workflow.sh tests/test_sglang_nsys_workflow.py
git commit -m "feat: add independent SGLang Nsight workflow"
```

### Task 4: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a distinct Nsight section**

Document prerequisites, dry-run, custom naming, multiple scenarios, parser outputs, and these real repository examples:

```bash
./scripts/sglang-nsys-workflow.sh \
  --model Qwen3.6-35B-A3B-FP8-TP4-P128D16 --nsys

./scripts/sglang-nsys-workflow.sh \
  --model DeepSeek-V4-Flash-FP8-TP8-Profile-P2048D32C64 \
  --nsys --nsys-output deepseek-profile

python3 scripts/tools/parse_nsys.py \
  results/<model>/nsys/<capture>.nsys-rep --top 20
```

State explicitly that the workflow does not activate or modify Torch Profiler.

- [ ] **Step 2: Check and commit**

Run: `git diff --check && rg -n "SGLang Nsight Systems|sglang-nsys-workflow|parse_nsys" README.md`

Expected: no whitespace errors and all terms present.

```bash
git add README.md
git commit -m "docs: document SGLang Nsight workflow"
```

### Task 5: Compatibility and Final Verification

- [ ] **Step 1: Verify protected files are untouched**

Run:

```bash
git diff bd8d758 -- scripts/sglang-auto-workflow.sh scripts/sglang-run-workflow.sh scripts/tools/sglang_profile_runner.py
```

Expected: empty output.

- [ ] **Step 2: Run new tests**

Run:

```bash
python3 -m pytest tests/test_cuda_profiler_launcher.py tests/test_parse_nsys.py tests/test_sglang_nsys_workflow.py -q
```

Expected: all pass.

- [ ] **Step 3: Run existing SGLang regressions**

Run:

```bash
python3 -m pytest tests/test_sglang_profile_runner.py tests/test_sglang_collect_metadata.py tests/test_sglang_perf_analysis_torch.py -q
```

Expected: all pass.

- [ ] **Step 4: Run syntax and diff checks**

Run:

```bash
bash -n scripts/sglang-nsys-workflow.sh
python3 -m py_compile scripts/tools/cuda_profiler_launcher.py scripts/tools/parse_nsys.py
git diff --check
git status --short
```

Expected: syntax and diff checks succeed. This host lacks `nsys` and `yq`, so the fake-tool integration tests provide local CLI verification.

- [ ] **Step 5: Record target-GPU acceptance commands**

On an NVIDIA host:

```bash
./scripts/sglang-nsys-workflow.sh \
  --model Qwen3.6-35B-A3B-FP8-TP4-P128D16 \
  --nsys --nsys-output qwen-smoke

python3 scripts/tools/parse_nsys.py \
  results/Qwen3.6-35B-A3B-FP8-TP4-P128D16/nsys/qwen-smoke.nsys-rep \
  --top 20
```

Expected: capture creates the report; parser prints CUDA kernels and percentages, NVTX range summary, and CUDA API summary.
