# Nsight Help-Reports Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept valid Nsight Systems 2025.3.1 help text despite exit code 1, and fall back to direct report probes without re-exporting a reusable SQLite database.

**Architecture:** Extend the existing streaming runner with an opt-in merged stdout/stderr mode. Parse concrete report tokens independently from process status; if help remains unusable, let the existing collector probe every selected report directly against the resolved SQLite.

**Tech Stack:** Python 3 standard library, `subprocess.Popen`, `unittest`, temporary fake `nsys` executables and SQLite fixtures.

---

### Task 1: Content-aware merged help detection

**Files:**
- Modify: `scripts/tools/nsys/progress.py`
- Modify: `scripts/tools/nsys/collect_stats.py`
- Test: `tests/test_nsys_collect_stats.py`

- [ ] **Step 1: Write failing help compatibility tests**

Extend the fake executable with environment-controlled help text, stream, and
exit status. Add tests asserting:

```python
valid = (
    "The following built-in reports are available:\n"
    "cuda_gpu_kern_sum\n"
    "cuda_api_sum\n"
)
```

- exit 1 plus `valid` on stdout returns both core reports and emits WARNING;
- exit 1 plus empty output raises `CoreReportError`;
- exit 1 plus `valid` on stderr succeeds;
- `parse_help_report_names("nvtx_sum[:nvtx-name][:base|:mangled]")`
  contains `nvtx_sum` but not grammar fragments.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python3 -m unittest tests.test_nsys_collect_stats -v`

Expected: the nonzero valid-help and stderr-help tests raise
`CoreReportError`, and the parser helper import is missing.

- [ ] **Step 3: Implement merged stream and content parsing**

Add an optional flag without changing existing callers:

```python
def run_streaming_command(..., merge_stderr=False):
    stderr_target = subprocess.STDOUT if merge_stderr else subprocess.PIPE
```

When merged, stdout and stderr both flow into the supplied stdout file. Add:

```python
def parse_help_report_names(text: str) -> Set[str]:
    pattern = r"\b((?:cuda|nvtx)_[a-z0-9_]+(?::(?:base|mangled))?)(?=\[|\s|,|$)"
    return set(re.findall(pattern, text, re.IGNORECASE))
```

`detect_supported_reports()` reads merged output before judging the status.
For a nonzero status, it accepts only text containing the built-in-report
heading plus `cuda_gpu_kern_sum` and `cuda_api_sum`, emits WARNING, and returns
the parsed set. Empty/unparseable output or a missing core kernel report raises.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `python3 -m unittest tests.test_nsys_collect_stats -v`

Expected: all report detection/collection tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/tools/nsys/progress.py scripts/tools/nsys/collect_stats.py tests/test_nsys_collect_stats.py
git commit -m "fix: accept valid Nsight help with nonzero status"
```

### Task 2: Direct-probe fallback with SQLite reuse

**Files:**
- Modify: `scripts/tools/nsys/collect_stats.py`
- Modify: `scripts/tools/parse_nsys.py`
- Test: `tests/test_nsys_collect_stats.py`
- Test: `tests/test_parse_nsys_cli.py`

- [ ] **Step 1: Write failing fallback tests**

Add a collector test that passes `supported_reports=None` and verifies it runs
the requested core and optional reports without pre-filtering. Add a CLI fake
configured so `--help-reports` exits 1 with no output while normal stats work.
Create a valid `qwen-tp4-full.sqlite` newer than its `.nsys-rep`, run the
parser, and assert the recorded commands contain stats probes but zero lines
starting with `export `.

Also assert a failed fallback core report returns nonzero while a failed
optional report records WARNING and still produces Markdown.

- [ ] **Step 2: Run the fallback tests and verify RED**

Run:

```bash
python3 -m unittest tests.test_nsys_collect_stats tests.test_parse_nsys_cli -v
```

Expected: `None` is not iterable in the collector or help failure terminates
the facade before report commands run.

- [ ] **Step 3: Implement one-pass direct probing**

Change the collector type to `Optional[Set[str]]`; skip the unsupported-report
precheck when the set is `None`, so the ordinary stats command both probes and
persists each CSV. In `parse_nsys.py`, catch help `CoreReportError`, emit and
retain a `WarningRecord`, set `supported=None`, then call the normal collector.
Append the detection warning to the collection before metadata/rendering.

- [ ] **Step 4: Run focused and parser tests and verify GREEN**

Run:

```bash
python3 -m unittest tests.test_nsys_collect_stats tests.test_parse_nsys tests.test_parse_nsys_cli -v
```

Expected: all tests pass and the reuse regression records no export command.

- [ ] **Step 5: Run static/relevant verification**

```bash
python3 -m unittest tests.test_nsys_export_report tests.test_sglang_nsys_workflow -v
python3 -m compileall -q scripts/tools tests
bash -n scripts/sglang-nsys-workflow.sh
git diff --check
```

Expected: all tests/checks pass without real Nsight or GPU access.

- [ ] **Step 6: Commit**

```bash
git add scripts/tools/nsys/collect_stats.py scripts/tools/parse_nsys.py tests/test_nsys_collect_stats.py tests/test_parse_nsys_cli.py
git commit -m "fix: probe reports when Nsight help is unavailable"
```
