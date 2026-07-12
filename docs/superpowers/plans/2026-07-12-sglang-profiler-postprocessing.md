# SGLang Profiler Postprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make SGLang profiler reports use mentor percentages, latest-rank trace selection, auditable deduplication, evidence-preserving kernel/source mapping, and explicit communication tables.

**Architecture:** Parse the selected trace into provenance-aware aggregate keys, deduplicate exact GPU events before aggregation, and keep denominator selection independent from communication classification. A dedicated kernel-name YAML supplies SGLang mappings, while the formatter consumes the same classified rows used by the mentor table.

**Tech Stack:** Python 3 dataclasses/argparse/unittest, PyYAML, Markdown/openpyxl reporting, Bash launchers.

---

### Task 1: Trace selection and exact-event deduplication

**Files:**
- Modify: `scripts/tools/sglang_perf_analysis_torch.py`
- Test: `tests/test_sglang_perf_analysis_torch.py`

- [ ] **Step 1: Write failing selection tests**

Add tests which create two root-level TP-0 traces with timestamp prefixes plus an archived newer trace, then assert `select_trace_files(report_dir, "0")` returns only the newest root-level TP-0 path. Add a rank-all case asserting one newest trace per rank.

- [ ] **Step 2: Run selection tests and verify RED**

Run: `python3 -m unittest tests.test_sglang_perf_analysis_torch.SGLangPerfAnalysisTorchTest.test_select_trace_files_uses_latest_root_trace_per_rank -v`

Expected: FAIL because all matching root traces are currently returned.

- [ ] **Step 3: Implement latest-per-rank selection**

Add a capture timestamp helper using the leading numeric filename component with mtime fallback. Group root-level glob results by `extract_rank()`, select `max()` for each requested rank, and return deterministic rank order.

- [ ] **Step 4: Write failing dedup tests**

Create duplicate kernel events with identical `ts/dur/pid/tid/name/correlation/device/stream`, plus a legitimate same-name event at another timestamp. Assert raw count/time includes all three, dedup count/time includes two, and communication-specific filtered count/time is populated.

- [ ] **Step 5: Run dedup tests and verify RED**

Run: `python3 -m unittest tests.test_sglang_perf_analysis_torch.SGLangPerfAnalysisTorchTest.test_parser_deduplicates_only_identical_gpu_events -v`

Expected: FAIL because `RankStats` has no raw/dedup duration fields and counts all events.

- [ ] **Step 6: Implement dedup stats and cache schema**

Extend `RankStats` with raw/dedup GPU and communication counters/times plus filtered counters/times. Build the exact fingerprint before aggregation, skip duplicates, aggregate only dedup events, serialize every new field, increment `CACHE_SCHEMA_VERSION`, and merge fields across ranks.

- [ ] **Step 7: Run focused tests and commit**

Run: `python3 -m unittest tests.test_sglang_perf_analysis_torch -v`

Expected: PASS.

Commit: `git commit -am "fix: select and deduplicate sglang traces"`

### Task 2: Provenance-aware kernel mapping

**Files:**
- Create: `scripts/tools/sglang_kernel_name_mapping.yaml`
- Modify: `scripts/tools/sglang_perf_analysis_torch.py`
- Modify: `scripts/tools/sglang_comm_report_formatter.py`
- Test: `tests/test_sglang_perf_analysis_torch.py`

- [ ] **Step 1: Write failing mapping-priority tests**

Test profiler stack over correlation, correlation over kernel mapping, kernel mapping over source map, and fallback last. Assert the resulting aggregate key retains explicit `source_type`, provider, op kind, communication type, confidence, and source-check flag.

- [ ] **Step 2: Run mapping tests and verify RED**

Run: `python3 -m unittest tests.test_sglang_perf_analysis_torch.SGLangPerfAnalysisTorchTest.test_kernel_mapping_preserves_provenance_priority -v`

Expected: FAIL because aggregate keys currently contain only kind/op/kernel/source and infer provenance from source text.

- [ ] **Step 3: Add the mapping catalog**

Create `sglang_kernel_name_mapping.yaml` with the requested FlashInfer, generic Triton forward, MoE, DeepGEMM, quantization, SGLang custom-all-reduce, and NCCL entries. Add locally audited mappings for high-frequency SGLang kernel families found under `/Users/ykw/Code/Pycharm/sglang/python/sglang` and use non-empty low-confidence fallback source ranges only when no stronger evidence exists.

- [ ] **Step 4: Implement structured provenance**

Add mapping loading/matching helpers to the analyzer, extend the aggregate key with provenance fields, and apply sources in the required priority order. Replace `source_type_for(source_file)` inference with stored provenance. Ensure mapping failures remain `source_type=unknown` instead of being labeled profiler stack.

- [ ] **Step 5: Align formatter classification**

Have formatter mapping results use `source_type=kernel_name_mapping` and explicit mapping `source_file`, provider, op kind, and communication type. Keep MoE compute communication type `none` regardless of neighboring EP evidence.

- [ ] **Step 6: Run tests and commit**

Run: `python3 -m unittest tests.test_sglang_perf_analysis_torch -v`

Expected: PASS.

Commit: `git add scripts/tools/sglang_kernel_name_mapping.yaml scripts/tools/sglang_perf_analysis_torch.py scripts/tools/sglang_comm_report_formatter.py tests/test_sglang_perf_analysis_torch.py && git commit -m "fix: add auditable sglang kernel mapping"`

### Task 3: Mentor denominator and main report table

**Files:**
- Modify: `scripts/tools/sglang_perf_analysis_torch.py`
- Test: `tests/test_sglang_perf_analysis_torch.py`

- [ ] **Step 1: Write failing mentor-percentage tests**

Use a 100 us primary custom all-reduce, 80 us norm, and 20 us NCCL AllGather. Assert primary pct=50% with `total_kernel`, norm pct=80% and NCCL pct=20% with `kernel_excluding_primary_allreduce`, and overall percentages 50/40/10%.

- [ ] **Step 2: Run test and verify RED**

Run: `python3 -m unittest tests.test_sglang_perf_analysis_torch.SGLangPerfAnalysisTorchTest.test_mentor_pct_uses_primary_allreduce_only -v`

Expected: FAIL because all distributed rows currently use total duration.

- [ ] **Step 3: Implement denominator helpers and CLI**

Add `is_primary_allreduce()` covering the exact SGLang/vLLM rules, compute primary and residual totals from dedup aggregates, add default `--pct-mode mentor`, and return `pct`, `pct_denom`, and `overall_pct` for every row.

- [ ] **Step 4: Write failing main-table layout test**

Assert `## Mentor Style CUDA Kernel（按 op_name 聚合）` occurs before `## Top 10 Kernel 源码核查表`, contains the twelve required columns, joins multiple kernel names with `<br>`, and limits positive rows to 80.

- [ ] **Step 5: Implement main table and credibility section**

Replace the old front CUDA table, add the exact percentage explanation, and emit raw/dedup statistics, >5% warning, parser comparability note, source mapping percentages, unresolved appendix threshold, and duration-not-bytes note.

- [ ] **Step 6: Run tests and commit**

Run: `python3 -m unittest tests.test_sglang_perf_analysis_torch -v`

Expected: PASS.

Commit: `git commit -am "fix: align sglang report with mentor percentages"`

### Task 4: Explicit and possible communication sections

**Files:**
- Modify: `scripts/tools/sglang_comm_report_formatter.py`
- Modify: `scripts/tools/sglang_perf_analysis_torch.py`
- Test: `tests/test_sglang_perf_analysis_torch.py`

- [ ] **Step 1: Write failing report-section tests**

Provide custom all-reduce, NCCL, Gloo, MoE compute, and no EP/fusion evidence. Assert explicit communication contains only custom/NCCL/Gloo, possible communication has the required no-evidence sentence, and the corrected MoE conclusion is present.

- [ ] **Step 2: Run test and verify RED**

Run: `python3 -m unittest tests.test_sglang_perf_analysis_torch.SGLangPerfAnalysisTorchTest.test_report_separates_explicit_and_possible_communication -v`

Expected: FAIL because the formatter currently emits one combined communication table.

- [ ] **Step 3: Split sections and correct MoE logic**

Rename the explicit section, create a possible-fusion section driven only by all-to-all/reduce-scatter/dispatch/combine/DeepEP/FlashInfer fusion evidence, exclude MoE compute/routing, and insert the exact no-evidence and MoE conclusion wording.

- [ ] **Step 4: Run tests and commit**

Run: `python3 -m unittest tests.test_sglang_perf_analysis_torch -v`

Expected: PASS.

Commit: `git commit -am "fix: separate explicit and possible communication"`

### Task 5: Full-stack profile mode and final verification

**Files:**
- Modify: `scripts/sglang-auto-workflow.sh`
- Modify: `scripts/sglang-run-workflow.sh`
- Modify: `scripts/tools/sglang_profile_runner.py`
- Modify: `scripts/tools/sglang_collect_metadata.py`
- Test: `tests/test_sglang_profile_runner.py`
- Test: `tests/test_sglang_collect_metadata.py`
- Test: `tests/test_sglang_perf_analysis_torch.py`

- [ ] **Step 1: Write failing profile-detail tests**

Assert light sets stack/shapes/modules/memory to `0`; full_stack sets stack/shapes/modules to `1` and memory to `0`; invalid modes fail. Add shell text assertions that both workflow entrypoints accept and forward `--profile-detail`.

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m unittest tests.test_sglang_profile_runner tests.test_sglang_collect_metadata -v`

Expected: FAIL because `profile_detail` is not supported.

- [ ] **Step 3: Implement profile-detail propagation**

Parse `--profile-detail light|full_stack` in both shell scripts, write it into `current_run.profile_detail`, derive profiler env exclusively from the selected preset, and record the selected detail in run metadata. Preserve old `torch_profiler_light` config as a compatibility fallback when the new value is absent.

- [ ] **Step 4: Run all static verification**

Run:

```bash
python3 -m unittest tests.test_sglang_perf_analysis_torch tests.test_sglang_profile_runner tests.test_sglang_collect_metadata -v
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/tools/sglang_perf_analysis_torch.py scripts/tools/sglang_comm_report_formatter.py scripts/tools/sglang_profile_runner.py scripts/tools/sglang_collect_metadata.py
bash -n scripts/sglang-auto-workflow.sh scripts/sglang-run-workflow.sh scripts/sglang-auto-processing.sh scripts/sglang-run-processing.sh
git diff --check
```

Expected: all tests pass, compilation/syntax checks are silent, and `git diff --check` reports no errors.

- [ ] **Step 5: Commit final workflow change**

Commit: `git add scripts/sglang-auto-workflow.sh scripts/sglang-run-workflow.sh scripts/tools/sglang_profile_runner.py scripts/tools/sglang_collect_metadata.py tests && git commit -m "feat: add sglang profiler detail presets"`
