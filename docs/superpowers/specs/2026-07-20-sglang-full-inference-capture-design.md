# SGLang Full-Inference Nsight Capture Design

## Goal

Profile one warmed-up, measured SGLang serving round as a single Nsight Systems
window. The window must include the complete prefill and decode work for every
request in the configured concurrent workload, then feed the trace to the
existing Top-kernel, operator-classification, dependency, communication, and
fusion-candidate analyses.

The two acceptance workloads are:

| Config | Shape | Requests / max concurrency | TP |
| --- | --- | --- | --- |
| `config.yaml.Qwen3.6-35B-A3B-FP8-TP4-P32768D1024C64` | 32768 input / 1024 output | 64 / 64 | 4 |
| `config.yaml.DeepSeek-V4-Flash-FP8-TP8-P8192D256C32` | 8192 input / 256 output | 32 / 32 | 8 |

## Scope

The serving workflow will have one phase-independent capture path named
`server-full`. Phase selection, scheduler-step selection, and decode-log gates
are out of scope because they can truncate a request and do not answer the
whole-inference Top-kernel question.

The existing `full-offline` path remains available as an explicit compatibility
mode. It is not used for the two acceptance workloads because it includes model
startup and does not isolate a warmed-up serving round. `server-full` becomes
the default capture mode.

No model configuration file is added or changed. The two target configurations
already contain the required shape, concurrency, TP, dataset, and server
settings.

## Capture Lifecycle

`nsys profile` launches and owns `sglang.launch_server` with
`--capture-range=cudaProfilerApi`. Merely launching the server does not begin
CUDA collection.

After the server readiness endpoint succeeds, the supervisor executes the same
benchmark workload repeatedly according to `benchmark.num_runs`:

1. Runs `1..num_runs-1` execute outside the capture range and warm the exact
   configured shapes and concurrency.
2. After the warmup runs finish and the server is idle, the supervisor posts
   `/flush_cache`. This preserves compiled/JIT kernels and allocator warmup but
   prevents the measured prompts from reusing warmup KV/prefix cache entries.
3. Immediately before the final run, the supervisor posts `/start_profile`
   with only `activities: [CUDA_PROFILER]`. It deliberately omits `start_step`
   and `num_steps`.
4. The final run uses the scenario's `num_prompts` (falling back to
   `concurrency`) and `max-concurrency`. This entire run is the measured window.
5. After every measured request finishes, the supervisor posts
   `/stop_profile`.
6. The owned Nsight/server process group is terminated gracefully so Nsight can
   finalize a non-empty `.nsys-rep`.

For both acceptance configs, `benchmark.num_runs` is `2`: run 1 is the
uncaptured warmup and run 2 is the only captured round. If a future config uses
`num_runs: 1`, it receives no client warmup and its first round is captured.

This is launch-skip-like behavior based on workload boundaries rather than a
fixed time delay. It cannot start late because compilation or server readiness
took longer than expected, and it cannot stop before a long decode completes.

## Workflow and CLI

`scripts/sglang-nsys-workflow.sh` remains the user-facing command. It will:

- accept `server-full|full-offline`, defaulting to `server-full`;
- remove `--profile-phase`, `--profile-start-step`,
  `--profile-num-steps`, and `--profile-warmup-prompts`;
- read warmup/captured-run count from `benchmark.num_runs`;
- read requests and concurrency from the selected scenario;
- invoke `scripts/tools/sglang_server_capture.py` directly;
- retain report parsing, dependency analysis, communication analysis, CUDA
  graph tracing, layerwise NVTX, readiness timeout, and output-path controls.

The obsolete `scripts/tools/sglang_server_steps.py` compatibility entry point
is removed. This also removes the absolute-path import failure that produced
`ModuleNotFoundError: No module named 'scripts'` before SGLang was launched.

## Metadata and Evidence

Successful server captures write metadata only after the report exists and is
non-empty. The metadata records at least:

- `capture_status: PASS`;
- `capture_mode: server-full`;
- `capture_scope: measured_inference`;
- `inference_scope: prefill_and_decode`;
- `total_runs`, `warmup_runs`, and `captured_run`;
- input tokens, output tokens, request count, max concurrency, and TP size;
- benchmark and capture timestamps/durations;
- cache-flush/start/stop endpoint exchanges and readiness endpoint;
- report size, commands, logs, visible GPUs, git state, script hashes, Nsight
  version, CUDA graph state, layerwise NVTX state, and detected JIT/fallback
  warnings.

A retry clears stale PASS metadata before launching. A failed warmup, measured
benchmark, profile endpoint, server/Nsight process, or empty report is fatal and
must not leave successful metadata.

## Reports

Each acceptance command uses `--parse --analyze-dependencies
--analyze-communication --dependency-trace`. The parser consumes the complete
captured round and writes its existing artifacts, including:

- `summary/nsys_analysis.md` for the human-readable Top-kernel report;
- `summary/operator_hotspots.csv` and
  `summary/kernel_classification.csv` for computation, custom, fused, and
  external-library operator attribution;
- `summary/kernel_events.csv` and `summary/kernel_adjacency.csv` for event and
  dependency detail;
- `summary/communication_*.csv` and `summary/fusion_candidates.csv` for
  communication and fusion analysis;
- `summary/metadata.json` for parser integrity and provenance.

The README will contain exact commands for the two configs and describe the
warmup-run/captured-run boundary. It will not instruct users to run separate
prefill and decode captures.

## Error Handling

The supervisor must preserve separate server, Nsight, and benchmark logs. Error
messages identify the failing lifecycle stage and point to the relevant log.
HTTP non-2xx responses and JSON responses containing `error` or
`success: false` from cache-flush/profile control endpoints are fatal. Cleanup
stops any owned benchmark and server/Nsight process groups without killing
unrelated processes.

## Verification

Tests are written before implementation and cover:

- default and explicit capture-mode validation;
- removal/rejection of phase/step CLI options;
- exact `num_prompts` and `max-concurrency` mapping for C64 and C32 dry runs;
- `num_runs: 2` producing one identical uncaptured warmup run and captured run
  2;
- `/start_profile` omitting step fields and `/stop_profile` occurring only
  after the measured benchmark succeeds;
- `/flush_cache` occurring after the last warmup and before `/start_profile`;
- HTTP rejection, child exit, benchmark failure, timeout, cleanup, stale
  metadata, and empty-report failures;
- direct absolute-path execution of `sglang_server_capture.py` without relying
  on repository-root module imports;
- metadata semantics and README commands.

Local verification consists of focused unit tests, the full unit-test suite,
shell syntax checking, Python compilation, dry runs for both target configs,
and `git diff --check`.

Real `.nsys-rep` files and GPU-derived reports can only be generated on the
Ubuntu host that provides `/data/models`, `/data/yangkw/datasets`, eight GPUs,
SGLang, and Nsight Systems. On that host, acceptance requires both target
commands to finish with non-empty reports, PASS capture metadata, and generated
`summary/nsys_analysis.md` files.
