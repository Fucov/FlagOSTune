# Independent SGLang Nsight Systems Workflow Design

## Goal

Add an Nsight Systems profiling path for SGLang that is completely independent
from the existing Torch Profiler workflow. The new path must reuse existing
`config.yaml` or `config.yaml.<model>` model and benchmark configuration without
changing `scripts/sglang-auto-workflow.sh`, `scripts/sglang-run-workflow.sh`, or
`scripts/tools/sglang_profile_runner.py`.

The supported model configurations are:

- `Qwen3.6-35B-A3B-FP8-TP4`, with tensor parallel size 4.
- `DeepSeek-V4-Flash-FP8-TP8`, with tensor parallel size 8.

The workflow accepts the existing configuration suffixes for these model
families, including scenario-specific names such as `*-Smoke` and
`*-P2048D32C64`. Family recognition uses both the selected configuration suffix
and `model.path`, while the tensor parallel size is always validated as TP4 for
Qwen and TP8 for DeepSeek. This supports the repository's existing files even
when their `model.name` contains an extra scenario label or omits the TP label.

## Command-Line Interface

The entry point is `scripts/sglang-nsys-workflow.sh`.

Required invocation fields:

```text
scripts/sglang-nsys-workflow.sh --model <config-suffix> --nsys
```

Supported options are:

- `--model <suffix>` selects `config.yaml.<suffix>`. If it is omitted, the
  workflow reads the repository-root `config.yaml`.
- `--nsys` explicitly enables this workflow and is required. This prevents an
  accidental unprofiled model run.
- `--nsys-output <prefix-or-path>` controls the report basename. A bare name is
  placed in the default model output directory. A relative or absolute path is
  honored as a custom destination. An optional `.nsys-rep` suffix is removed
  before passing the prefix to `nsys`, because Nsight adds it itself.
- `--scenario <optimized|full|shape>` selects the configured scenario group and
  defaults to `optimized`.
- `--dry-run` validates configuration and prints the exact command without
  loading the model.
- `--help` documents the interface and examples.

The default report prefix is:

```text
results/<model.name>/nsys/<model.name>-<scenario-name>-<timestamp>
```

The resulting report is therefore:

```text
results/<model.name>/nsys/<model.name>-<scenario-name>-<timestamp>.nsys-rep
```

The workflow intentionally does not reuse `paths.results` or
`output.result_dir`: the requested default contract is always the
repository-root `results/<model.name>/nsys/` directory. A custom
`--nsys-output` is the explicit way to select another directory.

## Architecture and Data Flow

The shell workflow owns dependency checks, configuration resolution, supported
model validation, output path construction, and construction of the exact
`nsys profile` command. It reads YAML through `yq`, matching existing repository
shell workflows.

For every scenario in the selected group, it constructs the SGLang offline
throughput arguments from these existing fields:

- `model.path`, `model.tokenizer_path`, and `model.tensor_parallel_size`.
- `serve.trust_remote_code`.
- `sglang.dtype`, `sglang.mem_fraction_static`, `sglang.context_length`,
  `sglang.load_format`, and `sglang.extra_args`.
- `benchmark.dataset_name`, `benchmark.dataset_path`, and scenario input,
  output, concurrency, or `num_prompts` values.

The Nsight run wraps a new small Python launcher. The launcher calls
`torch.cuda.cudart().cudaProfilerStart()`, executes
`sglang.bench_offline_throughput` in the same Python process via `runpy`, and
calls `cudaProfilerStop()` in a `finally` block. This activates the requested
CUDA Profiler API capture range without passing SGLang's `--profile` flag and
therefore without activating Torch Profiler.

The generated command uses exactly the required collection controls:

```text
nsys profile
  --trace=cuda,nvtx,osrt
  --sample=none
  --cpuctxsw=none
  --capture-range=cudaProfilerApi
```

It also supplies the resolved output prefix and enables overwrite so rerunning
an explicitly named output is deterministic. Each configured benchmark
scenario is a separate `nsys profile` invocation and produces a separate
`.nsys-rep`. For multiple scenarios, their scenario names are appended to an
explicit `--nsys-output` prefix to prevent collisions; a single scenario uses
the explicit prefix unchanged.

## Parser

`scripts/tools/parse_nsys.py` accepts one `.nsys-rep` path and an optional top-N
limit. It invokes `nsys stats` separately for these built-in reports:

- `cuda_gpu_kern_sum`
- `nvtx_sum`
- `cuda_api_sum`

Separate invocations avoid ambiguous multi-report CSV streams. The parser reads
CSV output with header aliases so it tolerates minor Nsight column-label
differences. It renders three terminal tables:

1. Top CUDA kernels, including total time, call count, average time, and kernel
   time percentage.
2. NVTX range summary.
3. CUDA API summary.

Kernel time percentage is computed from the total time of all returned CUDA
kernel rows before the top-N display is applied. If Nsight cannot produce a
report or required columns are absent, the parser exits nonzero with the report
name and diagnostic instead of silently returning incomplete data.

## Error Handling

The workflow fails before execution when `nsys`, `yq`, Python, the configuration
file, the requested scenario group, the model path, or the supported
model/tensor-parallel pairing is invalid. Dry-run mode does not require `nsys`
or an installed SGLang package, allowing command validation on development
machines.

The launcher propagates SGLang's exit status. CUDA profiler initialization and
shutdown failures produce explicit errors. The parser checks the input suffix,
file existence, `nsys` availability, subprocess status, and expected report
schema.

## Testing and Compatibility

Tests cover:

- CLI help, required `--nsys`, and unknown arguments.
- Resolution of default and explicit output prefixes.
- Both supported model families and TP mismatch rejection.
- Exact required Nsight options in dry-run output.
- Absence of SGLang `--profile` and Torch Profiler environment variables.
- Scenario-to-SGLang argument mapping.
- CUDA Profiler API launcher start/stop behavior, including exception cleanup.
- Parser CSV normalization, ordering, top-N selection, kernel percentages, and
  failed `nsys stats` diagnostics.

Compatibility verification records hashes or diffs for the three existing
SGLang Torch Profile implementation files before and after development and runs
their current tests. No existing Torch Profile source file is edited.
