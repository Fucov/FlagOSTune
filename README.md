# FlagTune 文档

FlagTune 是一个面向 vLLM + CUDA/FlagGems 的性能测试与分析项目，用于统一管理模型配置、执行 benchmark 测试，并生成 Torch Profiler 分析报告。项目目标是把模型评测流程标准化，便于对比 CUDA 与 FlagGems 在不同模型和场景下的性能表现。

## 1. 模型 config 配置

模型配置文件命名规则：

```bash
config.yaml.<模型名>
```

例如当前仓库内已有：

- `config.yaml.Qwen3.5-35B-A3B`
- `config.yaml.Qwen3.5-397B-A17B`
- `config.yaml.Deepseek-3.2`
- `config.yaml.glm-5-fp8`
- `config.yaml.kimi-K2.5`

新模型建议从模板复制：

```bash
cp config.yaml.template config.yaml.<模型名>
```

一个典型配置如下：

```yaml
model:
  path: /models/Qwen3.5-35B-A3B
  name: Qwen3.5-35B-A3B
  tensor_parallel_size: 1
  tokenizer_path: null

serve:
  gpu_memory_utilization: 0.85
  max_num_batched_tokens: 16384
  max_num_seqs: 2048
  trust_remote_code: true
  reasoning_parser: qwen3
  load_format: auto
  extra_args: ""

benchmark:
  host: 127.0.0.1
  port_base: 2345
  num_runs: 5
  scenarios:
    optimized:
      - name: p32768d1024
        input_len: 32768
        output_len: 1024
        concurrency: 8
    full:
      - name: p128d128
        input_len: 128
        output_len: 128
        concurrency: 100
    shape:
      - name: p1024d1024
        input_len: 1024
        output_len: 1024
        concurrency: 100

target_ops:
  - reciprocal
  - cat
  - gather
  - lt
  - le
  - softmax
  - scatter
  - cumsum_out
  - layer_norm

paths:
  results: results
  reports: reports
  use_model_name: true
```

重点字段：

- `model.path`：模型目录
- `model.name`：vllm服务名
- `model.tensor_parallel_size`：tp并行数
- `benchmark.num_runs`：每个场景重复次数
- `benchmark.scenarios.optimized`：快速测试场景
- `benchmark.scenarios.full`：完整 benchmark 场景
- `benchmark.scenarios.shape`：flaggems 算子shape导出场景
- `paths.results` / `paths.reports`：结果与报告目录

---

## 2. Benchmark 测试

### 2.1 运行测试

使用 `auto-workflow.sh` 运行 benchmark：

```bash
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --mode cuda --scenario optimized
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --mode gems --scenario optimized
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --scenario optimized
```

常用参数：

- `--model`：选择 `config.yaml.<模型名>`
- `--mode cuda|gems|all`：运行目标，默认 `all`(即同时运行 CUDA 和 FlagGems)
- `--device`：GPU 编号
- `--scenario optimized|full|shape`：测试场景
- `--gems-mode`：指定 FlagGems 模式，默认 `all`

### 2.2 结果目录

运行后数据默认保存在：

- `results/<model>/bench_optimized_log/...`
- `results/<model>/bench_log/...`

### 2.3 生成 benchmark 报告

```bash
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow bench
```

默认会处理 optimized benchmark 结果，并在下面生成报告：

```bash
reports/<model>/bench-optimized-report-<date>.md
```

---

## 3. Torch Profiler 测试与报告

### 3.1 运行 Torch Profiler 测试

推荐分别采集 CUDA 和 FlagGems 的 profiler 原始数据：

```bash
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --mode cuda --torch
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --mode gems --torch
```

说明：

- `--torch` 模式下会强制使用 `benchmark.num_runs=2`
- 如需一次性同时采集cuda和flagems双侧数据，也可以直接执行：
  - `./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --torch`
- profiler 原始数据默认输出到：
  - `results/<model>/torch-raw/report-cuda`
  - `results/<model>/torch-raw/report-gems-all`

### 3.2 生成 Torch Profiler 报告

处理单侧结果：

```bash
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow torch --mode cuda
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow torch --mode gems
```

执行对比分析：不用执行单侧处理

```bash
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow torch --mode compare
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow torch --mode compare --rank 0
```

说明：

- `--mode compare` 只对已有的 CUDA / FlagGems profiler 结果做对比分析
- `--rank` 默认为 `0`，也可指定 `all`
- 若直接执行 `./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow torch`，默认等价于 `--mode compare`

生成文件：

- `reports/<model>/perf_analysis_torch.md`
- `reports/<model>/perf_analysis_torch.xlsx`
- `reports/perf_summary_torch.md`
- `reports/perf_summary_torch.xlsx`

如需汇总多个模型的 torch profiling 报告：

```bash
python scripts/tools/perf_summary_torch.py
```

报告内容包括：

- CUDA kernel 排序结果
- FlagGems kernel 排序结果
- CUDA / FlagGems 对比表
- 按 `rank` 维度的 profiler 统计

### 3.3 SGLang Nsight Systems Profiling

该入口独立于已有 Torch Profiler workflow；两者不能同时启用。运行服务器需
安装 `nsys`、`yq` v4、带 CUDA 的 PyTorch 和 SGLang。本地无 GPU/NSys 时可先
追加 `--dry-run`，检查模型配置、设备、TP、输出路径和完整命令。

Qwen TP4 prefill scheduler-step 采集、SQLite 导出和通信分析：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/sglang-nsys-workflow.sh \
  --model Qwen3.6-35B-A3B-FP8-TP4-P128D16 \
  --nsys \
  --capture-mode server-steps \
  --profile-phase prefill \
  --profile-start-step 0 \
  --profile-num-steps 1 \
  --profile-warmup-prompts 2 \
  --profile-concurrency 1 \
  --cuda-graph-trace node \
  --layerwise-nvtx auto \
  --nsys-output qwen-tp4-prefill \
  --parse \
  --parse-top 20 \
  --parse-output-dir results/Qwen3.6-35B-A3B-FP8-TP4-P128D16/nsys/qwen-tp4-prefill-summary \
  --analyze-dependencies \
  --analyze-communication \
  --dependency-trace
```

DeepSeek TP8 decode 采集由 server log 驱动，不使用固定 sleep 猜测开始时间：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
./scripts/sglang-nsys-workflow.sh \
  --model DeepSeek-V4-Flash-FP8-TP8-Profile-P2048D32C64 \
  --nsys \
  --capture-mode server-steps \
  --profile-phase decode \
  --profile-start-step 0 \
  --profile-num-steps 5 \
  --profile-warmup-prompts 2 \
  --profile-concurrency 64 \
  --cuda-graph-trace node \
  --layerwise-nvtx auto \
  --nsys-output deepseek-tp8-decode \
  --parse \
  --parse-top 20 \
  --parse-output-dir results/DeepSeek-V4-Flash-FP8-TP8-Profile-P2048D32C64/nsys/deepseek-tp8-decode-summary \
  --analyze-dependencies \
  --analyze-communication \
  --dependency-trace
```

`server-steps` 使用 `nsys profile` 启动 `python -m sglang.launch_server`，等待
`/health_generate`、`/health` 或 fallback `/v1/models` ready，再在 capture 外运行
`sglang.bench_serving` warmup。prefill/full 在正式 benchmark 前调用
`/start_profile`；decode 先后台启动 benchmark，随后只扫描 benchmark 启动后新增的
server log，检测到 `Decode batch` 且 running request 大于 0 后才调用
`/start_profile`。请求体固定为：

```json
{"start_step": 0, "num_steps": 5, "activities": ["CUDA_PROFILER"]}
```

phase gate 已经决定采集开始时机，所以 HTTP 使用相对 `start_step=0`；
`--profile-start-step` 保存用户请求值和兼容 metadata，不把它伪报为 endpoint 的绝对
step。指定 `num_steps` 后 SGLang 会自动调用 `cudaProfilerStop()`；workflow 等待
server log 的 `Profiling done` 证据，之后等待 benchmark 完成、优雅停止 server 并
等待 Nsight finalize。该路径不调用手动 `/stop_profile`，避免把指定 steps 扩大成
整个 benchmark capture。

`full-offline` 继续使用 `cuda_profiler_launcher.py` →
`sglang.bench_offline_throughput`，只用于 startup/full-process 调查。metadata 固定写入
`capture_scope=startup_and_full_process`、`profile_phase=full_process` 和
`steady_state_guaranteed=false`。它包含模型初始化和 NCCL init，可能包含 DeepGEMM
JIT，不能用于稳定 decode 通信占比，也不会自动标注为 decode。`server-steps` 的
`startup` 发生在 HTTP ready 后，只代表 post-load warmup；真正的模型启动采集应使用
`full-offline --profile-phase startup`。

`--dependency-trace` 会增加较昂贵的 CUDA event tracing，仅在需要事件级分析时
启用。只需要全窗口 Top 算子、NVTX、CUDA API 和多卡汇总时可删除最后三项分析参数。
采集命令固定包含 `--trace-fork-before-exec=true` 和
`--capture-range-end=stop`。输出包括 `.nsys-rep`、`.nsys.log`、相邻的
`.metadata.json` 和指定 summary 目录内的 `nsys_analysis.md`。
Nsight Systems 的 `--cuda-graph-trace` 只接受 `graph|node`；workflow 的
`--cuda-graph-trace none` 表示不向 Nsight 传该选项（使用 Nsight 默认行为），
而不是传递无效的 `--cuda-graph-trace=none`。

报告主表把 capture range 内所有 GPU kernel 按归一化 family 累计总时间后统一排序，
phase 由 workflow metadata、server log 和 NVTX 证据共同校验。每个 Top 算子都给出
通算融合 `YES / NO / UNKNOWN`：

- `YES`：同一个 kernel 名称同时具备明确 fusion、通信和模型计算证据。
- `NO`：可确认是独立计算、独立通信或通信初始化 kernel。
- `UNKNOWN`：名称或映射证据不足，需要源码或 profiler stack 继续确认。

same-stream 邻接和 cross-stream overlap 只能生成融合候选，不能直接判定已经融合。
多卡 `Total (ms)` 是所有已采集 GPU kernel duration 的累计值，不等于 wall-clock
latency。`full-offline` 仅保留为显式的 startup/full-process 调查模式；它包含模型
加载、NCCL init、allocator warmup，并可能包含 DeepGEMM JIT，不用于主热点结论。

server stdout/stderr、Nsight stdout/stderr 和 benchmark log 分别写入
`.server.log`、`.nsys.log` 和 `.benchmark.log`。readiness、warmup、decode evidence、
`/start_profile`、自动 `Profiling done`、benchmark、server、Nsight finalize 或空
report 任一失败时，workflow 返回非零且不写 PASS metadata。

单独解析已有报告（进度写 stderr，Markdown 写 stdout，因此 `tee` 可用）：

```bash
python3 scripts/tools/parse_nsys.py \
  results/Qwen3.6-35B-A3B-FP8-TP4-P128D16/nsys/qwen-tp4-full.nsys-rep \
  --output-dir results/Qwen3.6-35B-A3B-FP8-TP4-P128D16/nsys/qwen-tp4-full-summary \
  --top 20 \
  --analyze-dependencies \
  --analyze-communication \
  | tee qwen-nsys-summary.txt
```

默认会复用 mtime 不早于 `.nsys-rep` 的相邻 SQLite。强制重新导出或禁用缓存：

```bash
python3 scripts/tools/parse_nsys.py REPORT.nsys-rep --force-export
python3 scripts/tools/parse_nsys.py REPORT.nsys-rep --no-reuse-sqlite
python3 scripts/tools/parse_nsys.py REPORT.sqlite --reports cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum
```

Nsight Systems 2025.3.1 的事件 report 使用 fallback：
`cuda_gpu_trace:nvtx-name → cuda_gpu_trace → direct SQLite query`、
`cuda_kern_exec_trace:nvtx-name → cuda_kern_exec_trace → direct SQLite query`，
以及 `nvtx_kern_sum → direct SQLite attribution`。`nsys stats --help-reports`
即使 exit 1，只要输出含有效 report body 仍会继续。metadata 分别记录
`raw_report_integrity=PASS|PARTIAL|FAIL` 和
`analysis_completeness=PASS|PARTIAL|FAIL`，不会把缺少事件、TP device、phase 或
runtime collective 的分析伪报为 PASS。

大型 TP4/TP8 报告解析时可在另一个终端监控真实进度：

```bash
watch -n 5 'ls -lh REPORT.nsys-rep REPORT.sqlite SUMMARY/*.csv 2>/dev/null'
tail -f SUMMARY/progress.log SUMMARY/export_sqlite.log
ps -ef | grep -E '[p]arse_nsys|[n]sys (export|stats)'
```

summary 主要文件含义：

| 文件 | 含义 |
| --- | --- |
| `metadata.json` | 输入、SQLite、NSys 版本、模型/TP/设备、report 状态和 warning |
| `progress.log` / `export_sqlite.log` | 阶段、命令、heartbeat、导出 stderr |
| `cuda_gpu_kern_sum.csv` | 完整 kernel 汇总；base family 由 parser 归一化 |
| `operator_hotspots.csv` | 全推理窗口统一热点排名及逐项通算融合判断 |
| `cuda_gpu_kern_gb_sum.csv` | kernel grid/block 启动形状汇总 |
| `cuda_api_sum.csv` / `cuda_kern_exec_sum.csv` | CUDA API 与 launch/queue/kernel 时间 |
| `nvtx_sum.csv` / `nvtx_gpu_proj_sum.csv` | NVTX range 与 GPU projection |
| `cuda_gpu_mem_time_sum.csv` / `cuda_gpu_mem_size_sum.csv` | GPU memory 操作时间和大小 |
| `cuda_gpu_trace.csv` / `cuda_kern_exec_trace.csv` | 可选 native 事件 trace；不可用时直读 SQLite |
| `nvtx_kern_sum.csv` / `nvtx_gpu_proj_trace.csv` | 可选 NVTX-kernel/投影 trace |
| `kernel_events.csv` / `stream_timeline.csv` / `sqlite_schema.json` | SQLite event、stream 时间线与 schema introspection |
| `kernel_classification.csv` / `unknown_kernels.csv` | 规则分类和未识别 kernel |
| `device_summary.csv` | 每张 GPU 的事件数、累计时间、计算/通信和不均衡 |
| `kernel_adjacency.csv` | same-stream 时序邻接；不代表 Tensor 数据依赖 |
| `communication_events.csv` | 通信 overlap 与派生 exposed 时间 |
| `communication_summary.csv` / `communication_arrival_skew.csv` | 每设备/collective P50/P95、provider 和可证实时的 arrival skew |
| `communication_chains.csv` / `fusion_candidates.csv` | 通信-计算链和透明启发式候选 |
| `nsys_analysis.md` | 固定 16 节最终分析报告 |

---

## 4. Gems shape 导出

如果需要导出 FlagGems 的 shape 信息，先运行 shape 场景：

```bash
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --scenario shape
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --scenario shape --gems-mode mm
```

shape 原始导出文件默认保存在：

- `results/<model>/gems-config-shape/gems-all.txt`
- `results/<model>/gems-config-shape/marker.txt`

再执行 shape 处理：

```bash
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow shape
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow shape --gems-mode mm #只导出mm算子的shape
```

处理后会在下面生成按场景拆分的 shape 文件：

```bash
reports/<model>/shape/*.txt
```

例如：

- `reports/Qwen3.5-35B-A3B/shape/Qwen3.5-35B-A3B-p1024d1024.txt`
- `reports/Qwen3.5-35B-A3B/shape/Qwen3.5-35B-A3B-p32768d1024.txt`

---

## 5. Pretune 功能

FlagGems 的 pretune 功能可提前对指定场景进行性能测试，筛选并缓存更优的 kernel 配置，供后续 benchmark 复用，从而提升整体测试性能表现。

Gems Shape 场景 MM 性能测试

```bash
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --scenario shape --gems-mode mm --gems-once true
```

Gems Shape 场景 MM pretune 性能测试
```bash
./scripts/auto-workflow.sh --model Qwen3.5-35B-A3B --device 0 --scenario shape --gems-mode mm --gems-once true --pretune
```

说明：`--gems-once` 在`shape`场景写默认值为 `false`，需要改为 `true` 否则影响性能

生成 MM pretune性能测试报告

```bash
./scripts/auto-processing.sh --model Qwen3.5-35B-A3B --workflow shape --report
```

## 6. function_test.sh 说明

[`function_test.sh`](function_test.sh) 是一个简单的功能验证脚本，用来串联本仓库常用命令，方便快速执行以下流程：

- optimized benchmark 测试与报告生成
- torch profiling 采集、单侧分析、对比分析
- 多模型 torch profiling 汇总
- FlagGems shape 导出与解析
- FlagGems pretune 性能测试

使用时可直接参考脚本中的命令顺序，按需手动执行，或将其作为日常回归测试的参考清单。

---

## 7. SGLang 多卡分布式算子 Torch Profiler

本流程用于分析 SGLang 原生执行路径下的多卡分布式算子使用情况，重点关注 NCCL / collective / distributed kernel，例如 `all_reduce`、`all_gather`、`reduce_scatter`、`all_to_all`、`broadcast`、`send/recv`、`barrier` 等。

说明：

- `--model` 仍然表示模型配置文件后缀，例如 `--model DeepSeek-V4-Flash` 会读取 `config.yaml.DeepSeek-V4-Flash`。
- 这是 SGLang native 单侧 profile，不分析 FlagGems 替换效果，也不做 CUDA/GEMS 对比。
- 采集方式采用 SGLang offline throughput profiling，更接近当前 vLLM `--torch` 的内置引擎 profiling 模式。
- 原有 vLLM 脚本不需要改动；SGLang 使用新增的 `sglang-*` 脚本。

### 7.1 可选 SGLang 配置

模型配置仍复用现有字段：

- `model.path`
- `model.name`
- `model.tensor_parallel_size`
- `model.tokenizer_path`
- `serve.trust_remote_code`
- `benchmark.num_runs`
- `benchmark.scenarios`
- `paths.results`
- `paths.reports`

如需传递 SGLang 专用参数，可在对应 `config.yaml.<模型名>` 中增加可选字段：

```yaml
sglang:
  extra_args: "--mem-fraction-static 0.82 --attention-backend flashinfer"
```

如果没有配置 `sglang.extra_args`，runner 会回退使用 `serve.extra_args`。如果 vLLM 和 SGLang 参数不兼容，建议显式配置 `sglang.extra_args`。

### 7.2 采集 SGLang Torch Profiler

DeepSeek 示例：

```bash
./scripts/sglang-auto-workflow.sh --model DeepSeek-V4-Flash --device 0 --torch --scenario optimized
```

Qwen 示例：

```bash
./scripts/sglang-auto-workflow.sh --model Qwen3.5-397B-A17B --device 0 --torch --scenario optimized
```

覆盖运行轮数：

```bash
./scripts/sglang-auto-workflow.sh --model DeepSeek-V4-Flash --device 0 --torch --scenario optimized --runs 2
```

Dry run 只打印将执行的 SGLang 命令，不启动模型：

```bash
./scripts/sglang-auto-workflow.sh --model DeepSeek-V4-Flash --device 0 --torch --scenario optimized --dry-run
```

采集逻辑：

- 每个 scenario 会运行 `benchmark.num_runs` 轮。
- 最后一轮启用 profiler，前面的轮次用于预热。
- profiler 目录通过 `SGLANG_TORCH_PROFILER_DIR` 传给 SGLang。

原始输出默认保存到：

```bash
results/<model>/sglang-bench_<scenario>_torch_profile_log/sglang_bench_logs/
results/<model>/sglang-torch-raw/report-sglang/
```

### 7.3 生成分布式算子报告

分析 rank 0：

```bash
./scripts/sglang-auto-processing.sh --model DeepSeek-V4-Flash --workflow torch --rank 0
```

分析全部 rank：

```bash
./scripts/sglang-auto-processing.sh --model DeepSeek-V4-Flash --workflow torch --rank all
```

生成文件：

```bash
reports/<model>/sglang_perf_analysis_torch.md
reports/<model>/sglang_perf_analysis_torch.xlsx
```

报告包含：

- Profile 概览：trace 文件数、rank 列表、总 kernel 时间、分布式 kernel 时间与占比。
- 分布式算子总览：按 `all_reduce`、`all_gather`、`reduce_scatter` 等类型聚合调用次数和耗时。
- 按 Rank 对比：展示每类分布式算子在各 rank 上的耗时、调用次数和 `max/min` 不均衡比例。
- Top 分布式 Kernel：列出耗时最高的分布式 kernel、关联 op 和 source 信息。
- Top 全部 Kernel：保留整体 kernel 视角，便于判断通信与计算占比。

### 7.4 静态检查

代码变更后可先做静态检查：

```bash
bash -n scripts/sglang-auto-workflow.sh scripts/sglang-run-workflow.sh scripts/sglang-auto-processing.sh scripts/sglang-run-processing.sh
python3 -m py_compile scripts/tools/sglang_profile_runner.py scripts/tools/sglang_perf_analysis_torch.py
```

服务器具备 SGLang 环境后，再执行采集与分析命令。
