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

SGLang Nsight Systems 使用独立入口，不会启用或修改已有 Torch Profiler
workflow。运行环境需要安装 `nsys`、`yq` v4、带 CUDA 的 PyTorch 和 SGLang。

建议先用 `--dry-run` 校验模型、TP 大小、场景和最终命令。dry-run 不要求本机
安装 `nsys` 或加载模型：

```bash
./scripts/sglang-nsys-workflow.sh \
  --model Qwen3.6-35B-A3B-FP8-TP4-P128D16 \
  --nsys \
  --dry-run

./scripts/sglang-nsys-workflow.sh \
  --model DeepSeek-V4-Flash-FP8-TP8-Profile-P2048D32C64 \
  --nsys \
  --dry-run
```

采集 Qwen TP4 和 DeepSeek TP8：

```bash
./scripts/sglang-nsys-workflow.sh \
  --model Qwen3.6-35B-A3B-FP8-TP4-P128D16 \
  --nsys \
  --nsys-output qwen-tp4-profile

./scripts/sglang-nsys-workflow.sh \
  --model DeepSeek-V4-Flash-FP8-TP8-Profile-P2048D32C64 \
  --nsys \
  --nsys-output deepseek-tp8-profile
```

裸文件名形式的 `--nsys-output` 会写入模型默认目录：

```text
results/Qwen3.6-35B-A3B-FP8-TP4-P128D16/nsys/qwen-tp4-profile.nsys-rep
results/DeepSeek-V4-Flash-FP8-TP8-Profile-P2048D32C64/nsys/deepseek-tp8-profile.nsys-rep
```

也可以传入相对或绝对路径。相对路径以仓库根目录为基准；可选的
`.nsys-rep` 后缀会自动移除后再交给 `nsys`。如果所选场景组包含多个场景，
脚本会给显式前缀追加场景名，每个场景生成一个独立报告。

解析报告：

```bash
python3 scripts/tools/parse_nsys.py \
  results/Qwen3.6-35B-A3B-FP8-TP4-P128D16/nsys/qwen-tp4-profile.nsys-rep \
  --top 20

python3 scripts/tools/parse_nsys.py \
  results/DeepSeek-V4-Flash-FP8-TP8-Profile-P2048D32C64/nsys/deepseek-tp8-profile.nsys-rep \
  --top 20
```

解析器分别输出 Top CUDA Kernels（含 kernel time percentage）、NVTX Range
Summary 和 CUDA API Summary。可用 `--nsys /path/to/nsys` 指定非默认的
Nsight Systems 可执行文件。

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
