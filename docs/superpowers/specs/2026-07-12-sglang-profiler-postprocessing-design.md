# SGLang Torch Profiler 后处理统计修复设计

## 目标

修复 FlagOSTune SGLang Torch profiler 后处理脚本的统计口径、事件去重、算子与源码映射以及通信分类，使报告主表与 mentor 报告一致，同时如实展示映射证据与可信度。

本次只修改本地代码并执行静态测试。服务器 trace 的重新处理与数值验收由用户手动完成。

## 输入选择

- 默认只分析 rank 0。
- 只扫描 `report-sglang` 根目录，不递归读取 `old_partial_*`、`archive_*` 等归档目录。
- 同一 rank 存在多份 trace 时，按文件名中的采集时间戳排序；无法解析时间戳时使用文件 mtime，选择最新一份。
- `--rank all` 保持可用，但每个 rank 仍只选择各自最新的一份 trace。
- 报告列出实际选中的 trace，避免把历史采集误合并到当前报告。

## 数据模型与映射顺序

解析后的每个 GPU kernel 记录独立保存：rank、kernel name、op name、source file、source type、provider、op kind、communication type、confidence、timestamp、duration、process/thread、device/stream、external id 和 correlation id。

映射严格使用以下优先级：

1. kernel 事件自带的 profiler op/source/stack；
2. CPU parent、external id 和 CUDA runtime correlation；
3. `sglang_kernel_name_mapping.yaml` 的 kernel-name 映射；
4. `sglang_kernel_source_map.yaml` 的启发式 source map；
5. 低置信度 fallback。

`source_type` 是独立字段，不再通过 `source_file` 中的后缀反推。允许值为 `profiler_stack`、`correlation`、`kernel_name_mapping`、`source_map_high`、`source_map_medium`、`source_map_low`、`unknown`。

主表中的 `op_name` 和 `source_file` 不留空。未能精确确认时使用可搜索的 fallback 名称与候选源码范围，同时明确标记 `source_type=unknown`、`confidence=low` 和 `needs_source_check=true`。不得把 kernel mapping、source map 或 fallback 伪装为 profiler stack。结合 `/Users/ykw/Code/Pycharm/sglang` 源码核对主要 kernel family，尽量消除 fallback。

## 去重与可信度

只对选中的 trace 做事件级去重。稳定指纹由 rank、timestamp、duration、pid、tid、kernel name、correlation/external id、device 和 stream 组成；只有全部可用字段一致的 GPU kernel 才视为重复，避免误删循环中名称和 duration 相同但时间戳不同的合法调用。

同时记录：

- raw 与 dedup GPU kernel 事件数和 duration；
- filtered GPU kernel 事件数和 duration；
- raw 与 dedup communication kernel 事件数和 duration；
- filtered communication kernel 事件数和 duration。

去重后数据是所有聚合和百分比的唯一输入。raw/dedup duration 差异超过 5% 时报告打印 warning。报告注明 parser schema/version 变化或 trace 采集不同会使总时间与旧报告不可直接比较；不会为了匹配旧数值缩放 duration。

## Mentor 百分比口径

新增 `--pct-mode mentor`，并设为默认值。

`primary_allreduce` 只根据明确规则判断：SGLang 的 `sglang::outplace_all_reduce`、one-shot/two-shot custom-all-reduce kernel/provider，以及 vLLM 的 `_C_custom_ar::all_reduce`、`vllm::all_reduce`、`cross_device_reduce`。`record_param_comms` 和 `ncclDevKernel_*` 不属于 primary all-reduce。

- primary all-reduce 的 `pct` 分母为全部去重后 True GPU Kernel duration；
- 其它算子的 `pct` 分母为全部 duration 减 primary all-reduce duration；
- `overall_pct` 始终使用全部 duration；
- 每行输出 `pct_denom`，分类与分母选择完全解耦。

## 报告结构

主表名称为 `Mentor Style CUDA Kernel（按 op_name 聚合）`，位于 Top 10 源码核查表之前。按 op name 聚合，同一 op 的 kernel name 用 `<br>` 拼接，输出 Top 80 或所有 duration 大于 0.001 ms 的项。字段为：

`source_file | op_name | kernel_name | 调用次数 | 总时间(ms) | 平均时间(us) | pct | pct_denom | overall_pct | source_type | provider | op_kind`

报告另外包含：

- `明确通信算子拆解`：只含 SGLang Custom AllReduce、NCCL collective，以及注明不计入 True GPU Kernel 的 Gloo/control-plane event；
- `可能通信融合 / 待确认算子`：只在检测到 all-to-all、reduce-scatter、dispatch/combine、DeepEP 或 FlashInfer communication fusion 证据时列入；
- `Source 映射可信度`：按调用次数统计 profiler、correlation、kernel mapping、source map、missing source 和 unknown op 比例；
- unknown 比例超过 30% 时的 Top unresolved kernel、原因和待核查源码；
- 去重前后统计和 >5% warning；
- duration 是 kernel 时间而非通信 bytes 的说明。

MoE fused expert、top-k、align 和 local sum-reduce 默认为 compute/routing/local reduce，不进入明确通信表。只有实际出现 EP 通信证据时，相关通信事件才进入可能通信表。

## Profile Detail

workflow、runner 和元数据链路增加 `--profile-detail light|full_stack`：

- `light`：`with_stack=0`、`record_shapes=0`、`with_modules=0`；
- `full_stack`：`with_stack=1`、`record_shapes=1`、`with_modules=1`、`profile_memory=0`。

默认使用 light；长上下文场景不会自动开启 full stack。小场景 full-stack trace 用于验证 kernel-name mapping，再将验证后的映射应用到长上下文 light trace。

## 错误处理与兼容性

- mapping YAML 缺失或格式错误时提供明确诊断，不能静默把所有记录归为 profiler stack。
- residual duration 小于等于零时，`pct` 输出 0 并在报告中 warning，避免除零。
- cache schema 提升；旧 cache 自动失效，防止缺少 provenance/dedup 字段的缓存污染新报告。
- 现有 `--source-map`、`--rank all`、Markdown 和 Excel 输出继续可用。

## 测试与验收

采用 TDD，先覆盖以下失败用例，再实现：

- 根目录最新 rank-0 trace 选择及归档目录排除；
- 完全相同事件被去重、合法重复调用不被误删、GPU/communication filtered duration 正确；
- primary all-reduce、NCCL 和普通计算分别使用正确分母；
- mapping 优先级及 source type 不被伪装；
- 关键 SGLang kernel 映射后主表没有空 op/source；
- Mentor Style Top 表字段、顺序、Top 80 和位置；
- MoE compute 不进入明确通信表，没有 EP 证据时输出正确结论；
- `--profile-detail` 从 shell 入口透传到 runner 环境；
- cache schema 升级与可信度字段。

本地验证执行相关 Python 单元测试、完整测试集、`python -m py_compile` 和改动 shell 文件的 `bash -n`。服务器数值验收重点检查主 all-reduce、FlashInfer norm、NCCL AllGather 的 mentor-style 百分比以及 raw/dedup 时间。
