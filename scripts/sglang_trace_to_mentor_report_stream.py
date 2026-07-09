import argparse
import gzip
import json
import re
from pathlib import Path
from collections import defaultdict

def pick_latest_prefix(trace_dir: Path):
    groups = defaultdict(set)
    for p in trace_dir.glob("*.trace.json.gz"):
        if "-TP-" not in p.name:
            continue
        prefix = p.name.split("-TP-")[0]
        rank = p.name.split("-TP-")[1].split(".")[0]
        groups[prefix].add(rank)

    complete = []
    for prefix, ranks in groups.items():
        if {"0", "1", "2", "3"}.issubset(ranks):
            complete.append(prefix)

    if not complete:
        raise RuntimeError(f"No complete TP0-TP3 trace group found in {trace_dir}")

    def key(x):
        try:
            return float(x)
        except Exception:
            return x

    return sorted(complete, key=key)[-1]

def find_rank_trace(trace_dir: Path, prefix: str, rank: int):
    p = trace_dir / f"{prefix}-TP-{rank}.trace.json.gz"
    if not p.exists():
        raise FileNotFoundError(p)
    return p

def is_gpu_kernel_event(ev):
    if ev.get("ph") != "X":
        return False

    cat = str(ev.get("cat", "")).lower()
    name = str(ev.get("name", "")).lower()
    args = str(ev.get("args", {}) or {}).lower()

    if "cpu_op" in cat or "python" in cat or "user_annotation" in cat:
        return False
    if "kernel" in cat:
        return True

    # 某些 PyTorch profiler trace 的 GPU kernel cat 不稳定，用名字兜底
    keywords = [
        "nccl", "all_reduce", "allreduce", "allgather", "all_gather",
        "triton", "cutlass", "gemm", "moe", "expert", "rmsnorm",
        "layernorm", "attention", "attn", "mamba", "gdn", "causal_conv",
        "memcpy", "memset", "softmax", "argmax", "topk", "store_kvcache",
        "deep_gemm", "nvjet"
    ]

    return any(k in name for k in keywords) and (
        "device" in args or "stream" in args or "cuda" in args or "kernel" in cat
    )

def canonical_op(name: str):
    n = name.lower()

    # 对齐 mentor 口径：只把 SGLang custom all_reduce 作为特殊 all_reduce 分母项。
    # 不要把 all_gather、multimem_all_reduce 混进这个特殊项。
    if (
        "all_reduce_one_shot" in n
        or "all_reduce_two_shot" in n
        or "cross_device_reduce" in n
        or "_c_custom_ar" in n
        or "custom_ar" in n
    ):
        return "sglang::custom_all_reduce"

    if "multimem_all_reduce" in n:
        return "symm_mem::multimem_all_reduce"

    if "allgather" in n or "all_gather" in n:
        return "record_param_comms/all_gather"

    # 注意：flashinfernorm 不是 attention
    if (
        "fused_add_rmsnorm" in n
        or "rmsnorm" in n
        or "layernorm" in n
        or "normkernel" in n
        or "flashinfernorm" in n
    ):
        return "rms_norm/fused_add_rmsnorm"

    if "fused_moe" in n or "moe_kernel" in n or "moe_forward" in n:
        return "moe_forward/fused_moe"

    if "topkgating" in n or "topk" in n or "router" in n or "gatingsoftmax" in n:
        return "moe_router/topk_softmax"

    if (
        "deep_gemm" in n
        or "gemm" in n
        or "nvjet" in n
        or "matmul" in n
        or "cublas" in n
        or "cutlass" in n and "flashattn" not in n.lower()
    ):
        return "gemm/linear"

    if "flash_attn" in n or "flashattn" in n or "attention" in n or "_attn" in n:
        return "attention"

    if (
        "mamba" in n
        or "gdn" in n
        or "causal_conv1d" in n
        or "gated_delta" in n
        or "recurrent_gated_delta" in n
        or "chunk_gated_delta" in n
    ):
        return "hybrid_linear_attention/gdn_mamba"

    if "store_kvcache" in n or "kv_cache" in n or "kvcache" in n:
        return "kv_cache"

    if "softmax" in n or "argmax" in n or "sampling" in n or "logit" in n or "exponential" in n:
        return "sampling/logits"

    if "memcpy" in n or "memset" in n or "copy" in n:
        return "memcpy/copy"

    if "triton" in n:
        return "triton_other"

    if "embedding" in n or "embed" in n:
        return "embedding"

    return "other"

def source_file_for_op(op: str):
    mapping = {
        "sglang::custom_all_reduce": "sglang/srt/distributed/custom_all_reduce",
        "symm_mem::multimem_all_reduce": "sglang/srt/distributed/symm_mem",
        "record_param_comms/all_gather": "torch/nccl",
        "rms_norm/fused_add_rmsnorm": "sglang/srt/layers/layernorm / flashinfer_norm",
        "moe_forward/fused_moe": "sglang/srt/layers/moe",
        "moe_router/topk_softmax": "sglang/srt/layers/moe/router",
        "gemm/linear": "sglang/srt/layers/linear / deep_gemm",
        "attention": "sglang/srt/layers/attention",
        "hybrid_linear_attention/gdn_mamba": "sglang/srt/layers/attention / mamba_gdn",
        "kv_cache": "sglang/srt/mem_cache",
        "sampling/logits": "sglang/srt/sampling",
        "memcpy/copy": "cuda/memory",
        "triton_other": "triton",
        "embedding": "sglang/srt/layers/vocab_parallel_embedding",
        "other": "",
    }
    return mapping.get(op, "")

def clean_kernel_name(name: str, limit=360):
    name = name.replace("|", "\\|").replace("\n", " ")
    if len(name) > limit:
        return name[:limit] + "..."
    return name

def iter_trace_events_stream(path: Path, chunk_size=1024 * 1024 * 8):
    decoder = json.JSONDecoder()
    buf = ""
    in_events = False

    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            buf += chunk

            if not in_events:
                idx = buf.find('"traceEvents"')
                if idx < 0:
                    # 保留尾部，防止 traceEvents 被 chunk 切开
                    buf = buf[-64:]
                    continue
                lb = buf.find("[", idx)
                if lb < 0:
                    buf = buf[idx:]
                    continue
                buf = buf[lb + 1 :]
                in_events = True

            while True:
                buf = buf.lstrip()
                if not buf:
                    break

                if buf[0] == ",":
                    buf = buf[1:]
                    continue

                if buf[0] == "]":
                    return

                try:
                    ev, end = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    # 当前 buffer 不够一个完整 JSON object，继续读
                    # 防止 buffer 无限增长过头，一般单个 event 不会太大
                    break

                yield ev
                buf = buf[end:]

def parse_bench_log(log_path: Path):
    if not log_path.exists():
        return {}

    text = log_path.read_text(errors="ignore")
    fields = [
        "Successful requests",
        "Benchmark duration (s)",
        "Total input tokens",
        "Total generated tokens",
        "Request throughput (req/s)",
        "Input token throughput (tok/s)",
        "Output token throughput (tok/s)",
        "Total token throughput (tok/s)",
    ]
    out = {}
    for f in fields:
        m = re.search(re.escape(f) + r":\s+([0-9.]+)", text)
        if m:
            out[f] = m.group(1)
    return out

def parse_trace(trace_path: Path):
    op_dur = defaultdict(float)      # us
    op_count = defaultdict(int)
    op_kernel_dur = defaultdict(lambda: defaultdict(float))
    kernel_event_count = 0

    for ev in iter_trace_events_stream(trace_path):
        if not is_gpu_kernel_event(ev):
            continue

        dur = float(ev.get("dur", 0) or 0)
        if dur <= 0:
            continue

        name = str(ev.get("name", "unknown"))
        op = canonical_op(name)

        op_dur[op] += dur
        op_count[op] += 1
        op_kernel_dur[op][name] += dur
        kernel_event_count += 1

    return op_dur, op_count, op_kernel_dur, kernel_event_count

def write_report(args, trace_path, prefix, bench1, bench2, op_dur, op_count, op_kernel_dur, kernel_event_count):
    total_us = sum(op_dur.values())
    ar_op = "sglang::custom_all_reduce"
    all_reduce_us = op_dur.get(ar_op, 0.0)
    non_ar_us = max(total_us - all_reduce_us, 1e-9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for op, dur in sorted(op_dur.items(), key=lambda x: x[1], reverse=True):
        count = op_count[op]
        avg_us = dur / count if count else 0.0

        if op == ar_op:
            pct = 100.0 * dur / total_us if total_us > 0 else 0.0
        else:
            pct = 100.0 * dur / non_ar_us

        top_kernels = sorted(op_kernel_dur[op].items(), key=lambda x: x[1], reverse=True)[: args.top_kernels_per_op]
        kernel_name = "<br>".join(clean_kernel_name(k) for k, _ in top_kernels)

        rows.append((source_file_for_op(op), op, kernel_name, count, dur / 1000.0, avg_us, pct))

    with out.open("w", encoding="utf-8") as w:
        w.write(f"# FlagOSTune Torch Profiling 之 SGLang Qwen3.6-35B-A3B-FP8 TP4\n\n")

        w.write("# 环境\n\n")
        w.write("单机四卡 H20，SGLang 0.5.14，torch 2.11.0+cu130\n\n")
        w.write("模型：`/data/models/Qwen3.6-35B-A3B-FP8`\n\n")
        w.write("并行：TP=4，DP=1，PP=1\n\n")
        w.write("说明：本报告为 SGLang 分布式推理 profiling，和 vLLM 报告结构对齐，但 kernel 名称与 source file 按 SGLang 路径归类。\n\n")

        w.write("# 参数\n\n")
        w.write("```text\n")
        w.write("--mem-fraction-static 0.8\n")
        w.write("--context-length 65536\n")
        w.write("--max-running-requests 2048\n")
        w.write("--chunked-prefill-size 16384\n")
        w.write("--max-prefill-tokens 16384\n")
        w.write("--disable-cuda-graph\n")
        w.write("--attention-backend triton\n")
        w.write("--sampling-backend pytorch\n")
        w.write("--tp-size 4\n")
        w.write("```\n\n")

        w.write("# 测试场景\n\n")
        w.write("p32768d1024 concurrency: 1\n\n")

        w.write("# Benchmark 结果\n\n")
        w.write("## run1: profile=False\n\n")
        if bench1:
            w.write("|metric|value|\n|---|---:|\n")
            for k, v in bench1.items():
                w.write(f"|{k}|{v}|\n")
            w.write("\n")
        else:
            w.write("未解析到 run1 benchmark 结果。\n\n")

        w.write("## run2: profile=True\n\n")
        if bench2:
            w.write("|metric|value|\n|---|---:|\n")
            for k, v in bench2.items():
                w.write(f"|{k}|{v}|\n")
            w.write("\n")
        else:
            w.write("未解析到 run2 benchmark 结果。\n\n")

        w.write("# 算子数据\n\n")
        w.write("1. 占比说明：`sglang::custom_all_reduce` 使用全部 kernel 总时间作为分母；其它算子使用排除 custom_all_reduce 后的剩余 kernel 总时间作为分母。\n")
        w.write("2. 基于 torch profiler TP-0 trace json 生成。\n")
        w.write("3. 本表只统计 GPU kernel event，不统计 Python / CPU op / profiler 自身事件。\n\n")

        w.write(f"trace prefix: `{prefix}`\n\n")
        w.write(f"rank trace: `{trace_path}`\n\n")
        w.write(f"gpu kernel events: {kernel_event_count}\n\n")
        w.write(f"total kernel time(ms): {total_us / 1000.0:.3f}\n\n")
        w.write(f"custom_all_reduce time(ms): {all_reduce_us / 1000.0:.3f}\n\n")
        w.write(f"non-custom_all_reduce time(ms): {non_ar_us / 1000.0:.3f}\n\n")

        w.write("## CUDA kernel（按总时间排序）\n\n")
        w.write("|source file|op_name|kernel_name|调用次数|总时间(ms)|平均时间(us)|占比|\n")
        w.write("|---|---|---|---:|---:|---:|---:|\n")

        for source, op, kernel_name, count, total_ms, avg_us, pct in rows:
            w.write(f"|{source}|{op}|{kernel_name}|{count}|{total_ms:.3f}|{avg_us:.3f}|{pct:.2f}%|\n")

        w.write("\n# 初步观察\n\n")
        w.write("- 本场景为长上下文 prefill + 长 decode：p32768d1024_c1。\n")
        w.write("- custom all_reduce 为 TP 分布式推理中的核心通信开销，应单独作为通信瓶颈观察。\n")
        w.write("- MoE、DeepGEMM/GEMM、GDN/Mamba、RMSNorm、KV cache 和 sampling/logits 等算子使用排除 custom all_reduce 后的剩余 kernel 时间作为分母，更便于观察非通信计算部分的占比。\n")
        w.write("- SGLang 和 vLLM 的 op_name/source file 不完全一致，不能直接逐行对齐，但可以对齐到通信、MoE、GEMM、Attention/GDN、Norm、KV cache、Sampling 等类别。\n")

    print(f"[OK] report written to {out}")
    print(f"[INFO] total kernel time(ms): {total_us / 1000.0:.3f}")
    print(f"[INFO] custom all_reduce time(ms): {all_reduce_us / 1000.0:.3f}")
    print(f"[INFO] non custom all_reduce time(ms): {non_ar_us / 1000.0:.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--scenario", default="p32768d1024_c1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-kernels-per-op", type=int, default=5)
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    log_dir = Path(args.log_dir)

    prefix = args.prefix or pick_latest_prefix(trace_dir)
    trace_path = find_rank_trace(trace_dir, prefix, args.rank)

    print(f"[INFO] selected prefix: {prefix}")
    print(f"[INFO] selected trace: {trace_path}")
    print(f"[INFO] trace size GB: {trace_path.stat().st_size / 1024**3:.3f}")

    run1_log = log_dir / f"{args.scenario}_run1.log"
    run2_log = log_dir / f"{args.scenario}_run2.log"

    bench1 = parse_bench_log(run1_log)
    bench2 = parse_bench_log(run2_log)

    op_dur, op_count, op_kernel_dur, kernel_event_count = parse_trace(trace_path)

    write_report(args, trace_path, prefix, bench1, bench2, op_dur, op_count, op_kernel_dur, kernel_event_count)

if __name__ == "__main__":
    main()
