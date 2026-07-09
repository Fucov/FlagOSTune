import gzip
import json
import sys
from pathlib import Path
from collections import defaultdict

def load_trace(path: Path):
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)

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

    keywords = [
        "nccl", "all_reduce", "allreduce", "allgather", "all_gather",
        "triton", "cutlass", "gemm", "moe", "expert", "rmsnorm",
        "layernorm", "attention", "attn", "mamba", "gdn", "memcpy",
        "memset", "softmax", "argmax", "topk", "store_kvcache"
    ]

    return any(k in name for k in keywords) and (
        "device" in args or "stream" in args or "cuda" in args or "kernel" in cat
    )

def canonical_op(name: str):
    n = name.lower()

    # all_reduce 必须单独识别，mentor 口径对它使用 total kernel time 分母
    if "all_reduce" in n or "allreduce" in n or "all_reduce_one_shot" in n or "all_reduce_two_shot" in n or "multimem_all_reduce" in n:
        return "all_reduce"

    if "allgather" in n or "all_gather" in n:
        return "all_gather"

    # 注意：flashinfernorm 不是 attention
    if "fused_add_rmsnorm" in n or "rmsnorm" in n or "layernorm" in n or "normkernel" in n:
        return "rms_norm/fused_add_rmsnorm"

    if "fused_moe" in n or "moe" in n or "expert" in n:
        return "moe"

    if "topkgating" in n or "topk" in n or "router" in n:
        return "router/topk"

    if "deep_gemm" in n or "gemm" in n or "nvjet" in n or "matmul" in n or "mm" in n or "cutlass" in n:
        return "gemm/linear"

    if "flash_attn" in n or "attention" in n or "attn" in n:
        return "attention"

    if "mamba" in n or "gdn" in n or "causal_conv1d" in n or "gated_delta" in n:
        return "hybrid_linear_attn/mamba_gdn"

    if "store_kvcache" in n or "kv_cache" in n or "kvcache" in n:
        return "kv_cache"

    if "softmax" in n or "argmax" in n or "sampling" in n or "logit" in n:
        return "sampling/logits"

    if "memcpy" in n or "memset" in n or "copy" in n:
        return "memcpy/memset"

    if "triton" in n:
        return "triton_other"

    return "other"

def short_source(op_name: str):
    mapping = {
        "all_reduce": "sglang/distributed/custom_all_reduce",
        "all_gather": "torch/nccl",
        "rms_norm/fused_add_rmsnorm": "sglang/flashinfer_norm",
        "moe": "sglang/moe",
        "router/topk": "sglang/moe/router",
        "gemm/linear": "sglang/deep_gemm_or_cublas",
        "attention": "sglang/attention",
        "hybrid_linear_attn/mamba_gdn": "sglang/hybrid_linear_attention",
        "kv_cache": "sglang/kv_cache",
        "sampling/logits": "sglang/sampling",
        "memcpy/memset": "cuda/memory",
        "triton_other": "triton",
        "other": "unknown",
    }
    return mapping.get(op_name, "unknown")

def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python make_cuda_kernel_table_rank0.py <trace_dir_or_rank0_trace>")

    p = Path(sys.argv[1])
    if p.is_dir():
        candidates = sorted(p.glob("*TP-0.trace.json.gz")) + sorted(p.glob("*TP-0.trace.json"))
        if not candidates:
            raise SystemExit(f"No TP-0 trace found in {p}")
        trace_path = candidates[-1]
    else:
        trace_path = p

    data = load_trace(trace_path)
    events = data.get("traceEvents", [])

    # op_name -> total dur us, count, kernel names
    op_dur = defaultdict(float)
    op_count = defaultdict(int)
    op_kernels = defaultdict(lambda: defaultdict(float))

    for ev in events:
        if not is_gpu_kernel_event(ev):
            continue

        dur = float(ev.get("dur", 0) or 0)
        if dur <= 0:
            continue

        name = str(ev.get("name", "unknown"))
        op = canonical_op(name)

        op_dur[op] += dur
        op_count[op] += 1
        op_kernels[op][name] += dur

    total_us = sum(op_dur.values())
    all_reduce_us = op_dur.get("all_reduce", 0.0)
    non_ar_us = max(total_us - all_reduce_us, 1e-9)

    print(f"# CUDA kernel（按总时间排序）")
    print()
    print(f"source trace: {trace_path}")
    print(f"total kernel time(ms): {total_us / 1000:.3f}")
    print(f"all_reduce time(ms): {all_reduce_us / 1000:.3f}")
    print(f"non-all_reduce time(ms): {non_ar_us / 1000:.3f}")
    print()
    print("占比说明：all_reduce 使用全部 kernel 总时间作为分母；其它算子使用排除 all_reduce 后的剩余 kernel 总时间作为分母。")
    print()
    print("|source file|op_name|kernel_name|调用次数|总时间(ms)|平均时间(us)|占比|")
    print("|---|---|---|---:|---:|---:|---:|")

    for op, dur in sorted(op_dur.items(), key=lambda x: x[1], reverse=True):
        count = op_count[op]
        avg_us = dur / count if count else 0.0

        if op == "all_reduce":
            pct = 100.0 * dur / total_us if total_us > 0 else 0.0
        else:
            pct = 100.0 * dur / non_ar_us

        top_kernel = max(op_kernels[op].items(), key=lambda x: x[1])[0]
        top_kernel = top_kernel.replace("|", "\\|")
        if len(top_kernel) > 260:
            top_kernel = top_kernel[:260] + "..."

        print(f"|{short_source(op)}|{op}|{top_kernel}|{count}|{dur/1000:.3f}|{avg_us:.3f}|{pct:.2f}%|")

if __name__ == "__main__":
    main()
