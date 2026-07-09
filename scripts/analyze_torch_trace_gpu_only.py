import gzip
import json
import sys
from pathlib import Path
from collections import defaultdict

def classify(name: str) -> str:
    n = name.lower()

    if "nccl" in n or "allreduce" in n or "all_reduce" in n or "allgather" in n or "all_gather" in n or "reducescatter" in n or "reduce_scatter" in n:
        return "comm/nccl"

    if "moe" in n or "expert" in n or "fused_experts" in n or "outplace_fused_experts" in n or "topk" in n or "router" in n:
        return "moe"

    if "mamba" in n or "gdn" in n or "ssm" in n or "selective_scan" in n:
        return "linear_attn/gdn/mamba"

    if "attention" in n or "attn" in n or "flash" in n or "paged" in n:
        return "attention"

    if "gemm" in n or "matmul" in n or "bmm" in n or "addmm" in n or "mm_" in n or "linear" in n or "cutlass" in n:
        return "gemm/linear"

    if "rmsnorm" in n or "layernorm" in n or "norm" in n:
        return "norm"

    if "embedding" in n or "embed" in n:
        return "embedding"

    if "softmax" in n or "sampling" in n or "logit" in n:
        return "sampling/logits"

    if "memcpy" in n or "memset" in n or "copy" in n:
        return "memcpy/memset"

    if "triton" in n:
        return "triton_other"

    return "other_kernel"

def load_trace(path: Path):
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)

def is_gpu_kernel_event(ev):
    # PyTorch profiler chrome trace 中 CUDA kernel 通常 cat/name/args 里会带 kernel/device/cuda 信息。
    cat = str(ev.get("cat", "")).lower()
    name = str(ev.get("name", "")).lower()
    args = ev.get("args", {}) or {}
    args_s = str(args).lower()

    if ev.get("ph") != "X":
        return False

    if "kernel" in cat:
        return True
    if "cuda_runtime" in cat or "cuda_driver" in cat:
        return False
    if "cpu_op" in cat or "python" in cat or "user_annotation" in cat:
        return False

    # 一些 trace 的 GPU kernel 事件 cat 不稳定，用名字兜底。
    gpu_keywords = [
        "nccl", "triton", "cutlass", "gemm", "attention", "attn",
        "moe", "fused", "kernel", "cuda", "memcpy", "memset",
        "mamba", "gdn", "rmsnorm", "layernorm"
    ]

    if any(k in name for k in gpu_keywords) and ("device" in args_s or "stream" in args_s or "cuda" in args_s):
        return True

    return False

def summarize(trace_dir: Path):
    files = sorted(trace_dir.glob("*.trace.json.gz")) + sorted(trace_dir.glob("*.trace.json"))
    if not files:
        raise SystemExit(f"No trace files found in {trace_dir}")

    for path in files:
        data = load_trace(path)
        events = data.get("traceEvents", [])

        cat_dur = defaultdict(float)
        name_dur = defaultdict(float)
        kernel_events = 0

        for ev in events:
            if not is_gpu_kernel_event(ev):
                continue
            dur = float(ev.get("dur", 0) or 0)
            if dur <= 0:
                continue
            name = str(ev.get("name", "unknown"))
            cat = classify(name)
            cat_dur[cat] += dur
            name_dur[name] += dur
            kernel_events += 1

        total = sum(cat_dur.values())

        print("\n" + "=" * 100)
        print(path.name)
        print(f"gpu kernel events: {kernel_events}")
        print(f"gpu kernel total duration: {total/1000:.3f} ms")

        if total <= 0:
            print("[WARN] No GPU kernel events detected. Need inspect trace schema.")
            continue

        print("\n[GPU kernel category summary]")
        for cat, dur in sorted(cat_dur.items(), key=lambda x: x[1], reverse=True):
            print(f"{cat:28s} {dur/1000:12.3f} ms  {100*dur/total:6.2f}%")

        print("\n[Top GPU kernels]")
        for name, dur in sorted(name_dur.items(), key=lambda x: x[1], reverse=True)[:40]:
            print(f"{dur/1000:12.3f} ms  {100*dur/total:6.2f}%  {name[:160]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python analyze_torch_trace_gpu_only.py <trace_dir>")
    summarize(Path(sys.argv[1]))
