import gzip
import json
import sys
from pathlib import Path
from collections import defaultdict

def classify(name: str) -> str:
    n = name.lower()

    if "nccl" in n or "allreduce" in n or "all_reduce" in n or "allgather" in n or "all_gather" in n or "reduce_scatter" in n:
        return "comm/nccl"
    if "moe" in n or "expert" in n or "topk" in n or "router" in n:
        return "moe"
    if "attention" in n or "attn" in n or "flash" in n:
        return "attention"
    if "gemm" in n or "matmul" in n or "mm" in n or "linear" in n:
        return "gemm/linear"
    if "mamba" in n or "gdn" in n or "ssm" in n:
        return "linear_attn/gdn/mamba"
    if "layernorm" in n or "rmsnorm" in n or "norm" in n:
        return "norm"
    if "embedding" in n or "embed" in n:
        return "embedding"
    if "sampling" in n or "softmax" in n or "logits" in n:
        return "sampling/logits"
    if "memcpy" in n or "copy" in n:
        return "memcpy"
    if "triton" in n:
        return "triton_other"
    if "cuda" in n:
        return "cuda_other"
    return "other"

def load_trace(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)

def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_torch_trace_simple.py <trace_dir>")
        raise SystemExit(1)

    trace_dir = Path(sys.argv[1])
    files = sorted(trace_dir.glob("*.trace.json.gz")) + sorted(trace_dir.glob("*.trace.json"))

    if not files:
        print(f"No trace files found in {trace_dir}")
        raise SystemExit(1)

    total_by_rank = {}
    by_rank_cat = {}
    by_rank_top = {}

    for path in files:
        data = load_trace(path)
        events = data.get("traceEvents", [])

        cat_dur = defaultdict(float)
        name_dur = defaultdict(float)

        for ev in events:
            if ev.get("ph") != "X":
                continue

            dur = ev.get("dur", 0) or 0
            name = str(ev.get("name", "unknown"))

            # dur is usually microseconds in Chrome trace.
            c = classify(name)
            cat_dur[c] += dur
            name_dur[name] += dur

        total = sum(cat_dur.values())
        rank_name = path.name

        total_by_rank[rank_name] = total
        by_rank_cat[rank_name] = cat_dur
        by_rank_top[rank_name] = sorted(name_dur.items(), key=lambda x: x[1], reverse=True)[:30]

    for rank, total in total_by_rank.items():
        print("\n" + "=" * 100)
        print(rank)
        print(f"total duration in trace events: {total/1000:.3f} ms")

        print("\n[Category summary]")
        for cat, dur in sorted(by_rank_cat[rank].items(), key=lambda x: x[1], reverse=True):
            pct = 100 * dur / total if total > 0 else 0
            print(f"{cat:24s} {dur/1000:12.3f} ms  {pct:6.2f}%")

        print("\n[Top ops/kernels]")
        for name, dur in by_rank_top[rank]:
            pct = 100 * dur / total if total > 0 else 0
            short = name[:120].replace("\n", " ")
            print(f"{dur/1000:12.3f} ms  {pct:6.2f}%  {short}")

if __name__ == "__main__":
    main()
