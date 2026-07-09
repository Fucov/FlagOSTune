#!/usr/bin/env python3
import argparse
import gzip
import os
import re
import sys
import time
from collections import defaultdict

NAME_RE = re.compile(rb'"name"\s*:\s*"((?:\\.|[^"\\])*)"')
CAT_RE = re.compile(rb'"cat"\s*:\s*"((?:\\.|[^"\\])*)"')
DUR_RE = re.compile(rb'"dur"\s*:\s*([0-9]+(?:\.[0-9]+)?)')

def bdecode(x: bytes) -> str:
    return x.decode("utf-8", "replace")

def extract_str(obj: bytes, regex):
    m = regex.search(obj)
    if not m:
        return ""
    return bdecode(m.group(1))

def extract_dur(obj: bytes):
    m = DUR_RE.search(obj)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None

def classify_kernel(name: str, cat: str) -> str:
    s = (name + " " + cat).lower()

    if "nccl" in s or "allreduce" in s or "all_reduce" in s or "allgather" in s or "all_gather" in s or "reducescatter" in s or "reduce_scatter" in s:
        return "Communication / NCCL"
    if "flash" in s or "attention" in s or "attn" in s or "fmha" in s or "fa2" in s or "fa3" in s:
        return "Attention"
    if "moe" in s or "expert" in s or "grouped" in s or "deepgemm" in s:
        return "MoE / Expert"
    if "gemm" in s or "matmul" in s or "cublas" in s or "cutlass" in s or "wgmma" in s or "mma" in s:
        return "GEMM / Linear"
    if "rmsnorm" in s or "layernorm" in s or "norm" in s:
        return "Norm"
    if "rope" in s or "rotary" in s:
        return "RoPE / Position"
    if "softmax" in s or "topk" in s or "sampling" in s or "argmax" in s:
        return "Sampling / Softmax"
    if "memcpy" in s or "memset" in s or "copy" in s:
        return "Memcpy / Memset"
    if "triton" in s:
        return "Triton Other"
    return "Other GPU Kernel"

def event_objects_from_trace_gz(path: str, chunk_size: int = 8 * 1024 * 1024):
    """
    Stream Chrome trace JSON.gz and yield event objects inside traceEvents array.
    This avoids json.load() on huge 10GB+ decompressed traces.
    """
    marker = b'"traceEvents"'
    buf = b""
    in_events = False
    in_obj = False
    obj = bytearray()
    depth = 0
    in_str = False
    esc = False

    with gzip.open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            if not in_events:
                buf += chunk
                pos = buf.find(marker)
                if pos < 0:
                    if len(buf) > len(marker) + 1024:
                        buf = buf[-(len(marker) + 1024):]
                    continue

                arr = buf.find(b"[", pos)
                if arr < 0:
                    buf = buf[pos:]
                    continue

                data = buf[arr + 1:]
                buf = b""
                in_events = True
            else:
                data = chunk

            for c in data:
                if not in_obj:
                    if c == ord("{"):
                        in_obj = True
                        obj = bytearray()
                        obj.append(c)
                        depth = 1
                        in_str = False
                        esc = False
                    elif c == ord("]"):
                        return
                    continue

                obj.append(c)

                if in_str:
                    if esc:
                        esc = False
                    elif c == ord("\\"):
                        esc = True
                    elif c == ord('"'):
                        in_str = False
                    continue

                if c == ord('"'):
                    in_str = True
                elif c == ord("{"):
                    depth += 1
                elif c == ord("}"):
                    depth -= 1
                    if depth == 0:
                        yield bytes(obj)
                        in_obj = False
                        obj = bytearray()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=-1)
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--progress-every", type=int, default=200000)
    args = ap.parse_args()

    trace = args.trace
    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)

    cat_stat = defaultdict(lambda: [0, 0.0, 0.0])
    kernel_stat = defaultdict(lambda: [0, 0.0, 0.0, ""])
    total_gpu_dur = 0.0
    total_events = 0
    gpu_events = 0
    t0 = time.time()

    for obj in event_objects_from_trace_gz(trace):
        total_events += 1

        # Fast prefilter: no duration => skip.
        if b'"dur"' not in obj:
            continue

        name = extract_str(obj, NAME_RE)
        cat = extract_str(obj, CAT_RE)
        dur = extract_dur(obj)
        if dur is None:
            continue

        probe = (name + " " + cat).lower()

        # Keep CUDA/GPU/kernel-like events. Skip most CPU op noise.
        if not (
            "kernel" in probe
            or "cuda" in probe
            or "nccl" in probe
            or "triton" in probe
            or "cublas" in probe
            or "cutlass" in probe
            or "gemm" in probe
            or "matmul" in probe
            or "attention" in probe
            or "attn" in probe
            or "flash" in probe
            or "moe" in probe
            or "memcpy" in probe
            or "memset" in probe
        ):
            continue

        group = classify_kernel(name, cat)
        gpu_events += 1
        total_gpu_dur += dur

        cat_stat[group][0] += 1
        cat_stat[group][1] += dur
        cat_stat[group][2] = max(cat_stat[group][2], dur)

        ks = kernel_stat[name]
        ks[0] += 1
        ks[1] += dur
        ks[2] = max(ks[2], dur)
        ks[3] = group

        if total_events % args.progress_every == 0:
            elapsed = time.time() - t0
            print(
                f"[PROGRESS] events={total_events:,} gpu_events={gpu_events:,} "
                f"elapsed={elapsed:.1f}s total_gpu_dur_us={total_gpu_dur:.0f}",
                file=sys.stderr,
                flush=True,
            )

    elapsed = time.time() - t0

    lines = []
    lines.append(f"# Fast CUDA Kernel Summary")
    lines.append("")
    lines.append(f"- Trace: `{trace}`")
    lines.append(f"- Rank: `{args.rank}`")
    lines.append(f"- Parsed events: `{total_events:,}`")
    lines.append(f"- GPU/kernel-like events: `{gpu_events:,}`")
    lines.append(f"- Total GPU/kernel-like duration: `{total_gpu_dur / 1000.0:.3f} ms`")
    lines.append(f"- Parser elapsed: `{elapsed:.2f} s`")
    lines.append("")
    lines.append("## Category Breakdown")
    lines.append("")
    lines.append("| Category | Count | Total ms | Percent | Max us |")
    lines.append("|---|---:|---:|---:|---:|")

    for group, (cnt, dur, max_dur) in sorted(cat_stat.items(), key=lambda x: x[1][1], reverse=True):
        pct = dur / total_gpu_dur * 100.0 if total_gpu_dur > 0 else 0.0
        lines.append(f"| {group} | {cnt:,} | {dur / 1000.0:.3f} | {pct:.2f}% | {max_dur:.3f} |")

    lines.append("")
    lines.append(f"## Top {args.top_k} Kernels by Total Duration")
    lines.append("")
    lines.append("| Rank | Kernel | Group | Count | Total ms | Percent | Avg us | Max us |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|")

    top = sorted(kernel_stat.items(), key=lambda x: x[1][1], reverse=True)[:args.top_k]
    for i, (name, (cnt, dur, max_dur, group)) in enumerate(top, 1):
        pct = dur / total_gpu_dur * 100.0 if total_gpu_dur > 0 else 0.0
        avg = dur / cnt if cnt else 0.0
        safe_name = name.replace("|", "\\|")
        if len(safe_name) > 180:
            safe_name = safe_name[:177] + "..."
        lines.append(f"| {i} | `{safe_name}` | {group} | {cnt:,} | {dur / 1000.0:.3f} | {pct:.2f}% | {avg:.3f} | {max_dur:.3f} |")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[DONE] wrote {out}", file=sys.stderr)

if __name__ == "__main__":
    main()
