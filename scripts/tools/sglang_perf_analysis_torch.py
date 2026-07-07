#!/usr/bin/env python3
"""Analyze SGLang Torch profiler traces with distributed-op focus."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

try:
    from openpyxl import Workbook
except ImportError:  # pragma: no cover
    Workbook = None


class OpKind(str, Enum):
    DISTRIBUTED_ALL_REDUCE = "distributed_all_reduce"
    DISTRIBUTED_ALL_GATHER = "distributed_all_gather"
    DISTRIBUTED_REDUCE_SCATTER = "distributed_reduce_scatter"
    DISTRIBUTED_ALL_TO_ALL = "distributed_all_to_all"
    DISTRIBUTED_BROADCAST = "distributed_broadcast"
    DISTRIBUTED_P2P = "distributed_p2p"
    DISTRIBUTED_NCCL_OTHER = "distributed_nccl_other"
    ATTENTION = "attention"
    MOE = "moe"
    GEMM = "gemm"
    NORM = "norm"
    ACTIVATION = "activation"
    KV_CACHE = "kv_cache"
    INDEXING = "indexing"
    MAMBA_OR_LINEAR_ATTENTION = "mamba_or_linear_attention"
    NON_DISTRIBUTED = "non_distributed"

    # Legacy enum member names; values intentionally use the new report labels.
    ALL_REDUCE = "distributed_all_reduce"
    ALL_GATHER = "distributed_all_gather"
    REDUCE_SCATTER = "distributed_reduce_scatter"
    ALL_TO_ALL = "distributed_all_to_all"
    BROADCAST = "distributed_broadcast"
    SEND_RECV = "distributed_p2p"
    OTHER_DISTRIBUTED = "distributed_nccl_other"


# Backward-compatible export name used by existing tests/callers.
DistributedOpKind = OpKind


DISTRIBUTED_PREFIX = "distributed_"

FLASH_ATTENTION_EXCLUSIONS = (
    "collectivemainloop",
    "collectiveepilogue",
    "cutlass::device_kernel<flash::",
    "flashattnfwd",
    "flash::prepare_varlen",
)

REAL_COMMUNICATION_KEYWORDS = (
    "nccl",
    "all_reduce",
    "allreduce",
    "all_gather",
    "allgather",
    "reduce_scatter",
    "reducescatter",
    "all_to_all",
    "alltoall",
    "broadcast",
    "send",
    "recv",
    "processgroup",
    "c10d",
    "record_param_comms",
    "custom_ar",
    "symm_mem all_reduce",
    "multimem_all_reduce",
    "vllm::all_reduce",
    "_c_custom_ar::all_reduce",
)


@dataclass
class Aggregate:
    calls: int = 0
    total_us: float = 0.0

    def add(self, dur_us: float, calls: int = 1) -> None:
        self.calls += calls
        self.total_us += dur_us

    @property
    def avg_us(self) -> float:
        return self.total_us / self.calls if self.calls else 0.0


@dataclass
class KernelRecord:
    rank: int
    name: str
    op_name: str
    source_file: str
    dur_us: float
    kind: OpKind


@dataclass
class RankStats:
    rank: int
    total_kernel_us: float = 0.0
    distributed: DefaultDict[OpKind, Aggregate] = field(
        default_factory=lambda: defaultdict(Aggregate)
    )
    kernels: List[KernelRecord] = field(default_factory=list)


@dataclass
class ProfileStats:
    rank_stats: Dict[int, RankStats]
    trace_files: List[Path]
    total_kernel_us: float
    profiler_txt_total_us: float

    @property
    def distributed_total_us(self) -> float:
        return sum(
            agg.total_us
            for rank_stat in self.rank_stats.values()
            for agg in rank_stat.distributed.values()
        )


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_tool_config() -> Dict[str, Any]:
    cfg = get_project_root() / "scripts" / "tools" / "sglang_tool_config.yaml"
    if not cfg.exists():
        return {}
    return yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else get_project_root() / path


def get_default_paths() -> Tuple[Path, Path]:
    cfg = load_tool_config()
    paths = cfg.get("paths", {}) or {}
    model_name = paths.get("model_name", "default")
    torch_dir = paths.get("torch_output_dir", f"results/{model_name}/sglang-torch-raw")
    reports_dir = paths.get("reports_dir", f"reports/{model_name}")
    return resolve_path(torch_dir), resolve_path(reports_dir)


def normalize_rank_selector(value: str) -> str:
    text = str(value).strip().lower()
    if text == "all":
        return text
    if text.isdigit():
        return str(int(text))
    raise argparse.ArgumentTypeError("--rank 仅支持数字或 all")


def normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def extract_rank(path: Path) -> Optional[int]:
    name = path.name
    patterns = (
        r"(?:^|[^a-z])rank[_-]?(\d+)(?:\D|$)",
        r"profiler_out[_-](\d+)(?:\D|$)",
        r"(?:^|[^A-Z])TP[_-](\d+)(?:\D|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def rank_matches(path: Path, rank_selector: str) -> bool:
    if rank_selector == "all":
        return True
    rank = extract_rank(path)
    return rank is not None and str(rank) == rank_selector


def classify_distributed_op(
    kernel_name: str,
    op_name: str = "",
    source_file: str = "",
) -> Optional[OpKind]:
    kind = classify_op_kind(kernel_name, op_name, source_file)
    return kind if is_distributed_kind(kind) else None


def is_distributed_kind(kind: OpKind) -> bool:
    return kind.value.startswith(DISTRIBUTED_PREFIX)


def normalized_search_text(
    kernel_name: str,
    op_name: str = "",
    source_file: str = "",
) -> Tuple[str, str]:
    text = " ".join((kernel_name, op_name, source_file)).lower()
    compact = re.sub(r"[^a-z0-9]+", "", text)
    return text, compact


def has_keyword(text: str, compact: str, *needles: str) -> bool:
    return any(needle in text or needle.replace("_", "") in compact for needle in needles)


def is_flash_attention_internal(text: str, compact: str) -> bool:
    return any(needle in text or needle.replace("_", "") in compact for needle in FLASH_ATTENTION_EXCLUSIONS)


def classify_op_kind(
    kernel_name: str,
    op_name: str = "",
    source_file: str = "",
) -> OpKind:
    text, compact = normalized_search_text(kernel_name, op_name, source_file)

    if is_flash_attention_internal(text, compact):
        return OpKind.ATTENTION

    if has_keyword(text, compact, "all_reduce", "allreduce", "all-reduce"):
        return OpKind.DISTRIBUTED_ALL_REDUCE
    if has_keyword(text, compact, "reduce_scatter", "reducescatter", "reduce-scatter"):
        return OpKind.DISTRIBUTED_REDUCE_SCATTER
    if has_keyword(text, compact, "all_gather", "allgather", "all-gather"):
        return OpKind.DISTRIBUTED_ALL_GATHER
    if has_keyword(text, compact, "all_to_all", "alltoall", "all-to-all"):
        return OpKind.DISTRIBUTED_ALL_TO_ALL
    if has_keyword(text, compact, "broadcast", "bcast"):
        return OpKind.DISTRIBUTED_BROADCAST
    if has_keyword(text, compact, "sendrecv", "send_recv", "send-recv", "recvsend", "isend", "irecv"):
        return OpKind.DISTRIBUTED_P2P
    if re.search(r"(^|[^a-z])(send|recv)([^a-z]|$)", text):
        return OpKind.DISTRIBUTED_P2P
    if "nccl" in text:
        return OpKind.DISTRIBUTED_NCCL_OTHER
    if has_keyword(text, compact, "processgroup", "c10d", "record_param_comms", "custom_ar"):
        return OpKind.DISTRIBUTED_NCCL_OTHER

    if has_keyword(
        text,
        compact,
        "flash_attn",
        "flashattn",
        "attention",
        "attn",
        "paged_attention",
        "sparse_attn",
        "rotary",
    ):
        return OpKind.ATTENTION
    if has_keyword(text, compact, "moe", "expert", "topk"):
        return OpKind.MOE
    if has_keyword(text, compact, "gemm", "matmul", "mm_kernel", "cutlass", "cublas"):
        return OpKind.GEMM
    if has_keyword(text, compact, "rms_norm", "rmsnorm", "layer_norm", "layernorm", "l2norm", "norm"):
        return OpKind.NORM
    if has_keyword(text, compact, "silu", "gelu", "relu", "activation", "doactivation", "swiglu"):
        return OpKind.ACTIVATION
    if has_keyword(text, compact, "kv_cache", "kvcache", "cache_kernel", "reshape_and_cache", "concat_and_cache"):
        return OpKind.KV_CACHE
    if has_keyword(text, compact, "index", "gather", "scatter", "sort", "nonzero", "where"):
        return OpKind.INDEXING
    if has_keyword(text, compact, "mamba", "linear_attention", "linearattention", "gated_delta"):
        return OpKind.MAMBA_OR_LINEAR_ATTENTION
    return OpKind.NON_DISTRIBUTED


def detect_distributed_keywords(profile: "ProfileStats") -> List[str]:
    detected = set()
    for rank_stat in profile.rank_stats.values():
        for record in rank_stat.kernels:
            text, compact = normalized_search_text(record.name, record.op_name, record.source_file)
            if is_flash_attention_internal(text, compact):
                continue
            for keyword in REAL_COMMUNICATION_KEYWORDS:
                if keyword in text or keyword.replace("_", "") in compact:
                    detected.add(keyword)
    return sorted(detected)


def has_real_multigpu_communication(profile: "ProfileStats") -> bool:
    decisive = {
        "nccl",
        "all_reduce",
        "allreduce",
        "all_gather",
        "allgather",
        "reduce_scatter",
        "reducescatter",
        "all_to_all",
        "alltoall",
    }
    return any(keyword in decisive for keyword in detect_distributed_keywords(profile))


def classify_distributed_op_legacy(
    kernel_name: str,
    op_name: str = "",
    source_file: str = "",
) -> Optional[OpKind]:
    """Deprecated compatibility shim for old internal callers."""
    return classify_distributed_op(kernel_name, op_name, source_file)


def open_trace(path: Path) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def iter_trace_events(path: Path) -> Iterable[Dict[str, Any]]:
    with open_trace(path) as f:
        data = json.load(f)
    events = data.get("traceEvents", []) if isinstance(data, dict) else []
    for event in events:
        if isinstance(event, dict):
            yield event


def parse_self_cuda_total_us(txt_path: Path) -> float:
    text = txt_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Self CUDA time total:\s*([0-9.]+)\s*([num]?s)", text)
    if not match:
        return 0.0
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "s":
        return value * 1_000_000.0
    if unit == "ms":
        return value * 1_000.0
    if unit == "us":
        return value
    if unit == "ns":
        return value / 1_000.0
    return 0.0


def parse_trace_file(path: Path, rank: int) -> RankStats:
    cpu_op_by_external_id: Dict[str, str] = {}
    source_by_external_id: Dict[str, str] = {}
    rank_stats = RankStats(rank=rank)

    events = list(iter_trace_events(path))
    for event in events:
        if event.get("cat") not in {"cpu_op", "python_function"} or event.get("ph") != "X":
            continue
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        external_id = args.get("External id")
        if external_id is None:
            continue
        external_key = str(external_id)
        name = str(event.get("name", ""))
        if event.get("cat") == "cpu_op":
            cpu_op_by_external_id[external_key] = name
        elif ".py" in name:
            source_by_external_id[external_key] = name.split(":", 1)[0]

    for event in events:
        if event.get("cat") != "kernel" or event.get("ph") != "X":
            continue
        dur = event.get("dur", 0.0)
        if not isinstance(dur, (int, float)):
            continue
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        external_id = args.get("External id")
        external_key = str(external_id) if external_id is not None else ""
        kernel_name = normalize_name(str(event.get("name", "unknown")))
        op_name = cpu_op_by_external_id.get(external_key, "")
        source_file = source_by_external_id.get(external_key, "")
        kind = classify_op_kind(kernel_name, op_name, source_file)
        record = KernelRecord(
            rank=rank,
            name=kernel_name,
            op_name=op_name or "unknown",
            source_file=source_file,
            dur_us=float(dur),
            kind=kind,
        )
        rank_stats.total_kernel_us += float(dur)
        rank_stats.kernels.append(record)
        if is_distributed_kind(kind):
            rank_stats.distributed[kind].add(float(dur))

    return rank_stats


def merge_rank_stats(target: RankStats, source: RankStats) -> None:
    target.total_kernel_us += source.total_kernel_us
    target.kernels.extend(source.kernels)
    for kind, agg in source.distributed.items():
        target.distributed[kind].add(agg.total_us, agg.calls)


def parse_profile_dir_by_rank(report_dir: Path, rank_selector: str = "0") -> ProfileStats:
    trace_files = sorted(
        list(report_dir.glob("*.pt.trace.json"))
        + list(report_dir.glob("*.trace.json"))
        + list(report_dir.glob("*.pt.trace.json.gz"))
        + list(report_dir.glob("*.trace.json.gz"))
    )
    trace_files = [path for path in trace_files if rank_matches(path, rank_selector)]
    if not trace_files:
        raise SystemExit(f"Missing SGLang trace file in {report_dir} for rank={rank_selector}")

    rank_stats: Dict[int, RankStats] = {}
    for trace_file in trace_files:
        rank = extract_rank(trace_file)
        if rank is None:
            rank = 0 if rank_selector != "all" else len(rank_stats)
        parsed = parse_trace_file(trace_file, rank)
        if rank not in rank_stats:
            rank_stats[rank] = RankStats(rank=rank)
        merge_rank_stats(rank_stats[rank], parsed)

    txt_total_us = 0.0
    for txt_file in sorted(report_dir.glob("profiler_out_*.txt")):
        if rank_matches(txt_file, rank_selector):
            txt_total_us += parse_self_cuda_total_us(txt_file)

    total_kernel_us = sum(item.total_kernel_us for item in rank_stats.values())
    return ProfileStats(
        rank_stats=rank_stats,
        trace_files=trace_files,
        total_kernel_us=total_kernel_us,
        profiler_txt_total_us=txt_total_us,
    )


def fmt_ms(us: float) -> str:
    return f"{us / 1000.0:.3f}"


def fmt_us(us: float) -> str:
    return f"{us:.3f}"


def fmt_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def md_escape(text: str) -> str:
    return str(text).replace("|", "\\|")


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(out)


def aggregate_distributed(profile: ProfileStats) -> Dict[OpKind, Aggregate]:
    out: Dict[OpKind, Aggregate] = defaultdict(Aggregate)
    for rank_stat in profile.rank_stats.values():
        for kind, agg in rank_stat.distributed.items():
            out[kind].add(agg.total_us, agg.calls)
    return out


def aggregate_records(records: Iterable[KernelRecord]) -> List[Tuple[Tuple[str, str, str, str], Aggregate]]:
    out: Dict[Tuple[str, str, str, str], Aggregate] = defaultdict(Aggregate)
    for record in records:
        out[(record.kind.value, record.op_name, record.name, record.source_file)].add(record.dur_us)
    return sorted(out.items(), key=lambda item: item[1].total_us, reverse=True)


def build_markdown(profile: ProfileStats, rank_selector: str) -> Tuple[str, List[Dict[str, Any]]]:
    lines: List[str] = ["# SGLang Torch Profiler 分布式算子报告", ""]
    tables: List[Dict[str, Any]] = []
    distributed_total = profile.distributed_total_us
    total_ref = profile.total_kernel_us or profile.profiler_txt_total_us

    overview_rows = [
        ["rank selector", rank_selector],
        ["trace files", str(len(profile.trace_files))],
        ["ranks", ",".join(str(rank) for rank in sorted(profile.rank_stats))],
        ["total kernel time(ms)", fmt_ms(profile.total_kernel_us)],
        ["distributed kernel time(ms)", fmt_ms(distributed_total)],
        ["distributed pct", fmt_pct(distributed_total / total_ref if total_ref > 0 else 0.0)],
    ]
    lines.append("## Profile 概览")
    lines.append("")
    lines.append(md_table(["字段", "值"], overview_rows))
    lines.append("")
    tables.append({"sheet_name": "Overview", "headers": ["字段", "值"], "rows": overview_rows})

    detected_keywords = detect_distributed_keywords(profile)
    ranks = sorted(profile.rank_stats)
    sanity_note = ""
    if len(ranks) == 1 and not has_real_multigpu_communication(profile):
        sanity_note = "当前 profile 未检测到真实多卡通信，可能是 TP=1 smoke test"
    sanity_rows = [
        ["trace files", "\n".join(str(path) for path in profile.trace_files)],
        ["ranks", ",".join(str(rank) for rank in ranks)],
        ["detected distributed keywords", ", ".join(detected_keywords) if detected_keywords else "none"],
        ["sanity note", sanity_note or "ok"],
    ]
    lines.append("## Sanity Check")
    lines.append("")
    lines.append(md_table(["字段", "值"], sanity_rows))
    lines.append("")
    tables.append({"sheet_name": "SanityCheck", "headers": ["字段", "值"], "rows": sanity_rows})

    dist_agg = aggregate_distributed(profile)
    dist_rows: List[List[str]] = []
    for kind, agg in sorted(dist_agg.items(), key=lambda item: item[1].total_us, reverse=True):
        dist_rows.append([
            kind.value,
            str(agg.calls),
            fmt_ms(agg.total_us),
            fmt_us(agg.avg_us),
            fmt_pct(agg.total_us / total_ref if total_ref > 0 else 0.0),
        ])
    lines.append("## 分布式算子总览")
    lines.append("")
    lines.append(md_table(["op_kind", "调用次数", "总时间(ms)", "平均时间(us)", "占比"], dist_rows))
    lines.append("")
    tables.append({
        "sheet_name": "DistributedSummary",
        "headers": ["op_kind", "调用次数", "总时间(ms)", "平均时间(us)", "占比"],
        "rows": dist_rows,
    })

    rank_rows: List[List[str]] = []
    all_kinds = sorted({kind for rank in profile.rank_stats.values() for kind in rank.distributed}, key=lambda x: x.value)
    for kind in all_kinds:
        values = [
            profile.rank_stats[rank].distributed.get(kind, Aggregate()).total_us
            for rank in sorted(profile.rank_stats)
        ]
        calls = [
            profile.rank_stats[rank].distributed.get(kind, Aggregate()).calls
            for rank in sorted(profile.rank_stats)
        ]
        max_us = max(values) if values else 0.0
        min_us = min(values) if values else 0.0
        imbalance = (max_us / min_us) if min_us > 0 else (math.inf if max_us > 0 else 1.0)
        rank_rows.append([
            kind.value,
            ",".join(fmt_ms(value) for value in values),
            ",".join(str(call) for call in calls),
            fmt_ms(max_us),
            fmt_ms(min_us),
            "inf" if math.isinf(imbalance) else f"{imbalance:.3f}",
        ])
    lines.append("## 按 Rank 对比")
    lines.append("")
    lines.append(md_table(["op_kind", "各rank总时间(ms)", "各rank调用次数", "max(ms)", "min(ms)", "max/min"], rank_rows))
    lines.append("")
    tables.append({
        "sheet_name": "RankCompare",
        "headers": ["op_kind", "各rank总时间(ms)", "各rank调用次数", "max(ms)", "min(ms)", "max/min"],
        "rows": rank_rows,
    })

    all_records = [record for stat in profile.rank_stats.values() for record in stat.kernels]
    distributed_records = [record for record in all_records if is_distributed_kind(record.kind)]
    dist_kernel_rows: List[List[str]] = []
    for (kind, op_name, kernel_name, source_file), agg in aggregate_records(distributed_records)[:200]:
        dist_kernel_rows.append([
            kind,
            md_escape(op_name),
            md_escape(kernel_name),
            md_escape(source_file),
            str(agg.calls),
            fmt_ms(agg.total_us),
            fmt_us(agg.avg_us),
        ])
    lines.append("## Top 分布式 Kernel")
    lines.append("")
    lines.append(md_table(["op_kind", "op_name", "kernel_name", "source_file", "调用次数", "总时间(ms)", "平均时间(us)"], dist_kernel_rows))
    lines.append("")
    tables.append({
        "sheet_name": "TopDistributedKernel",
        "headers": ["op_kind", "op_name", "kernel_name", "source_file", "调用次数", "总时间(ms)", "平均时间(us)"],
        "rows": dist_kernel_rows,
    })

    all_kernel_rows: List[List[str]] = []
    for (kind, op_name, kernel_name, source_file), agg in aggregate_records(all_records)[:200]:
        all_kernel_rows.append([
            kind,
            md_escape(op_name),
            md_escape(kernel_name),
            md_escape(source_file),
            str(agg.calls),
            fmt_ms(agg.total_us),
            fmt_us(agg.avg_us),
        ])
    lines.append("## Top 全部 Kernel")
    lines.append("")
    lines.append(md_table(["op_kind", "op_name", "kernel_name", "source_file", "调用次数", "总时间(ms)", "平均时间(us)"], all_kernel_rows))
    lines.append("")
    tables.append({
        "sheet_name": "TopAllKernel",
        "headers": ["op_kind", "op_name", "kernel_name", "source_file", "调用次数", "总时间(ms)", "平均时间(us)"],
        "rows": all_kernel_rows,
    })

    return "\n".join(lines), tables


def write_excel(path: Path, tables: List[Dict[str, Any]]) -> None:
    if Workbook is None:
        print("[WARN] 缺少 openpyxl，跳过 Excel 输出")
        return
    wb = Workbook()
    wb.remove(wb.active)
    for table in tables:
        ws = wb.create_sheet(title=str(table["sheet_name"])[:31])
        ws.append(table["headers"])
        for row in table["rows"]:
            ws.append(list(row))
    wb.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SGLang torch profiler distributed operators")
    parser.add_argument("--torch_path", type=str, default=None, help="包含 report-sglang 的目录")
    parser.add_argument("--output_path", type=str, default=None, help="报告输出目录")
    parser.add_argument("--rank", type=normalize_rank_selector, default="0", help="数字 rank 或 all")
    parser.add_argument("--workers", type=int, default=None, help="保留参数，当前解析器顺序解析")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_torch_dir, default_output_dir = get_default_paths()
    torch_dir = resolve_path(args.torch_path) if args.torch_path else default_torch_dir
    output_dir = resolve_path(args.output_path) if args.output_path else default_output_dir
    report_dir = torch_dir / "report-sglang"
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = parse_profile_dir_by_rank(report_dir, rank_selector=args.rank)
    markdown, tables = build_markdown(profile, args.rank)
    md_path = output_dir / "sglang_perf_analysis_torch.md"
    xlsx_path = output_dir / "sglang_perf_analysis_torch.xlsx"
    md_path.write_text(markdown, encoding="utf-8")
    write_excel(xlsx_path, tables)
    print(f"[INFO] Markdown report: {md_path}")
    print(f"[INFO] Excel report: {xlsx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
