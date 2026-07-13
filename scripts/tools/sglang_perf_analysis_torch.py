#!/usr/bin/env python3
"""Analyze SGLang Torch profiler traces with distributed-op focus."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    from openpyxl import Workbook
except ImportError:  # pragma: no cover
    Workbook = None

try:
    from .sglang_comm_report_formatter import (
        build_focus_report_sections,
        load_kernel_mappings,
    )
except ImportError:  # Direct script execution.
    from sglang_comm_report_formatter import (  # type: ignore
        build_focus_report_sections,
        load_kernel_mappings,
    )


class OpKind(str, Enum):
    DISTRIBUTED_ALL_REDUCE = "distributed_all_reduce"
    DISTRIBUTED_ALL_GATHER = "distributed_all_gather"
    DISTRIBUTED_REDUCE_SCATTER = "distributed_reduce_scatter"
    DISTRIBUTED_ALL_TO_ALL = "distributed_all_to_all"
    DISTRIBUTED_BROADCAST = "distributed_broadcast"
    DISTRIBUTED_P2P = "distributed_p2p"
    DISTRIBUTED_OTHER = "distributed_other"
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
    OTHER_DISTRIBUTED = "distributed_other"


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
    op_name: str = ""
    source_file: str = ""
    source_type: str = "unknown"
    provider: str = "Unknown"
    op_kind: str = ""
    communication_type: str = "unknown"
    confidence: str = "low"
    needs_source_check: bool = False

    def add(self, dur_us: float, calls: int = 1, **metadata: Any) -> None:
        self.calls += calls
        self.total_us += dur_us
        for name, value in metadata.items():
            if hasattr(self, name) and value not in (None, ""):
                setattr(self, name, value)

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
    raw_gpu_kernel_us: float = 0.0
    duplicate_gpu_kernel_us_filtered: float = 0.0
    raw_comm_kernel_us: float = 0.0
    dedup_comm_kernel_us: float = 0.0
    duplicate_comm_event_filtered_us: float = 0.0
    parsed_events: int = 0
    raw_gpu_kernel_events: int = 0
    gpu_kernel_events: int = 0
    duplicate_gpu_kernel_events_filtered: int = 0
    raw_comm_kernel_events: int = 0
    dedup_comm_kernel_events: int = 0
    profiler_events: int = 0
    cpu_op_events: int = 0
    cuda_runtime_events: int = 0
    distributed_events: int = 0
    duplicate_comm_event_filtered_count: int = 0
    distributed: DefaultDict[OpKind, Aggregate] = field(
        default_factory=lambda: defaultdict(Aggregate)
    )
    kernel_aggs: DefaultDict[Tuple[str, str, str, str], Aggregate] = field(
        default_factory=lambda: defaultdict(Aggregate)
    )
    event_hotspots: DefaultDict[Tuple[str, str, str], Aggregate] = field(
        default_factory=lambda: defaultdict(Aggregate)
    )
    kernels: List[KernelRecord] = field(default_factory=list)


@dataclass
class ProfileStats:
    rank_stats: Dict[int, RankStats]
    trace_files: List[Path]
    total_kernel_us: float
    profiler_txt_total_us: float
    cache_dir: Optional[Path] = None

    @property
    def distributed_total_us(self) -> float:
        return sum(
            agg.total_us
            for rank_stat in self.rank_stats.values()
            for agg in rank_stat.distributed.values()
        )


CACHE_SCHEMA_VERSION = 5


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_tool_config() -> Dict[str, Any]:
    cfg = get_project_root() / "scripts" / "tools" / "sglang_tool_config.yaml"
    if not cfg.exists() or yaml is None:
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
        return OpKind.DISTRIBUTED_OTHER
    if has_keyword(text, compact, "processgroup", "c10d", "record_param_comms", "custom_ar"):
        return OpKind.DISTRIBUTED_OTHER

    if has_keyword(text, compact, "rms_norm", "rmsnorm", "layer_norm", "layernorm", "l2norm", "fused_add_rmsnorm", "normkernel", "norm"):
        return OpKind.NORM
    if has_keyword(
        text,
        compact,
        "hybrid_linear_attn",
        "linear_attention",
        "linearattention",
        "gdn_backend",
        "gated_delta",
        "causal_conv1d",
        "fused_recurrent",
        "mamba",
    ):
        return OpKind.MAMBA_OR_LINEAR_ATTENTION
    if is_flash_attention_internal(text, compact):
        return OpKind.ATTENTION
    if has_keyword(
        text,
        compact,
        "flash_attn",
        "flashattn",
        "flashinfer",
        "attention",
        "attn",
        "radix_attention",
        "paged_attention",
        "sparse_attn",
        "rotary",
        "qkv",
        "kv_splits",
    ):
        return OpKind.ATTENTION
    if has_keyword(text, compact, "moe", "expert", "topk"):
        return OpKind.MOE
    if has_keyword(text, compact, "gemm", "matmul", "mm_kernel", "cutlass", "cublas"):
        return OpKind.GEMM
    if has_keyword(text, compact, "silu", "gelu", "relu", "activation", "doactivation", "swiglu"):
        return OpKind.ACTIVATION
    if has_keyword(text, compact, "kv_cache", "kvcache", "cache_kernel", "reshape_and_cache", "concat_and_cache"):
        return OpKind.KV_CACHE
    if has_keyword(text, compact, "index", "gather", "scatter", "sort", "nonzero", "where"):
        return OpKind.INDEXING
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
        for kind, op_name, kernel_name, source_file in rank_stat.kernel_aggs:
            text, compact = normalized_search_text(kernel_name, op_name, source_file)
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
        return gzip.open(path, "rb")
    return path.open("rb")


def iter_trace_event_objects(path: Path, chunk_size: int = 8 * 1024 * 1024) -> Iterator[bytes]:
    """Stream Chrome trace event JSON objects from traceEvents without loading the file."""
    marker = b'"traceEvents"'
    buf = b""
    in_events = False
    in_obj = False
    obj = bytearray()
    depth = 0
    in_str = False
    esc = False

    with open_trace(path) as f:
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


def iter_trace_events(path: Path) -> Iterable[Dict[str, Any]]:
    for obj in iter_trace_event_objects(path):
        try:
            event = json.loads(obj)
        except json.JSONDecodeError:
            continue
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


def trace_metadata(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def event_args(event: Dict[str, Any]) -> Dict[str, Any]:
    args = event.get("args")
    return args if isinstance(args, dict) else {}


def get_external_key(event: Dict[str, Any]) -> str:
    external_id = event_args(event).get("External id")
    return str(external_id) if external_id is not None else ""


def get_correlation_key(event: Dict[str, Any]) -> str:
    args = event_args(event)
    for key in ("correlation", "Correlation ID", "correlation id", "Correlation Id"):
        value = args.get(key)
        if value is not None:
            return str(value)
    return ""


def gpu_event_fingerprint(event: Dict[str, Any], rank: int) -> Tuple[str, ...]:
    """Return a conservative identity for exact duplicate GPU trace events."""
    args = event_args(event)

    def value(*names: str) -> str:
        for name in names:
            if name in args:
                return str(args[name])
        return ""

    return (
        str(rank),
        str(event.get("ts", "")),
        str(event.get("dur", "")),
        str(event.get("pid", "")),
        str(event.get("tid", "")),
        normalize_name(str(event.get("name", "unknown"))),
        get_correlation_key(event),
        get_external_key(event),
        value("device", "Device", "device id", "Device Id"),
        value("stream", "Stream", "stream id", "Stream Id"),
    )


def extract_source_from_args(args: Dict[str, Any]) -> str:
    for key in ("Source Location", "source", "Source", "file", "File", "filename", "Filename"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return value.split("\n", 1)[0]
    for key in ("Stack", "stack", "Call stack", "Call Stack", "Python stack"):
        value = args.get(key)
        if isinstance(value, str):
            for line in value.splitlines():
                if ".py" in line:
                    return line.strip()
        elif isinstance(value, list):
            for item in value:
                text = str(item)
                if ".py" in text:
                    return text.strip()
    return ""


def is_comm_text(*parts: str) -> bool:
    text, compact = normalized_search_text(" ".join(parts))
    return has_keyword(
        text,
        compact,
        "all_reduce",
        "allreduce",
        "all_gather",
        "allgather",
        "reduce_scatter",
        "reducescatter",
        "all_to_all",
        "alltoall",
        "broadcast",
        "barrier",
        "send",
        "recv",
        "nccl",
        "collective",
        "custom_all_reduce",
        "outplace_all_reduce",
    )


def load_source_map(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        elif yaml is not None:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            data = []
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("mappings", [])
    return [item for item in data if isinstance(item, dict)]


def match_kernel_mapping(
    mappings: Sequence[Dict[str, Any]], kernel_name: str
) -> Optional[Dict[str, Any]]:
    for item in mappings:
        pattern = str(item.get("pattern") or "")
        if not pattern:
            continue
        try:
            matched = re.search(pattern, kernel_name, flags=re.IGNORECASE) is not None
        except re.error:
            matched = pattern.lower() in kernel_name.lower()
        if matched:
            return item
    return None


def apply_source_map(
    source_map: List[Dict[str, Any]],
    *,
    kind: OpKind,
    op_name: str,
    kernel_name: str,
    source_file: str,
) -> Tuple[OpKind, str, str, str]:
    haystack = {
        "kernel_name": kernel_name,
        "op_name": op_name,
        "source_file": source_file,
        "all": " ".join((kernel_name, op_name, source_file)),
    }
    for item in source_map:
        pattern = str(item.get("pattern", ""))
        if not pattern:
            continue
        match_field = str(item.get("match_field", "all"))
        text = haystack.get(match_field, haystack["all"])
        try:
            matched = re.search(pattern, text, flags=re.IGNORECASE) is not None
        except re.error:
            matched = pattern.lower() in text.lower()
        if not matched:
            continue
        mapped_kind = kind
        kind_value = item.get("op_kind")
        if kind_value:
            try:
                mapped_kind = OpKind(str(kind_value))
            except ValueError:
                mapped_kind = kind
        mapped_op = str(item.get("op_name_override") or op_name or "unknown")
        mapped_source = source_file
        source_type = "unknown"
        if not mapped_source and item.get("source_file_guess"):
            confidence = str(item.get("confidence", "medium"))
            mapped_source = str(item.get("source_file_guess"))
            source_type = f"source_map_{confidence}"
        return mapped_kind, mapped_op, mapped_source, source_type
    return kind, op_name, source_file, "unknown"


def cache_paths(output_dir: Optional[Path], rank: int) -> Tuple[Optional[Path], Optional[Path]]:
    if output_dir is None:
        return None, None
    cache_dir = output_dir / "cache"
    return cache_dir / f"rank{rank}_kernel_agg.json", cache_dir / f"rank{rank}_event_hotspot_agg.json"


def summary_cache_path(output_dir: Optional[Path], rank: int) -> Optional[Path]:
    return (output_dir / "cache" / f"rank{rank}_trace_summary.json") if output_dir else None


def serialize_aggregate_map(data: Dict[Tuple[str, ...], Aggregate]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, agg in data.items():
        rows.append({
            "key": list(key),
            "calls": agg.calls,
            "total_us": agg.total_us,
            "metadata": {
                "op_name": agg.op_name,
                "source_file": agg.source_file,
                "source_type": agg.source_type,
                "provider": agg.provider,
                "op_kind": agg.op_kind,
                "communication_type": agg.communication_type,
                "confidence": agg.confidence,
                "needs_source_check": agg.needs_source_check,
            },
        })
    return rows


def load_aggregate_map(rows: Iterable[Dict[str, Any]]) -> DefaultDict[Tuple[str, ...], Aggregate]:
    out: DefaultDict[Tuple[str, ...], Aggregate] = defaultdict(Aggregate)
    for row in rows:
        key = tuple(str(item) for item in row.get("key", []))
        if not key:
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        out[key].add(float(row.get("total_us", 0.0)), int(row.get("calls", 0)), **metadata)
    return out


def load_rank_stats_from_cache(path: Path, rank: int, kernel_cache: Path, event_cache: Path) -> Optional[RankStats]:
    if not kernel_cache.exists() or not event_cache.exists():
        return None
    try:
        kernel_payload = json.loads(kernel_cache.read_text(encoding="utf-8"))
        event_payload = json.loads(event_cache.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if kernel_payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    meta = trace_metadata(path)
    if kernel_payload.get("trace") != meta or event_payload.get("trace") != meta:
        return None

    stats = RankStats(rank=rank)
    stats.total_kernel_us = float(kernel_payload.get("total_kernel_us", 0.0))
    stats.raw_gpu_kernel_us = float(kernel_payload.get("raw_gpu_kernel_us", stats.total_kernel_us))
    stats.duplicate_gpu_kernel_us_filtered = float(kernel_payload.get("duplicate_gpu_kernel_us_filtered", 0.0))
    stats.raw_comm_kernel_us = float(kernel_payload.get("raw_comm_kernel_us", 0.0))
    stats.dedup_comm_kernel_us = float(kernel_payload.get("dedup_comm_kernel_us", 0.0))
    stats.duplicate_comm_event_filtered_us = float(kernel_payload.get("duplicate_comm_event_filtered_us", 0.0))
    stats.parsed_events = int(kernel_payload.get("parsed_events", 0))
    stats.gpu_kernel_events = int(kernel_payload.get("gpu_kernel_events", 0))
    stats.raw_gpu_kernel_events = int(kernel_payload.get("raw_gpu_kernel_events", stats.gpu_kernel_events))
    stats.duplicate_gpu_kernel_events_filtered = int(kernel_payload.get("duplicate_gpu_kernel_events_filtered", 0))
    stats.raw_comm_kernel_events = int(kernel_payload.get("raw_comm_kernel_events", 0))
    stats.dedup_comm_kernel_events = int(kernel_payload.get("dedup_comm_kernel_events", 0))
    stats.profiler_events = int(kernel_payload.get("profiler_events", 0))
    stats.cpu_op_events = int(kernel_payload.get("cpu_op_events", 0))
    stats.cuda_runtime_events = int(kernel_payload.get("cuda_runtime_events", 0))
    stats.distributed_events = int(kernel_payload.get("distributed_events", 0))
    stats.duplicate_comm_event_filtered_count = int(kernel_payload.get("duplicate_comm_event_filtered_count", 0))
    stats.kernel_aggs = load_aggregate_map(kernel_payload.get("kernel_aggs", []))  # type: ignore[assignment]
    stats.event_hotspots = load_aggregate_map(event_payload.get("event_hotspots", []))  # type: ignore[assignment]
    for key, agg in stats.kernel_aggs.items():
        kind = OpKind(key[0])
        if is_distributed_kind(kind):
            stats.distributed[kind].add(agg.total_us, agg.calls)
    print(f"[INFO] Use cache for rank={rank}: {kernel_cache}", flush=True)
    return stats


def write_rank_stats_cache(path: Path, stats: RankStats, kernel_cache: Path, event_cache: Path) -> None:
    kernel_cache.parent.mkdir(parents=True, exist_ok=True)
    common = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "trace": trace_metadata(path),
        "rank": stats.rank,
        "total_kernel_us": stats.total_kernel_us,
        "raw_gpu_kernel_us": stats.raw_gpu_kernel_us,
        "duplicate_gpu_kernel_us_filtered": stats.duplicate_gpu_kernel_us_filtered,
        "raw_comm_kernel_us": stats.raw_comm_kernel_us,
        "dedup_comm_kernel_us": stats.dedup_comm_kernel_us,
        "duplicate_comm_event_filtered_us": stats.duplicate_comm_event_filtered_us,
        "parsed_events": stats.parsed_events,
        "gpu_kernel_events": stats.gpu_kernel_events,
        "raw_gpu_kernel_events": stats.raw_gpu_kernel_events,
        "duplicate_gpu_kernel_events_filtered": stats.duplicate_gpu_kernel_events_filtered,
        "raw_comm_kernel_events": stats.raw_comm_kernel_events,
        "dedup_comm_kernel_events": stats.dedup_comm_kernel_events,
        "profiler_events": stats.profiler_events,
        "cpu_op_events": stats.cpu_op_events,
        "cuda_runtime_events": stats.cuda_runtime_events,
        "distributed_events": stats.distributed_events,
        "duplicate_comm_event_filtered_count": stats.duplicate_comm_event_filtered_count,
    }
    kernel_payload = {
        **common,
        "kernel_aggs": serialize_aggregate_map(stats.kernel_aggs),
    }
    event_payload = {
        **common,
        "event_hotspots": serialize_aggregate_map(stats.event_hotspots),
    }
    kernel_cache.write_text(json.dumps(kernel_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    event_cache.write_text(json.dumps(event_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path = summary_cache_path(kernel_cache.parent.parent, stats.rank)
    if summary_path is not None:
        summary_path.write_text(json.dumps(common, ensure_ascii=False, indent=2), encoding="utf-8")


def is_gpu_kernel_event(event: Dict[str, Any]) -> bool:
    if event.get("ph") != "X":
        return False
    cat = str(event.get("cat", "")).lower()
    name = str(event.get("name", "")).lower()
    if cat in {"python_function", "user_annotation", "cpu_op", "cuda_runtime", "cuda_driver", "trace"}:
        return False
    if any(token in name for token in ("scheduler.run_batch", "scheduler.get_next_batch_to_run", "step[decode", "step[extend", "compiledfxgraph")):
        return False
    if cat in {"kernel", "gpu_memcpy", "gpu_memset"} or cat.startswith("kernel"):
        return True
    if cat.startswith("gpu_") and cat not in {"gpu_user_annotation", "gpu_annotation"}:
        return True
    args_text = " ".join(str(value).lower() for value in event_args(event).values())
    return bool(
        re.search(
            r"(nccldevkernel|void .*kernel|triton.*kernel|cutlass::device_kernel|nvjet|memcpy|memset)",
            name + " " + args_text,
        )
    )


def is_profiler_hotspot_event(event: Dict[str, Any]) -> bool:
    if event.get("ph") != "X" or is_gpu_kernel_event(event):
        return False
    dur = event.get("dur")
    return isinstance(dur, (int, float)) and dur > 0


def update_progress(stats: RankStats, progress_every: int, started_at: float) -> None:
    if progress_every <= 0 or stats.parsed_events % progress_every != 0:
        return
    elapsed = time.time() - started_at
    print(
        f"[PROGRESS] events={stats.parsed_events:,} "
        f"true_gpu_kernel_events={stats.gpu_kernel_events:,} "
        f"profiler_event_events={stats.profiler_events:,} "
        f"cpu_ops={stats.cpu_op_events:,} "
        f"cuda_runtime={stats.cuda_runtime_events:,} "
        f"distributed_kernel_events={stats.distributed_events:,} "
        f"elapsed={elapsed:.1f}s "
        f"total_kernel_time_us={stats.total_kernel_us:.0f}",
        file=sys.stderr,
        flush=True,
    )


def parse_trace_file(
    path: Path,
    rank: int,
    *,
    progress_every: int = 200000,
    max_events: Optional[int] = None,
    output_dir: Optional[Path] = None,
    use_cache: bool = True,
    source_map: Optional[List[Dict[str, Any]]] = None,
    kernel_mappings: Optional[Sequence[Dict[str, Any]]] = None,
) -> RankStats:
    kernel_cache, event_cache = cache_paths(output_dir, rank)
    if use_cache and kernel_cache is not None and event_cache is not None:
        cached = load_rank_stats_from_cache(path, rank, kernel_cache, event_cache)
        if cached is not None:
            return cached

    cpu_op_by_external_id: Dict[str, str] = {}
    source_by_external_id: Dict[str, str] = {}
    runtime_external_by_correlation: Dict[str, str] = {}
    seen_gpu_events: set[Tuple[str, ...]] = set()
    rank_stats = RankStats(rank=rank)
    started_at = time.time()

    print(f"[INFO] Parse trace rank={rank}: {path}", flush=True)
    for event in iter_trace_events(path):
        if max_events is not None and rank_stats.parsed_events >= max_events:
            break
        rank_stats.parsed_events += 1
        dur = event.get("dur", 0.0)
        if not isinstance(dur, (int, float)):
            update_progress(rank_stats, progress_every, started_at)
            continue
        args = event_args(event)
        external_key = get_external_key(event)
        correlation_key = get_correlation_key(event)

        cat = str(event.get("cat", ""))
        name = normalize_name(str(event.get("name", "unknown"))).replace("unknow", "unknown")
        if cat == "cpu_op":
            rank_stats.cpu_op_events += 1
        if cat in {"cuda_runtime", "cuda_driver"}:
            rank_stats.cuda_runtime_events += 1
        if external_key:
            source_from_args = extract_source_from_args(args)
            if source_from_args:
                source_by_external_id[external_key] = source_from_args
            if cat == "cpu_op":
                cpu_op_by_external_id[external_key] = name
            elif cat == "python_function" and ".py" in name:
                source_by_external_id[external_key] = name.split(":", 1)[0]
        if correlation_key and external_key and cat in {"cuda_runtime", "cuda_driver"}:
            runtime_external_by_correlation[correlation_key] = external_key

        if is_gpu_kernel_event(event):
            mapped_external_key = external_key or runtime_external_by_correlation.get(correlation_key, "")
            op_name = cpu_op_by_external_id.get(mapped_external_key, "")
            is_raw_comm = is_comm_text(name, op_name)
            rank_stats.raw_gpu_kernel_events += 1
            rank_stats.raw_gpu_kernel_us += float(dur)
            if is_raw_comm:
                rank_stats.raw_comm_kernel_events += 1
                rank_stats.raw_comm_kernel_us += float(dur)
            fingerprint = gpu_event_fingerprint(event, rank)
            if fingerprint in seen_gpu_events:
                rank_stats.duplicate_gpu_kernel_events_filtered += 1
                rank_stats.duplicate_gpu_kernel_us_filtered += float(dur)
                if is_raw_comm:
                    rank_stats.duplicate_comm_event_filtered_count += 1
                    rank_stats.duplicate_comm_event_filtered_us += float(dur)
                update_progress(rank_stats, progress_every, started_at)
                continue
            seen_gpu_events.add(fingerprint)
            direct_source = extract_source_from_args(args)
            correlated_source = source_by_external_id.get(mapped_external_key, "")
            source_file = direct_source or correlated_source
            source_type = "profiler_stack" if direct_source else ("correlation" if correlated_source else "unknown")
            kind = classify_op_kind(name, op_name, source_file)
            mapping = match_kernel_mapping(kernel_mappings or (), name)
            provider = "Unknown"
            op_kind = ""
            communication_type = "unknown"
            confidence = "low"
            needs_source_check = False
            if mapping is not None:
                if not op_name:
                    op_name = str(mapping.get("op_name") or "")
                if not source_file:
                    source_file = str(mapping.get("source_file") or "")
                    if source_file:
                        source_type = "kernel_name_mapping"
                provider = str(mapping.get("provider") or provider)
                op_kind = str(mapping.get("op_kind") or op_kind)
                communication_type = str(mapping.get("communication_type") or communication_type)
                confidence = str(mapping.get("confidence") or confidence)
                needs_source_check = bool(mapping.get("needs_source_check", False))
            if not source_file or not op_name:
                kind, op_name, mapped_source, mapped_source_type = apply_source_map(
                    source_map or [],
                    kind=kind,
                    op_name=op_name or "unknown",
                    kernel_name=name,
                    source_file=source_file,
                )
                if not source_file and mapped_source:
                    source_file = mapped_source
                    source_type = mapped_source_type
            rank_stats.total_kernel_us += float(dur)
            rank_stats.gpu_kernel_events += 1
            if is_raw_comm:
                rank_stats.dedup_comm_kernel_events += 1
                rank_stats.dedup_comm_kernel_us += float(dur)
            rank_stats.kernel_aggs[(kind.value, op_name or "unknown", name, source_file)].add(
                float(dur),
                op_name=op_name or "unknown",
                source_file=source_file,
                source_type=source_type,
                provider=provider,
                op_kind=op_kind,
                communication_type=communication_type,
                confidence=confidence,
                needs_source_check=needs_source_check,
            )
            if is_distributed_kind(kind):
                rank_stats.distributed[kind].add(float(dur))
                rank_stats.distributed_events += 1
        elif is_profiler_hotspot_event(event):
            rank_stats.profiler_events += 1
            source_file = source_by_external_id.get(external_key, "") or extract_source_from_args(args)
            rank_stats.event_hotspots[(cat or "unknown", source_file, name)].add(float(dur))

        update_progress(rank_stats, progress_every, started_at)

    print(
        f"[INFO] Parsed rank={rank} events={rank_stats.parsed_events:,} "
        f"gpu_kernel_events={rank_stats.gpu_kernel_events:,} "
        f"profiler_events={rank_stats.profiler_events:,} "
        f"distributed_events={rank_stats.distributed_events:,}",
        flush=True,
    )

    if use_cache and kernel_cache is not None and event_cache is not None:
        write_rank_stats_cache(path, rank_stats, kernel_cache, event_cache)

    return rank_stats


def merge_rank_stats(target: RankStats, source: RankStats) -> None:
    target.total_kernel_us += source.total_kernel_us
    target.raw_gpu_kernel_us += source.raw_gpu_kernel_us
    target.duplicate_gpu_kernel_us_filtered += source.duplicate_gpu_kernel_us_filtered
    target.raw_comm_kernel_us += source.raw_comm_kernel_us
    target.dedup_comm_kernel_us += source.dedup_comm_kernel_us
    target.duplicate_comm_event_filtered_us += source.duplicate_comm_event_filtered_us
    target.parsed_events += source.parsed_events
    target.gpu_kernel_events += source.gpu_kernel_events
    target.raw_gpu_kernel_events += source.raw_gpu_kernel_events
    target.duplicate_gpu_kernel_events_filtered += source.duplicate_gpu_kernel_events_filtered
    target.raw_comm_kernel_events += source.raw_comm_kernel_events
    target.dedup_comm_kernel_events += source.dedup_comm_kernel_events
    target.profiler_events += source.profiler_events
    target.cpu_op_events += source.cpu_op_events
    target.cuda_runtime_events += source.cuda_runtime_events
    target.distributed_events += source.distributed_events
    target.duplicate_comm_event_filtered_count += source.duplicate_comm_event_filtered_count
    target.kernels.extend(source.kernels)
    for kind, agg in source.distributed.items():
        target.distributed[kind].add(agg.total_us, agg.calls)
    for key, agg in source.kernel_aggs.items():
        target.kernel_aggs[key].add(
            agg.total_us,
            agg.calls,
            op_name=agg.op_name,
            source_file=agg.source_file,
            source_type=agg.source_type,
            provider=agg.provider,
            op_kind=agg.op_kind,
            communication_type=agg.communication_type,
            confidence=agg.confidence,
            needs_source_check=agg.needs_source_check,
        )
    for key, agg in source.event_hotspots.items():
        target.event_hotspots[key].add(agg.total_us, agg.calls)


def select_trace_files(report_dir: Path, rank_selector: str) -> List[Path]:
    trace_files = sorted(
        set(
            list(report_dir.glob("*.pt.trace.json"))
            + list(report_dir.glob("*.trace.json"))
            + list(report_dir.glob("*.pt.trace.json.gz"))
            + list(report_dir.glob("*.trace.json.gz"))
        )
    )
    trace_files = [path for path in trace_files if rank_matches(path, rank_selector)]
    latest_by_rank: Dict[int, Path] = {}
    for path in trace_files:
        rank = extract_rank(path)
        if rank is None:
            rank = 0
        current = latest_by_rank.get(rank)

        def capture_key(candidate: Path) -> Tuple[float, int, str]:
            match = re.match(r"(\d+(?:\.\d+)?)", candidate.name)
            capture_time = float(match.group(1)) if match else candidate.stat().st_mtime
            return capture_time, candidate.stat().st_mtime_ns, candidate.name

        if current is None or capture_key(path) > capture_key(current):
            latest_by_rank[rank] = path
    trace_files = [latest_by_rank[rank] for rank in sorted(latest_by_rank)]
    if not trace_files:
        raise SystemExit(f"Missing SGLang trace file in {report_dir} for rank={rank_selector}")
    return trace_files


def parse_profile_dir_by_rank(
    report_dir: Path,
    rank_selector: str = "0",
    *,
    progress_every: int = 200000,
    max_events: Optional[int] = None,
    output_dir: Optional[Path] = None,
    use_cache: bool = True,
    source_map: Optional[List[Dict[str, Any]]] = None,
    kernel_mappings: Optional[Sequence[Dict[str, Any]]] = None,
) -> ProfileStats:
    trace_files = select_trace_files(report_dir, rank_selector)

    rank_stats: Dict[int, RankStats] = {}
    for trace_file in trace_files:
        rank = extract_rank(trace_file)
        if rank is None:
            rank = 0 if rank_selector != "all" else len(rank_stats)
        parsed = parse_trace_file(
            trace_file,
            rank,
            progress_every=progress_every,
            max_events=max_events,
            output_dir=output_dir,
            use_cache=use_cache,
            source_map=source_map,
            kernel_mappings=kernel_mappings,
        )
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
        cache_dir=(output_dir / "cache") if output_dir else None,
    )


def fmt_ms(us: float) -> str:
    return f"{us / 1000.0:.3f}"


def fmt_us(us: float) -> str:
    return f"{us:.3f}"


def fmt_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def md_escape(text: str) -> str:
    value = str(text or "").replace("None", "")
    value = re.sub(r"\bunknow\b", "unknown", value)
    return value.replace("|", "\\|")


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


def aggregate_kernel_aggs(profile: ProfileStats) -> List[Tuple[Tuple[str, str, str, str], Aggregate]]:
    out: Dict[Tuple[str, str, str, str], Aggregate] = defaultdict(Aggregate)
    for stat in profile.rank_stats.values():
        for key, agg in stat.kernel_aggs.items():
            out[key].add(
                agg.total_us,
                agg.calls,
                op_name=agg.op_name,
                source_file=agg.source_file,
                source_type=agg.source_type,
                provider=agg.provider,
                op_kind=agg.op_kind,
                communication_type=agg.communication_type,
                confidence=agg.confidence,
                needs_source_check=agg.needs_source_check,
            )
    if out:
        return sorted(out.items(), key=lambda item: item[1].total_us, reverse=True)
    all_records = [record for stat in profile.rank_stats.values() for record in stat.kernels]
    return aggregate_records(all_records)


def aggregate_event_hotspots(profile: ProfileStats) -> List[Tuple[Tuple[str, str, str], Aggregate]]:
    out: Dict[Tuple[str, str, str], Aggregate] = defaultdict(Aggregate)
    for stat in profile.rank_stats.values():
        for key, agg in stat.event_hotspots.items():
            out[key].add(agg.total_us, agg.calls)
    return sorted(out.items(), key=lambda item: item[1].total_us, reverse=True)


REPORT_TYPE_LABELS = {
    OpKind.MOE.value: "MoE/Expert",
    OpKind.ATTENTION.value: "Attention",
    OpKind.MAMBA_OR_LINEAR_ATTENTION.value: "Linear Attention / Mamba",
    OpKind.GEMM.value: "GEMM/Linear",
    OpKind.NORM.value: "Norm/Fused Norm",
    OpKind.ACTIVATION.value: "Activation",
    OpKind.KV_CACHE.value: "KV Cache",
}


def report_type_for(kind: str, op_name: str, kernel_name: str, source_file: str) -> str:
    text, compact = normalized_search_text(kernel_name, op_name, source_file)
    if kind.startswith(DISTRIBUTED_PREFIX):
        if has_keyword(
            text,
            compact,
            "all_reduce_one_shot",
            "all_reduce_two_shot",
            "custom_all_reduce",
            "custom_ar",
            "outplace_all_reduce",
            "cross_device_reduce",
        ) and "nccldevkernel" not in compact:
            return "Communication/SGLang Custom AllReduce"
        if "nccl" in text:
            return "Communication/NCCL"
        if "gloo" in text or "control" in text:
            return "Communication/Control Plane"
        if "flashinfer" in text and ("comm" in text or "allreduce_fusion" in text):
            return "Communication/Fused Possible"
        return "Communication/Other"
    label = REPORT_TYPE_LABELS.get(kind)
    if label:
        return label
    if has_keyword(text, compact, "rope", "rotary", "position", "mrope"):
        return "RoPE/Position"
    if has_keyword(text, compact, "quant", "dequant", "per_token_group_quant", "fp8", "int8"):
        return "Quantization/Dequantization"
    if has_keyword(text, compact, "sampling", "softmax", "topk", "argmax", "exponential"):
        return "Sampling/Softmax"
    if has_keyword(text, compact, "index", "gather", "scatter", "sort", "where", "nonzero"):
        return "Indexing/Gather/Scatter"
    if has_keyword(text, compact, "memcpy", "memset", "copy", "fill"):
        return "Memcpy/Memset"
    if "triton" in text:
        return "Triton Other"
    return "Other GPU Kernel"


def compact_kernel_name(name: str, limit: int = 300) -> str:
    safe = re.sub(r"\bunknow\b", "unknown", str(name or ""))
    if len(safe) <= limit:
        return safe
    return safe[: limit - 3] + "..."


def mentor_title(model_name: str) -> str:
    base = model_name
    for token in ("-P32768D1024C1", "-P512D64C4", "-P128D16", "-Tiny"):
        base = base.replace(token, "")
    base = base.replace("-TP4", " TP4")
    return f"# FlagOSTune Torch Profiling 之 SGLang {base}"


def find_config_for_model(model_name: str) -> Optional[Path]:
    if not model_name:
        return None
    path = get_project_root() / f"config.yaml.{model_name}"
    return path if path.exists() else None


def load_model_config(model_name: str) -> Dict[str, Any]:
    path = find_config_for_model(model_name)
    if not path or yaml is None:
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def first_scenario(config: Dict[str, Any]) -> Dict[str, Any]:
    scenarios = ((config.get("benchmark") or {}).get("scenarios") or {})
    optimized = scenarios.get("optimized") or []
    if isinstance(optimized, list) and optimized:
        return optimized[0] or {}
    for value in scenarios.values():
        if isinstance(value, list) and value:
            return value[0] or {}
    return {}


class MetadataMismatchError(RuntimeError):
    """Raised before report generation when run identity is inconsistent."""


def _metadata_unwrap(value: Any) -> Any:
    if isinstance(value, dict) and "value" in value:
        return value.get("value")
    return value


def _metadata_nested(metadata: Dict[str, Any], section: str, key: str) -> Any:
    section_value = metadata.get(section) or {}
    if not isinstance(section_value, dict):
        return None
    return _metadata_unwrap(section_value.get(key))


def _scenario_identity(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _model_tp_size(model_name: str) -> Optional[int]:
    match = re.search(r"(?:^|-)TP(\d+)(?:-|$)", model_name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _model_scenario_identity(model_name: str) -> str:
    match = re.search(r"(?:^|-)P(\d+)D(\d+)C(\d+)(?:-|$)", model_name, flags=re.IGNORECASE)
    if not match:
        return ""
    return _scenario_identity(f"p{match.group(1)}d{match.group(2)}c{match.group(3)}")


def _trace_model_name(path: Path) -> str:
    parts = path.resolve().parts
    indices = [index for index, part in enumerate(parts) if part == "results"]
    if not indices:
        return ""
    index = indices[-1]
    return parts[index + 1] if index + 1 < len(parts) else ""


def _metadata_error(message: str) -> MetadataMismatchError:
    return MetadataMismatchError(f"metadata mismatch: {message}")


def validate_report_metadata(
    *,
    config_path: Path,
    trace_files: Sequence[Path],
    output_dir: Path,
    run_metadata_path: Path,
    expected_model: str,
    expected_scenario: str,
    expected_tp_size: int,
) -> Dict[str, Any]:
    """Validate config, report, trace and collected run metadata identities."""
    if not config_path.is_file():
        raise _metadata_error(f"config is missing: {config_path}")
    try:
        config_text = config_path.read_text(encoding="utf-8")
        config = (
            yaml.safe_load(config_text)
            if yaml is not None
            else json.loads(config_text)
        ) or {}
    except Exception as exc:
        raise _metadata_error(f"cannot read config: {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise _metadata_error(f"config is not a mapping: {config_path}")

    config_model = str(((config.get("model") or {}).get("name") or ""))
    config_tp = int(((config.get("model") or {}).get("tensor_parallel_size") or 0))
    scenario = first_scenario(config)
    config_scenario = str(scenario.get("name") or "")
    if config_model != expected_model:
        raise _metadata_error("report model does not match config model")
    if config_tp != int(expected_tp_size):
        raise _metadata_error("report TP size does not match config TP size")
    if _scenario_identity(config_scenario) != _scenario_identity(expected_scenario):
        raise _metadata_error("report scenario does not match config scenario")

    if output_dir.name != expected_model:
        raise _metadata_error("report model does not match output path")
    for trace_path in trace_files:
        trace_model = _trace_model_name(trace_path)
        if trace_model and trace_model != expected_model:
            raise _metadata_error("report model does not match trace path")

    embedded_tp = _model_tp_size(expected_model)
    if embedded_tp is not None and embedded_tp != int(expected_tp_size):
        raise _metadata_error("report TP size does not match model name")
    embedded_scenario = _model_scenario_identity(expected_model)
    if embedded_scenario and embedded_scenario != _scenario_identity(expected_scenario):
        raise _metadata_error("report scenario does not match model name")

    if not run_metadata_path.is_file():
        raise _metadata_error("run_metadata.json is missing")
    try:
        metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _metadata_error(f"cannot read run_metadata.json: {exc}") from exc
    if not isinstance(metadata, dict):
        raise _metadata_error("run_metadata.json is not an object")

    if str(_metadata_nested(metadata, "model", "model_name") or "") != expected_model:
        raise _metadata_error("report model does not match run_metadata.json")
    metadata_tp = _metadata_nested(metadata, "model", "tp_size")
    if int(metadata_tp or 0) != int(expected_tp_size):
        raise _metadata_error("report TP size does not match run_metadata.json")
    metadata_scenario = _metadata_nested(metadata, "benchmark", "scenario_name")
    if _scenario_identity(metadata_scenario) != _scenario_identity(expected_scenario):
        raise _metadata_error("report scenario does not match run_metadata.json")

    metadata_traces = _metadata_nested(metadata, "trace", "trace_files") or []
    recorded_paths = {
        str(Path(item.get("path", "")).resolve())
        for item in metadata_traces
        if isinstance(item, dict) and item.get("path")
    }
    for trace_path in trace_files:
        if str(trace_path.resolve()) not in recorded_paths:
            raise _metadata_error("selected trace path is not recorded in run_metadata.json")

    return {
        "model_name": expected_model,
        "scenario": expected_scenario,
        "tp_size": int(expected_tp_size),
        "config_path": str(config_path),
        "run_metadata_path": str(run_metadata_path),
        "trace_files": [str(path) for path in trace_files],
    }


def infer_model_name(profile: ProfileStats) -> str:
    for path in profile.trace_files:
        parts = list(path.parts)
        if "results" in parts:
            idx = parts.index("results")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return ""


def format_size(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "未采集"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def load_run_metadata(model_name: str) -> Dict[str, Any]:
    candidates = [
        get_project_root() / "reports" / model_name / "run_metadata.json",
    ]
    cfg = load_tool_config()
    reports_dir = ((cfg.get("paths") or {}).get("reports_dir"))
    if reports_dir:
        candidates.insert(0, resolve_path(reports_dir) / "run_metadata.json")
    for path in candidates:
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
        except Exception:
            continue
    return {}


def metadata_value(metadata: Dict[str, Any], section: str, key: str, default: Any = "未采集") -> Any:
    value = ((metadata.get(section) or {}).get(key) or {})
    if isinstance(value, dict) and "value" in value:
        return value.get("value") if value.get("value") not in (None, "") else default
    return default


def profiler_config_summary(metadata: Dict[str, Any]) -> str:
    trace = metadata.get("trace") or {}
    profiler = trace.get("profiler_config") or {}
    if not profiler:
        return "未采集"
    return " / ".join(f"{key}={value}" for key, value in profiler.items())


def build_environment_rows(profile: ProfileStats, rank_selector: str, model_name: str) -> List[List[str]]:
    metadata = load_run_metadata(model_name)
    config = load_model_config(model_name)
    scenario = first_scenario(config)
    model_cfg = config.get("model") or {}
    runtime_cfg = config.get("runtime") or {}
    sglang_cfg = config.get("sglang") or {}
    bench_cfg = config.get("benchmark") or {}
    trace = profile.trace_files[0] if profile.trace_files else Path("")
    cuda_visible = metadata_value(metadata, "gpu", "cuda_visible_devices", default="未采集")
    gpu_devices = metadata_value(metadata, "gpu", "visible_gpus", default=[])
    gpu_desc = "未采集"
    if isinstance(gpu_devices, list) and gpu_devices:
        gpu_desc = "; ".join(f"{item.get('index')}:{item.get('name')} {item.get('total_memory_mb')}MB" for item in gpu_devices)
    version_desc = " / ".join(
        str(value)
        for value in [
            metadata_value(metadata, "environment", "python_version", default="Python未采集"),
            metadata_value(metadata, "environment", "torch_version", default="Torch未采集"),
            metadata_value(metadata, "environment", "sglang_version", default="SGLang未采集"),
            metadata_value(metadata, "environment", "triton_version", default="Triton未采集"),
            metadata_value(metadata, "environment", "flashinfer_version", default="FlashInfer未采集"),
            metadata_value(metadata, "environment", "deep_gemm_version", default="DeepGEMM未采集"),
        ]
    )
    return [
        ["机器", "单机 H20，卡数/单卡显存未采集"],
        ["GPU", gpu_desc],
        ["框架", "SGLang + Torch profiler"],
        ["Python/Torch/CUDA/SGLang/Triton/FlashInfer/DeepGEMM 版本", version_desc],
        ["模型名", model_cfg.get("name") or model_name or "未采集"],
        ["model_path", metadata_value(metadata, "model", "model_path", default=model_cfg.get("path", "未采集"))],
        ["TP size", str(model_cfg.get("tensor_parallel_size", "未采集"))],
        ["dtype / FP8", str(runtime_cfg.get("dtype", "未采集")) + (" / FP8" if "FP8" in model_name.upper() else "")],
        ["CUDA_VISIBLE_DEVICES", cuda_visible],
        ["测试场景", str(scenario.get("name") or "未采集").replace("_c", " concurrency: ")],
        ["输入长度 / 输出长度 / 并发数 / runs", f"{scenario.get('input_len', '未采集')} / {scenario.get('output_len', '未采集')} / {scenario.get('concurrency', '未采集')} / {bench_cfg.get('num_runs', '未采集')}"],
        ["gpu_memory_utilization", str(sglang_cfg.get("gpu_memory_utilization", sglang_cfg.get("mem_fraction_static", "未采集")))],
        ["max_num_batched_tokens", str(sglang_cfg.get("max_num_batched_tokens", "未采集"))],
        ["max_num_seqs", str(sglang_cfg.get("max_num_seqs", "未采集"))],
        ["server_args", metadata_value(metadata, "model", "server_args", default=sglang_cfg.get("extra_args", "未采集"))],
        ["trace 文件路径", str(trace) if trace else "未采集"],
        ["trace 大小", format_size(trace) if trace else "未采集"],
        ["rank", rank_selector],
        ["profiler 配置 with_stack / record_shapes / profile_memory", profiler_config_summary(metadata)],
    ]


def row_value(rows: Sequence[Sequence[str]], key: str, default: str = "未采集") -> str:
    for row in rows:
        if len(row) >= 2 and row[0] == key:
            value = str(row[1])
            return value if value else default
    return default


def build_mentor_environment_section(env_rows: List[List[str]]) -> List[str]:
    machine = row_value(env_rows, "机器")
    framework = row_value(env_rows, "框架")
    tp_size = row_value(env_rows, "TP size")
    dtype = row_value(env_rows, "dtype / FP8")
    scenario = row_value(env_rows, "测试场景")
    io_runs = row_value(env_rows, "输入长度 / 输出长度 / 并发数 / runs")
    params = []
    for key, flag in [
        ("gpu_memory_utilization", "--gpu_memory_utilization"),
        ("max_num_batched_tokens", "--max-num-batched-tokens"),
        ("max_num_seqs", "--max-num-seqs"),
    ]:
        value = row_value(env_rows, key)
        if value != "未采集":
            params.append(f"{flag} {value}")
    if not params:
        server_args = row_value(env_rows, "server_args")
        if server_args != "未采集":
            params.append(server_args)

    lines = [
        "# 环境",
        "",
        f"{machine}，{framework}，TP={tp_size}，{dtype}",
        "",
        f"参数：{' '.join(params) if params else '未采集'}",
        "",
        f"测试场景：{scenario}；输入长度 / 输出长度 / 并发数 / runs：{io_runs}",
        "",
    ]
    return lines


def scan_log_summary(model_name: str) -> List[List[str]]:
    roots = [get_project_root() / "results" / model_name]
    text_parts: List[str] = []
    for root in roots:
        if not root.exists():
            continue
        for log in root.rglob("*.log"):
            try:
                text_parts.append(log.read_text(encoding="utf-8", errors="ignore")[-200000:])
            except OSError:
                pass
    text = "\n".join(text_parts)
    if not text:
        return [
            ["模型加载显存", "未采集"],
            ["profiling 前后显存", "未采集"],
            ["请求数/吞吐/latency/TTFT/ITL/TPOT", "未采集"],
            ["异常信息", "未在日志中发现 OOM / crash"],
        ]
    oom = re.findall(r".*(?:OOM|out of memory|crash|Traceback|ERROR).*", text, flags=re.IGNORECASE)
    metric_lines = re.findall(r".*(?:throughput|latency|TTFT|ITL|TPOT|request).*", text, flags=re.IGNORECASE)
    memory_lines = re.findall(
        r".*(?:Required memory for warmup|Available memory|GPU memory pool size|max_running_requests was reduced|total_gpu_memory|available_gpu_memory|memory_usage).*",
        text,
        flags=re.IGNORECASE,
    )
    load_memory = [line for line in memory_lines if re.search(r"Required memory for warmup|GPU memory pool size|total_gpu_memory", line, flags=re.IGNORECASE)]
    profile_memory = [line for line in memory_lines if re.search(r"Available memory|available_gpu_memory|memory_usage|max_running_requests was reduced", line, flags=re.IGNORECASE)]
    return [
        ["模型加载显存", md_escape("<br>".join(line[:220] for line in load_memory[:3])) if load_memory else "未采集"],
        ["profiling 前后显存", md_escape("<br>".join(line[:220] for line in profile_memory[:3])) if profile_memory else "未采集"],
        ["请求数/吞吐/latency/TTFT/ITL/TPOT", md_escape("<br>".join(line[:180] for line in metric_lines[:5])) if metric_lines else "未采集"],
        ["异常信息", md_escape("<br>".join(line[:180] for line in oom[:5])) if oom else "未在日志中发现 OOM / crash"],
    ]


def build_gpu_conclusions(kernel_aggs: List[Tuple[Tuple[str, str, str, str], Aggregate]], type_rows: List[List[str]]) -> List[str]:
    conclusions: List[str] = []
    if not kernel_aggs:
        return ["未采集到 CUDA/GPU kernel 数据。"]
    type_totals = [(row[0], float(row[2])) for row in type_rows if row[2] != "0.000"]
    if type_totals:
        dominant = max(type_totals, key=lambda item: item[1])
        conclusions.append(f"{dominant[0]} 是当前 rank 中总时间最高的类型，累计 {dominant[1]:.3f} ms。")
    comm = [row for row in type_rows if row[0].startswith("Communication/") and row[2] != "0.000"]
    if comm:
        comm_ms = sum(float(row[2]) for row in comm)
        conclusions.append(f"通信类 kernel 合计 {comm_ms:.3f} ms；custom all-reduce 与 NCCL 已分开统计。")
    top_names = [compact_kernel_name(item[0][2], 120) for item in kernel_aggs[:3]]
    conclusions.append(f"Top1 kernel 为 `{top_names[0]}`；Top3 由 {'; '.join('`' + name + '`' for name in top_names)} 构成。")
    labels = {row[0]: float(row[2]) for row in type_rows}
    focus = max(("MoE/Expert", "Attention", "Linear Attention / Mamba", "GEMM/Linear", "Communication/SGLang Custom AllReduce", "Communication/NCCL"), key=lambda name: labels.get(name, 0.0))
    conclusions.append(f"MoE / Attention / GEMM / communication 中，{focus} 在表格聚合中占主导。")
    if focus in {"MoE/Expert", "GEMM/Linear"}:
        conclusions.append("FlagTree/Triton/megakernel 优化优先看专家计算、GEMM 形状合并和 launch 开销。")
    elif focus == "Attention":
        conclusions.append("后续优化优先看 Attention/KV cache 路径的融合与长上下文访存。")
    else:
        conclusions.append("后续优化优先看通信规约、rank 间负载均衡和计算通信重叠。")
    return conclusions[:6]


def build_event_conclusions(event_aggs: List[Tuple[Tuple[str, str, str], Aggregate]]) -> List[str]:
    if not event_aggs:
        return ["未采集到 profiler event 热点。"]
    total = sum(agg.total_us for _, agg in event_aggs)
    top = event_aggs[0]
    event_type, _source, event_name = top[0]
    pct = top[1].total_us / total if total else 0.0
    return [
        f"Profiler Event 热点 Top1 为 `{compact_kernel_name(event_name, 120)}`，类型 `{event_type}`，占 profiler event 累计耗时 {fmt_pct(pct)}。",
        "该表用于观察端到端调度、Python、runtime 和 annotation 热点，不计入 True GPU Kernel 总时间。",
    ]


def build_unknown_rows(kernel_aggs: List[Tuple[Tuple[str, str, str, str], Aggregate]], limit: int = 50) -> List[List[str]]:
    rows: List[List[str]] = []
    for (kind, op_name, kernel_name, source_file), agg in kernel_aggs:
        if op_name != "unknown" and kind != OpKind.NON_DISTRIBUTED.value and source_file:
            continue
        rows.append([
            md_escape(compact_kernel_name(kernel_name)),
            str(agg.calls),
            fmt_ms(agg.total_us),
            report_type_for(kind, op_name, kernel_name, source_file),
            md_escape(source_file),
            "missing cpu_op/source mapping" if op_name == "unknown" or not source_file else "classified as other",
        ])
        if len(rows) >= limit:
            break
    return rows


def write_debug_outputs(profile: ProfileStats, output_dir: Path) -> None:
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    kernel_aggs = aggregate_kernel_aggs(profile)
    rows = build_unknown_rows(kernel_aggs, limit=100)
    md = "# Unknown / Unmapped Kernel\n\n" + md_table(
        ["kernel_name", "调用次数", "总时间(ms)", "guessed_type", "candidate_source", "reason"],
        rows,
    )
    (debug_dir / "rank0_unknown_op_top_kernels.md").write_text(md + "\n", encoding="utf-8")
    (debug_dir / "rank0_unmapped_top_kernels.md").write_text(md + "\n", encoding="utf-8")
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "unknown_rows": rows,
        "note": "candidate_profiler_events are not retained in streaming mode to keep memory bounded.",
    }
    (debug_dir / "rank0_source_mapping_debug.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_type_rows(
    kernel_aggs: List[Tuple[Tuple[str, str, str, str], Aggregate]],
    total_ref: float,
    non_comm_ref: float,
) -> List[List[str]]:
    type_order = [
        "Communication/SGLang Custom AllReduce",
        "Communication/NCCL",
        "Communication/Control Plane",
        "Communication/Fused Possible",
        "Communication/Other",
        "MoE/Expert",
        "Attention",
        "Linear Attention / Mamba",
        "GEMM/Linear",
        "Norm/Fused Norm",
        "Activation",
        "Quantization/Dequantization",
        "RoPE/Position",
        "KV Cache",
        "Sampling/Softmax",
        "Indexing/Gather/Scatter",
        "Memcpy/Memset",
        "Triton Other",
        "Other GPU Kernel",
    ]
    type_aggs: Dict[str, Aggregate] = {name: Aggregate() for name in type_order}
    for (kind, op_name, kernel_name, source_file), agg in kernel_aggs:
        type_aggs[report_type_for(kind, op_name, kernel_name, source_file)].add(agg.total_us, agg.calls)
    type_rows: List[List[str]] = []
    for label in type_order:
        agg = type_aggs[label]
        denom = total_ref if label.startswith("Communication/") else non_comm_ref
        type_rows.append([
            label,
            str(agg.calls),
            fmt_ms(agg.total_us),
            fmt_us(agg.avg_us),
            fmt_pct(agg.total_us / total_ref if total_ref > 0 else 0.0),
            fmt_pct(agg.total_us / denom if denom > 0 else 0.0),
        ])
    return type_rows


def build_credibility_rows(profile: ProfileStats) -> List[List[str]]:
    parsed_events = sum(stat.parsed_events for stat in profile.rank_stats.values())
    true_gpu_kernel_events = sum(stat.gpu_kernel_events for stat in profile.rank_stats.values())
    profiler_event_events = sum(stat.profiler_events for stat in profile.rank_stats.values())
    duplicate_comm = sum(stat.duplicate_comm_event_filtered_count for stat in profile.rank_stats.values())
    kernel_aggs = aggregate_kernel_aggs(profile)
    total_calls = sum(agg.calls for _, agg in kernel_aggs)
    unknown_calls = sum(agg.calls for (kind, op_name, _kernel, _source), agg in kernel_aggs if op_name == "unknown" or kind == OpKind.NON_DISTRIBUTED.value)
    unmapped_calls = sum(agg.calls for (_kind, op_name, _kernel, _source), agg in kernel_aggs if op_name == "unknown")
    missing_source_calls = sum(agg.calls for (_kind, _op_name, _kernel, source), agg in kernel_aggs if not source)
    source_map_calls = sum(agg.calls for (_kind, _op_name, _kernel, source), agg in kernel_aggs if "[source_map:" in source)
    correlation_source_calls = sum(agg.calls for (_kind, _op_name, _kernel, source), agg in kernel_aggs if "[correlation]" in source)
    profiler_source_calls = sum(
        agg.calls
        for (_kind, _op_name, _kernel, source), agg in kernel_aggs
        if source and "[source_map:" not in source and "[correlation]" not in source
    )
    distributed_total = profile.distributed_total_us
    return [
        ["parsed events", f"{parsed_events:,}"],
        ["true_gpu_kernel_events", f"{true_gpu_kernel_events:,}"],
        ["profiler_event_events", f"{profiler_event_events:,}"],
        ["total_true_gpu_kernel_time_ms", fmt_ms(profile.total_kernel_us)],
        ["distributed_kernel_time_ms", fmt_ms(distributed_total)],
        ["distributed_pct", fmt_pct(distributed_total / profile.total_kernel_us if profile.total_kernel_us else 0.0)],
        ["unmapped_kernel_pct", fmt_pct(unmapped_calls / total_calls if total_calls else 0.0)],
        ["unknown_op_pct", fmt_pct(unknown_calls / total_calls if total_calls else 0.0)],
        ["source_file_missing_pct", fmt_pct(missing_source_calls / total_calls if total_calls else 0.0)],
        ["source_file_from_profiler_pct", fmt_pct(profiler_source_calls / total_calls if total_calls else 0.0)],
        ["source_file_from_correlation_pct", fmt_pct(correlation_source_calls / total_calls if total_calls else 0.0)],
        ["source_file_from_source_map_pct", fmt_pct(source_map_calls / total_calls if total_calls else 0.0)],
        ["duplicate_comm_event_filtered_count", f"{duplicate_comm:,}"],
        ["duplicate_comm_event_filtered_time_ms", "未采集"],
        ["cache_used", "见解析日志"],
        ["parser version", str(CACHE_SCHEMA_VERSION)],
    ]


def build_op_kernel_rows(
    kernel_aggs: List[Tuple[Tuple[str, str, str, str], Aggregate]],
    total_ref: float,
    non_comm_ref: float,
    top_kernels_per_op: int,
) -> List[List[str]]:
    groups: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for (kind, op_name, kernel_name, source_file), agg in kernel_aggs:
        key = (kind, op_name, source_file)
        group = groups.setdefault(key, {"calls": 0, "total_us": 0.0, "kernels": defaultdict(float)})
        group["calls"] += agg.calls
        group["total_us"] += agg.total_us
        group["kernels"][kernel_name] += agg.total_us

    rows: List[List[str]] = []
    for (kind, op_name, source_file), group in sorted(groups.items(), key=lambda item: item[1]["total_us"], reverse=True)[:200]:
        total_us = float(group["total_us"])
        calls = int(group["calls"])
        kernels = sorted(group["kernels"].items(), key=lambda item: item[1], reverse=True)[:top_kernels_per_op]
        kernel_text = "<br>".join(md_escape(compact_kernel_name(name)) for name, _ in kernels)
        denom = total_ref if kind.startswith(DISTRIBUTED_PREFIX) else non_comm_ref
        rows.append([
            md_escape(source_file),
            source_type_for(source_file),
            md_escape(op_name),
            report_type_for(kind, op_name, "", source_file),
            kernel_text,
            str(calls),
            fmt_ms(total_us),
            fmt_us(total_us / calls if calls else 0.0),
            fmt_pct(total_us / denom if denom > 0 else 0.0),
            fmt_pct(total_us / total_ref if total_ref > 0 else 0.0),
        ])
    return rows


def build_mentor_cuda_rows(detail_rows: List[List[str]]) -> List[List[str]]:
    rows: List[List[str]] = []
    for row in detail_rows:
        rows.append([
            row[0],
            row[2],
            row[4],
            row[5],
            row[6],
            row[7],
            row[8],
        ])
    return rows


def source_type_for(source_file: str) -> str:
    if "[correlation]" in source_file:
        return "correlation"
    match = re.search(r"\[source_map:([a-z]+)\]", source_file)
    if match:
        return f"source_map_{match.group(1)}"
    if source_file:
        return "profiler_stack"
    return "unknown"


def build_markdown(
    profile: ProfileStats,
    rank_selector: str,
    *,
    model_name: str = "",
    top_kernels_per_op: int = 8,
    comm_mappings: Optional[Sequence[Dict[str, Any]]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    model_name = model_name or infer_model_name(profile) or "unknown"
    lines: List[str] = [mentor_title(model_name), ""]
    tables: List[Dict[str, Any]] = []
    distributed_total = profile.distributed_total_us
    total_ref = profile.total_kernel_us or profile.profiler_txt_total_us
    non_comm_ref = max(profile.total_kernel_us - distributed_total, 0.0) or profile.total_kernel_us
    parsed_events = sum(stat.parsed_events for stat in profile.rank_stats.values())
    gpu_events = sum(stat.gpu_kernel_events for stat in profile.rank_stats.values())
    all_kernel_aggs = aggregate_kernel_aggs(profile)
    event_aggs = aggregate_event_hotspots(profile)
    type_rows = build_type_rows(all_kernel_aggs, total_ref, non_comm_ref)

    env_rows = build_environment_rows(profile, rank_selector, model_name)
    lines.extend(build_mentor_environment_section(env_rows))
    tables.append({"sheet_name": "Environment", "headers": ["字段", "值"], "rows": env_rows})

    lines.append("## 算子数据")
    lines.append("")
    lines.append("1. 占比说明：Communication / NCCL / all_reduce 使用全部 GPU kernel 总时间作为分母；其它算子默认使用排除通信后的 GPU kernel 总时间作为分母；同时保留 overall_pct。")
    lines.append("2. Torch profiler duration 是事件耗时累计，不完全等同 wall-clock latency。")
    lines.append(f"3. 基于 torch profiler rank {rank_selector} trace 文件生成。")
    lines.append(f"4. parsed events: {parsed_events:,}；true gpu kernel events: {gpu_events:,}。")
    lines.append("5. 本报告 True GPU Kernel 只统计真实 CUDA/NCCL/Triton kernel；fast_trace_kernel_summary.py 的 gpu_events 是 kernel-like 快速过滤口径，会包含更多 CUDA/runtime/Triton/高层相似事件，二者不可直接对齐。")
    lines.append("")

    cuda_detail_rows = build_op_kernel_rows(all_kernel_aggs, total_ref, non_comm_ref, top_kernels_per_op)
    cuda_rows = build_mentor_cuda_rows(cuda_detail_rows)
    mentor_kernel_headers = ["source file", "op_name", "kernel_name", "调用次数", "总时间(ms)", "平均时间(us)", "占比"]
    lines.append("## CUDA kernel（按总时间排序）")
    lines.append("")
    lines.append(md_table(mentor_kernel_headers, cuda_rows))
    lines.append("")
    tables.append({"sheet_name": "CUDA_GPU_Kernel", "headers": mentor_kernel_headers, "rows": cuda_rows})

    if comm_mappings is None:
        comm_mappings = load_kernel_mappings(
            get_project_root() / "scripts" / "tools" / "sglang_comm_kernel_mapping.yaml"
        )
    focus_kernel_rows = [
        {
            "kind": kind,
            "op_name": op_name,
            "kernel_name": kernel_name,
            "source_file": source_file,
            "calls": agg.calls,
            "total_us": agg.total_us,
        }
        for (kind, op_name, kernel_name, source_file), agg in all_kernel_aggs
    ]
    focus_event_rows = [
        {
            "event_type": event_type,
            "source_file": source_file,
            "event_name": event_name,
            "calls": agg.calls,
            "total_us": agg.total_us,
        }
        for (event_type, source_file, event_name), agg in event_aggs
    ]
    focus_markdown, focus_tables = build_focus_report_sections(
        kernel_rows=focus_kernel_rows,
        event_rows=focus_event_rows,
        total_gpu_us=total_ref,
        mappings=comm_mappings,
    )

    credibility_rows = build_credibility_rows(profile)
    lines.append("# 数据可信度说明")
    lines.append("")
    lines.append(md_table(["字段", "值"], credibility_rows))
    lines.append("")
    lines.append("说明：如果 trace 不包含 stack 或 launch correlation，source_file 可能为空；unknown_op_pct 越高，表示 kernel 到 cpu_op 的关联越不完整。source_map 是候选源码路径，不是 profiler 原生 stack。")
    lines.append("")
    tables.append({"sheet_name": "Credibility", "headers": ["字段", "值"], "rows": credibility_rows})

    lines.append(focus_markdown)
    tables.extend(focus_tables)

    lines.append("# 核心结论")
    lines.append("")
    lines.append("GPU kernel 结论：")
    lines.extend(f"- {line}" for line in build_gpu_conclusions(all_kernel_aggs, type_rows))
    lines.append("")
    lines.append("Profiler Event 结论：")
    lines.extend(f"- {line}" for line in build_event_conclusions(event_aggs))
    lines.append("")

    log_rows = scan_log_summary(model_name)
    lines.append("# 显存与运行日志摘要")
    lines.append("")
    lines.append(md_table(["字段", "值"], log_rows))
    lines.append("")
    tables.append({"sheet_name": "LogSummary", "headers": ["字段", "值"], "rows": log_rows})

    lines.append("# 环境与运行配置明细")
    lines.append("")
    lines.append(md_table(["字段", "值"], env_rows))
    lines.append("")

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
    if sanity_note:
        lines.append(f"> {sanity_note}")
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

    lines.append("## True GPU Kernel 明细（含 source_type / op_kind / overall_pct）")
    lines.append("")
    kernel_headers = ["source file", "source_type", "op_name", "op_kind", "kernel_name", "调用次数", "总时间(ms)", "平均时间(us)", "占比", "overall_pct"]
    lines.append(md_table(kernel_headers, cuda_detail_rows))
    lines.append("")
    tables.append({"sheet_name": "CUDA_GPU_Kernel_Detail", "headers": kernel_headers, "rows": cuda_detail_rows})

    event_rows: List[List[str]] = []
    profiler_total = sum(agg.total_us for _, agg in event_aggs)
    for (event_type, source_file, event_name), agg in event_aggs[:200]:
        event_rows.append([
            md_escape(event_type),
            md_escape(source_file),
            md_escape(compact_kernel_name(event_name)),
            str(agg.calls),
            fmt_ms(agg.total_us),
            fmt_us(agg.avg_us),
            fmt_pct(agg.total_us / profiler_total if profiler_total > 0 else 0.0),
        ])
    lines.append("## Profiler Event 热点（按总时间排序）")
    lines.append("")
    lines.append(md_table(["event_type", "source file", "event_name", "调用次数", "总时间(ms)", "平均时间(us)", "占比"], event_rows))
    lines.append("")
    tables.append({"sheet_name": "ProfilerEventHotspot", "headers": ["event_type", "source file", "event_name", "调用次数", "总时间(ms)", "平均时间(us)", "占比"], "rows": event_rows})

    dist_kernel_rows: List[List[str]] = []
    for (kind, op_name, kernel_name, source_file), agg in [item for item in all_kernel_aggs if item[0][0].startswith(DISTRIBUTED_PREFIX)][:200]:
        dist_kernel_rows.append([
            kind,
            md_escape(op_name),
            md_escape(compact_kernel_name(kernel_name)),
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
    for (kind, op_name, kernel_name, source_file), agg in all_kernel_aggs[:200]:
        all_kernel_rows.append([
            kind,
            md_escape(op_name),
            md_escape(compact_kernel_name(kernel_name)),
            md_escape(source_file),
            str(agg.calls),
            fmt_ms(agg.total_us),
            fmt_us(agg.avg_us),
        ])
    lines.append("## Top 全部 GPU Kernel")
    lines.append("")
    lines.append(md_table(["op_kind", "op_name", "kernel_name", "source_file", "调用次数", "总时间(ms)", "平均时间(us)"], all_kernel_rows))
    lines.append("")
    tables.append({
        "sheet_name": "TopAllKernel",
        "headers": ["op_kind", "op_name", "kernel_name", "source_file", "调用次数", "总时间(ms)", "平均时间(us)"],
        "rows": all_kernel_rows,
    })

    lines.append("## 按类型聚合")
    lines.append("")
    lines.append(md_table(["类型", "调用次数", "总时间(ms)", "平均时间(us)", "overall_pct", "占比"], type_rows))
    lines.append("")
    tables.append({"sheet_name": "TypeAggregation", "headers": ["类型", "调用次数", "总时间(ms)", "平均时间(us)", "overall_pct", "占比"], "rows": type_rows})

    unknown_rows = build_unknown_rows(all_kernel_aggs)
    lines.append("## Unknown / Unmapped Kernel 附录")
    lines.append("")
    lines.append(md_table(["kernel_name", "调用次数", "总时间(ms)", "guessed_type", "candidate_source", "reason"], unknown_rows))
    lines.append("")
    tables.append({"sheet_name": "UnknownUnmapped", "headers": ["kernel_name", "调用次数", "总时间(ms)", "guessed_type", "candidate_source", "reason"], "rows": unknown_rows})

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
    lines.append("## Rank 对比")
    lines.append("")
    if rank_selector == "all":
        lines.append(md_table(["op_kind", "各rank总时间(ms)", "各rank调用次数", "max(ms)", "min(ms)", "max/min"], rank_rows))
    else:
        lines.append(f"本报告基于 rank{rank_selector}。")
    lines.append("")
    tables.append({
        "sheet_name": "RankCompare",
        "headers": ["op_kind", "各rank总时间(ms)", "各rank调用次数", "max(ms)", "min(ms)", "max/min"],
        "rows": rank_rows,
    })

    return "\n".join(lines), tables


def write_excel(path: Path, tables: List[Dict[str, Any]]) -> None:
    if Workbook is None:
        print("[WARN] 缺少 openpyxl，跳过 Excel 输出", flush=True)
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
    parser.add_argument("--progress-every", type=int, default=200000, help="每解析多少个 event 输出一次进度，0 表示关闭")
    parser.add_argument("--max-events", type=int, default=None, help="debug 用，最多解析多少个 event")
    parser.add_argument("--no-xlsx", action="store_true", help="只生成 markdown，不写 Excel")
    parser.add_argument("--use-cache", type=str, default="true", help="是否使用 cache: true/false")
    parser.add_argument("--force-reparse", type=str, default="false", help="强制重扫 trace: true/false")
    parser.add_argument("--top-k", type=int, default=200, help="保留参数：Top 表行数")
    parser.add_argument("--top-kernels-per-op", type=int, default=8, help="每个 op 展示的 kernel_name 数量")
    parser.add_argument("--source-map", type=str, default=None, help="kernel/source 静态映射 YAML/JSON")
    parser.add_argument("--config-path", type=str, default=None, help="本次报告使用的模型配置文件")
    parser.add_argument("--expected-model", type=str, default=None, help="launcher 解析出的模型名")
    parser.add_argument("--expected-scenario", type=str, default=None, help="launcher 解析出的场景名")
    parser.add_argument("--expected-tp-size", type=int, default=None, help="launcher 解析出的 TP size")
    parser.add_argument("--run-metadata", type=str, default=None, help="本次运行的 run_metadata.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    default_torch_dir, default_output_dir = get_default_paths()
    torch_dir = resolve_path(args.torch_path) if args.torch_path else default_torch_dir
    output_dir = resolve_path(args.output_path) if args.output_path else default_output_dir
    report_dir = torch_dir / "report-sglang"
    cfg = load_tool_config()
    configured_model = str(((cfg.get("paths") or {}).get("model_name") or ""))
    expected_model = str(args.expected_model or configured_model)
    config_path = (
        resolve_path(args.config_path)
        if args.config_path
        else find_config_for_model(expected_model)
    )
    if config_path is None:
        print("[ERROR] metadata mismatch: config is missing", file=sys.stderr, flush=True)
        return 2
    config = load_model_config(expected_model)
    scenario = first_scenario(config)
    expected_scenario = str(args.expected_scenario or scenario.get("name") or "")
    expected_tp_size = int(
        args.expected_tp_size
        if args.expected_tp_size is not None
        else ((config.get("model") or {}).get("tensor_parallel_size") or 0)
    )
    run_metadata_path = (
        resolve_path(args.run_metadata)
        if args.run_metadata
        else output_dir / "run_metadata.json"
    )
    selected_trace_files = select_trace_files(report_dir, args.rank)
    try:
        validated_identity = validate_report_metadata(
            config_path=config_path,
            trace_files=selected_trace_files,
            output_dir=output_dir,
            run_metadata_path=run_metadata_path,
            expected_model=expected_model,
            expected_scenario=expected_scenario,
            expected_tp_size=expected_tp_size,
        )
    except MetadataMismatchError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    source_map = load_source_map(resolve_path(args.source_map)) if args.source_map else load_source_map(get_project_root() / "scripts" / "tools" / "sglang_kernel_source_map.yaml")
    comm_mappings = load_kernel_mappings(
        get_project_root() / "scripts" / "tools" / "sglang_comm_kernel_mapping.yaml"
    )
    kernel_mappings = load_kernel_mappings(
        get_project_root() / "scripts" / "tools" / "sglang_kernel_name_mapping.yaml"
    )
    use_cache = str(args.use_cache).lower() in {"1", "true", "yes", "on"} and str(args.force_reparse).lower() not in {"1", "true", "yes", "on"}

    profile = parse_profile_dir_by_rank(
        report_dir,
        rank_selector=args.rank,
        progress_every=args.progress_every,
        max_events=args.max_events,
        output_dir=output_dir,
        use_cache=use_cache,
        source_map=source_map,
        kernel_mappings=kernel_mappings,
    )
    model_name = str(validated_identity["model_name"])
    markdown, tables = build_markdown(
        profile,
        args.rank,
        model_name=model_name,
        top_kernels_per_op=args.top_kernels_per_op,
        comm_mappings=comm_mappings,
    )
    md_path = output_dir / "sglang_perf_analysis_torch.md"
    xlsx_path = output_dir / "sglang_perf_analysis_torch.xlsx"
    md_path.write_text(markdown, encoding="utf-8")
    write_debug_outputs(profile, output_dir)
    if not args.no_xlsx:
        write_excel(xlsx_path, tables)
    print(f"[INFO] Markdown report: {md_path}", flush=True)
    if not args.no_xlsx:
        print(f"[INFO] Excel report: {xlsx_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
