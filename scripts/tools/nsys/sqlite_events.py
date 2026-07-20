"""Schema-tolerant, read-only extraction of Nsight SQLite GPU events."""

from __future__ import annotations

import json
import sqlite3
import statistics
import time
from array import array
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .classify_kernels import base_family, classify_kernel
from .models import KernelEvent
from .utils import atomic_write_json, normalize_header, write_csv


@dataclass(frozen=True)
class SchemaInventory:
    tables: Dict[str, Tuple[str, ...]]


@dataclass(frozen=True)
class MemoryEvent:
    event_id: int
    start_ns: int
    end_ns: int
    device_id: int
    kind: str
    bytes: Optional[int]
    source_table: str


@dataclass(frozen=True)
class CudaApiEvent:
    __slots__ = (
        "event_id", "start_ns", "end_ns", "process_id", "thread_id",
        "correlation_id", "name", "source_table",
    )
    event_id: int
    start_ns: int
    end_ns: int
    process_id: Optional[int]
    thread_id: Optional[int]
    correlation_id: Optional[int]
    name: str
    source_table: str


@dataclass(frozen=True, init=False)
class NvtxRange:
    __slots__ = (
        "range_id", "start_ns", "end_ns", "process_id", "thread_id", "text",
        "source_table", "event_type", "domain_id", "global_tid", "end_global_tid",
    )
    range_id: int
    start_ns: int
    end_ns: int
    process_id: Optional[int]
    thread_id: Optional[int]
    text: str
    source_table: str
    event_type: Optional[int]
    domain_id: Optional[int]
    global_tid: Optional[int]
    end_global_tid: Optional[int]

    def __init__(
        self,
        range_id: int,
        start_ns: int,
        end_ns: int,
        process_id: Optional[int],
        thread_id: Optional[int],
        text: str,
        source_table: str,
        event_type: Optional[int] = None,
        domain_id: Optional[int] = None,
        global_tid: Optional[int] = None,
        end_global_tid: Optional[int] = None,
    ) -> None:
        for name, value in (
            ("range_id", range_id), ("start_ns", start_ns), ("end_ns", end_ns),
            ("process_id", process_id), ("thread_id", thread_id), ("text", text),
            ("source_table", source_table), ("event_type", event_type),
            ("domain_id", domain_id), ("global_tid", global_tid),
            ("end_global_tid", end_global_tid),
        ):
            object.__setattr__(self, name, value)


@dataclass
class NvtxLoadStats:
    table_name: Optional[str] = None
    total_rows: int = 0
    valid_closed_ranges: int = 0
    null_start_rows: int = 0
    null_end_rows: int = 0
    invalid_interval_rows: int = 0
    skipped_non_interval_rows: int = 0
    negative_start_rows: int = 0
    counts_by_event_type: Dict[int, int] = field(default_factory=dict)
    skipped_by_event_type: Dict[int, int] = field(default_factory=dict)
    skipped_text_samples: List[str] = field(default_factory=list)
    load_duration_seconds: float = 0.0
    estimated_memory_bytes: int = 0


@dataclass(frozen=True)
class NvtxLoadResult:
    ranges: List[NvtxRange]
    stats: NvtxLoadStats
    warnings: List[str]


@dataclass(frozen=True)
class EventExtraction:
    events: List[KernelEvent]
    api_events: List[CudaApiEvent]
    nvtx_ranges: List[NvtxRange]
    inventory: SchemaInventory
    missing_capabilities: List[str]
    column_mapping: Dict[str, Optional[str]]
    nvtx_stats: Optional[NvtxLoadStats] = None
    warnings: List[str] = field(default_factory=list)
    base_stats_status: str = "NOT_APPLICABLE"
    kernel_event_status: str = "PASS"
    nvtx_load_status: str = "NOT_REQUESTED"
    nvtx_attribution_status: str = "NOT_REQUESTED"
    dependency_analysis_status: str = "NOT_REQUESTED"
    communication_analysis_status: str = "NOT_REQUESTED"
    overall_status: str = "PASS"
    attribution_query_count: int = 0
    attribution_candidate_checks: int = 0


@dataclass(frozen=True)
class _RangeIndex:
    starts: Sequence[int]
    ranges: Sequence[NvtxRange]
    prefix_max_end: Sequence[int]


@dataclass
class _AttributionCounters:
    queries: int = 0
    candidate_checks: int = 0


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def inspect_schema(connection: sqlite3.Connection) -> SchemaInventory:
    tables: Dict[str, Tuple[str, ...]] = {}
    for (name,) in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ):
        columns = tuple(
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info({_quote(str(name))})")
        )
        tables[str(name)] = columns
    return SchemaInventory(tables)


def _table(inventory: SchemaInventory, candidates: Sequence[str]) -> Optional[str]:
    normalized = {normalize_header(name): name for name in inventory.tables}
    for candidate in candidates:
        found = normalized.get(normalize_header(candidate))
        if found:
            return found
    for normalized_name, original in normalized.items():
        if any(normalize_header(candidate) in normalized_name for candidate in candidates):
            return original
    return None


def _column(columns: Sequence[str], *aliases: str) -> Optional[str]:
    normalized = {normalize_header(name): name for name in columns}
    for alias in aliases:
        found = normalized.get(normalize_header(alias))
        if found:
            return found
    return None


def _strings(connection: sqlite3.Connection, inventory: SchemaInventory) -> Dict[int, str]:
    table = _table(inventory, ("StringIds", "StringId"))
    if not table:
        return {}
    columns = inventory.tables[table]
    id_column = _column(columns, "id", "stringId")
    value_column = _column(columns, "value", "string", "text")
    if not id_column or not value_column:
        return {}
    return {
        int(key): str(value)
        for key, value in connection.execute(
            f"SELECT {_quote(id_column)}, {_quote(value_column)} FROM {_quote(table)}"
        )
        if key is not None and value is not None
    }


def _int_or_none(value: object) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _connect_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _process_thread(
    process_value: object, thread_value: object
) -> Tuple[Optional[int], Optional[int]]:
    """Decode Nsight's packed globalTid while preserving legacy plain IDs."""
    process_id = _int_or_none(process_value)
    global_tid = _int_or_none(thread_value)
    if global_tid is None:
        return process_id, None
    if global_tid > 0xFFFFFF:
        return process_id if process_id is not None else global_tid >> 24, global_tid & 0xFFFFFF
    return process_id, global_tid


def _resolved_text(
    raw: Mapping[str, object], strings: Mapping[int, str]
) -> str:
    direct = raw.get("text")
    if direct not in (None, ""):
        return str(direct)
    text_id = _int_or_none(raw.get("text_id"))
    if text_id is None:
        return ""
    return strings.get(text_id, str(text_id))


def _load_cuda_api_events(
    connection: sqlite3.Connection,
    inventory: SchemaInventory,
    strings: Mapping[int, str],
) -> List[CudaApiEvent]:
    output = []
    event_id = 0
    for candidate in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        table = _table(inventory, (candidate,))
        if not table:
            continue
        columns = inventory.tables[table]
        mapping = {
            "start": _column(columns, "start", "startNs", "timestamp"),
            "end": _column(columns, "end", "endNs"),
            "process": _column(columns, "globalPid", "processId", "pid"),
            "thread": _column(columns, "globalTid", "threadId", "tid"),
            "correlation": _column(columns, "correlationId", "correlation"),
            "name": _column(columns, "name", "apiName"),
            "name_id": _column(columns, "nameId", "apiNameId"),
        }
        if not mapping["start"] or not mapping["end"] or not (
            mapping["name"] or mapping["name_id"]
        ):
            continue
        keys = [key for key, value in mapping.items() if value]
        query = "SELECT " + ", ".join(
            _quote(str(mapping[key])) for key in keys
        ) + f" FROM {_quote(table)} ORDER BY {_quote(str(mapping['start']))}"
        for values in connection.execute(query):
            raw = dict(zip(keys, values))
            if mapping["name"]:
                name = str(raw.get("name") or "")
            else:
                raw_id = _int_or_none(raw.get("name_id"))
                name = strings.get(raw_id, str(raw.get("name_id") or "Unknown"))
            start = _int_or_none(raw.get("start"))
            end = _int_or_none(raw.get("end"))
            if start is None or end is None or end < start:
                continue
            process_id, thread_id = _process_thread(
                raw.get("process"), raw.get("thread")
            )
            output.append(
                CudaApiEvent(
                    event_id,
                    start,
                    end,
                    process_id,
                    thread_id,
                    _int_or_none(raw.get("correlation")),
                    name,
                    table,
                )
            )
            event_id += 1
    return output


def _load_nvtx_ranges(
    connection: sqlite3.Connection,
    inventory: SchemaInventory,
    strings: Mapping[int, str],
) -> NvtxLoadResult:
    started = time.monotonic()
    table = _table(inventory, ("NVTX_EVENTS", "NVTX_EVENT", "NVTX_RANGES"))
    stats = NvtxLoadStats(table_name=table)
    if not table:
        stats.load_duration_seconds = time.monotonic() - started
        return NvtxLoadResult([], stats, ["NVTX event table is unavailable"])
    columns = inventory.tables[table]
    mapping = {
        "start": _column(columns, "start", "startNs", "timestamp"),
        "end": _column(columns, "end", "endNs"),
        "process": _column(columns, "globalPid", "processId", "pid"),
        "thread": _column(columns, "globalTid", "threadId", "tid"),
        "end_thread": _column(columns, "endGlobalTid", "endThreadId"),
        "text": _column(columns, "text", "message", "name"),
        "text_id": _column(columns, "textId", "messageId", "nameId"),
        "event_type": _column(columns, "eventType", "type"),
        "range_id": _column(columns, "rangeId", "id"),
        "domain_id": _column(columns, "domainId", "domain"),
    }
    if not mapping["start"] or not mapping["end"]:
        stats.load_duration_seconds = time.monotonic() - started
        return NvtxLoadResult(
            [], stats, [f"NVTX start/end columns are unavailable in {table}"]
        )
    keys = [key for key, value in mapping.items() if value]
    query = "SELECT " + ", ".join(
        _quote(str(mapping[key])) for key in keys
    ) + f" FROM {_quote(table)}"
    output: List[NvtxRange] = []
    counts: Dict[int, int] = defaultdict(int)
    skipped: Dict[int, int] = defaultdict(int)
    for ordinal, values in enumerate(connection.execute(query)):
        raw = dict(zip(keys, values))
        stats.total_rows += 1
        event_type = _int_or_none(raw.get("event_type"))
        if event_type is not None:
            counts[event_type] += 1
        start_ns = _int_or_none(raw.get("start"))
        end_ns = _int_or_none(raw.get("end"))
        if start_ns is not None and start_ns < 0:
            stats.negative_start_rows += 1
        if start_ns is None:
            stats.null_start_rows += 1
            stats.skipped_non_interval_rows += 1
            if event_type is not None:
                skipped[event_type] += 1
            continue
        if end_ns is None:
            stats.null_end_rows += 1
            stats.skipped_non_interval_rows += 1
            if event_type is not None:
                skipped[event_type] += 1
            if len(stats.skipped_text_samples) < 20:
                stats.skipped_text_samples.append(
                    json.dumps(
                        {
                            "eventType": event_type,
                            "text": _resolved_text(raw, strings),
                            "rangeId": _int_or_none(raw.get("range_id")),
                            "domainId": _int_or_none(raw.get("domain_id")),
                            "globalTid": _int_or_none(raw.get("thread")),
                            "start": start_ns,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            continue
        if end_ns < start_ns:
            stats.invalid_interval_rows += 1
            continue
        process_id, thread_id = _process_thread(
            raw.get("process"), raw.get("thread")
        )
        output.append(
            NvtxRange(
                _int_or_none(raw.get("range_id")) or ordinal,
                start_ns,
                end_ns,
                process_id,
                thread_id,
                _resolved_text(raw, strings),
                table,
                event_type,
                _int_or_none(raw.get("domain_id")),
                _int_or_none(raw.get("thread")),
                _int_or_none(raw.get("end_thread")),
            )
        )
    stats.valid_closed_ranges = len(output)
    stats.counts_by_event_type = dict(counts)
    stats.skipped_by_event_type = dict(skipped)
    stats.load_duration_seconds = time.monotonic() - started
    stats.estimated_memory_bytes = len(output) * 400
    warnings = []
    if stats.skipped_non_interval_rows:
        warnings.append(
            f"{table}: excluded {stats.skipped_non_interval_rows} non-interval "
            f"NVTX rows without synthesizing end timestamps; eventType counts="
            f"{stats.skipped_by_event_type}; samples={stats.skipped_text_samples[:3]}"
        )
    if stats.invalid_interval_rows:
        warnings.append(
            f"{table}: excluded {stats.invalid_interval_rows} reversed NVTX intervals"
        )
    return NvtxLoadResult(output, stats, warnings)


def _build_range_indexes(
    ranges: Sequence[NvtxRange],
) -> Dict[Tuple[Optional[int], Optional[int]], _RangeIndex]:
    grouped: Dict[Tuple[Optional[int], Optional[int]], List[NvtxRange]] = defaultdict(list)
    for nvtx in ranges:
        grouped[(nvtx.process_id, nvtx.thread_id)].append(nvtx)
    indexes = {}
    for key, values in grouped.items():
        values.sort(key=lambda row: (row.start_ns, -row.end_ns))
        maximum = -2**63
        starts = array("q")
        prefix = array("q")
        for row in values:
            starts.append(row.start_ns)
            maximum = max(maximum, row.end_ns)
            prefix.append(maximum)
        indexes[key] = _RangeIndex(starts, values, prefix)
    return indexes


def _matching_indexes(
    indexes: Mapping[Tuple[Optional[int], Optional[int]], _RangeIndex],
    process_id: Optional[int],
    thread_id: Optional[int],
) -> List[_RangeIndex]:
    exact = indexes.get((process_id, thread_id))
    if exact is not None:
        return [exact]
    if thread_id is None:
        return []
    return [
        index
        for (candidate_process, candidate_thread), index in indexes.items()
        if candidate_thread == thread_id
        and (process_id is None or candidate_process in (None, process_id))
    ]


def _innermost_range(
    indexes: Mapping[Tuple[Optional[int], Optional[int]], _RangeIndex],
    process_id: Optional[int],
    thread_id: Optional[int],
    start_ns: int,
    end_ns: int,
    counters: _AttributionCounters,
) -> Optional[NvtxRange]:
    candidates = []
    counters.queries += 1
    for index in _matching_indexes(indexes, process_id, thread_id):
        position = bisect_right(index.starts, start_ns) - 1
        while position >= 0 and index.prefix_max_end[position] >= end_ns:
            row = index.ranges[position]
            counters.candidate_checks += 1
            if row.start_ns <= start_ns and row.end_ns >= end_ns:
                candidates.append(row)
            position -= 1
    if not candidates:
        return None
    return min(candidates, key=lambda row: (row.end_ns - row.start_ns, -row.start_ns))


def _is_launch_api(name: str) -> bool:
    lowered = name.lower()
    return any(
        token in lowered
        for token in (
            "cudalaunchkernel", "culaunchkernel", "cudagraphlaunch", "cugraphlaunch"
        )
    )


def _attribute_nvtx(
    events: Sequence[KernelEvent],
    api_events: Sequence[CudaApiEvent],
    ranges: Sequence[NvtxRange],
) -> Tuple[List[KernelEvent], _AttributionCounters]:
    indexes = _build_range_indexes(ranges)
    api_by_correlation: Dict[int, List[CudaApiEvent]] = defaultdict(list)
    for api in api_events:
        if api.correlation_id is not None:
            api_by_correlation[api.correlation_id].append(api)
    counters = _AttributionCounters()
    output = []
    for event in events:
        selected = None
        selected_api = None
        reason = "kernel has no correlation id"
        correlated = (
            api_by_correlation.get(event.correlation_id, [])
            if event.correlation_id is not None
            else []
        )
        ordered_apis = sorted(
            correlated,
            key=lambda row: (not _is_launch_api(row.name), row.start_ns, row.end_ns),
        )
        for api in ordered_apis:
            candidate = _innermost_range(
                indexes, api.process_id, api.thread_id, api.start_ns, api.end_ns, counters
            )
            if candidate is not None:
                selected, selected_api = candidate, api
                if _is_launch_api(api.name):
                    break
        if selected is None and not correlated:
            reason = (
                "correlation id has no CUDA API event"
                if event.correlation_id is not None
                else reason
            )
            selected = _innermost_range(
                indexes,
                event.process_id,
                event.thread_id,
                event.start_ns,
                event.end_ns,
                counters,
            )
        elif selected is None:
            reason = "correlated CUDA API was not enclosed by a matching NVTX interval"
        if selected is None:
            output.append(
                replace(
                    event,
                    nvtx_attribution_source="UNATTRIBUTED",
                    nvtx_attribution_confidence="NONE",
                    nvtx_attribution_reason=reason,
                )
            )
            continue
        classification = classify_kernel(
            event.name, nvtx=selected.text, module=event.module
        )
        output.append(
            replace(
                event,
                nvtx_range=selected.text,
                nvtx_attribution_source=(
                    "CUDA_API_CORRELATION" if selected_api else "KERNEL_TIME_FALLBACK"
                ),
                nvtx_attribution_confidence="HIGH" if selected_api else "LOW",
                nvtx_attribution_reason=(
                    "matched CUDA API correlation and launch-thread NVTX interval"
                    if selected_api
                    else reason
                ),
                launch_api_name=selected_api.name if selected_api else "N/A",
                launch_api_start_ns=selected_api.start_ns if selected_api else None,
                category=classification.category,
                classification_rule=classification.rule,
                classification_confidence=classification.confidence,
            )
        )
    return output, counters


def load_kernel_events(
    path: Path,
    *,
    include_cuda_api: bool = False,
    include_nvtx: bool = False,
    attribute_nvtx: bool = False,
) -> EventExtraction:
    if attribute_nvtx:
        include_cuda_api = True
        include_nvtx = True
    connection = _connect_read_only(path)
    try:
        inventory = inspect_schema(connection)
        table = _table(
            inventory,
            ("CUPTI_ACTIVITY_KIND_KERNEL", "CUDA_GPU_KERNEL", "KERNEL_EVENTS"),
        )
        if not table:
            return EventExtraction(
                [], [], [], inventory, ["kernel event table is unavailable"], {},
                kernel_event_status="FAILED", overall_status="PARTIAL",
            )
        columns = inventory.tables[table]
        mapping = {
            "start": _column(columns, "start", "startNs", "timestamp"),
            "end": _column(columns, "end", "endNs"),
            "duration": _column(columns, "duration", "durationNs"),
            "device": _column(columns, "deviceId", "device", "gpuId"),
            "context": _column(columns, "contextId", "context"),
            "stream": _column(columns, "streamId", "stream"),
            "process": _column(columns, "globalPid", "processId", "pid"),
            "thread": _column(columns, "globalTid", "threadId", "tid"),
            "correlation": _column(columns, "correlationId", "correlation"),
            "name": _column(columns, "demangledName", "kernelName", "name"),
            "name_id": _column(columns, "demangledNameId", "nameId", "shortName"),
        }
        missing = []
        for required in ("start", "device", "context", "stream"):
            if not mapping[required]:
                missing.append(f"kernel event column {required} is unavailable in {table}")
        if not mapping["end"] and not mapping["duration"]:
            missing.append(f"kernel event end/duration column is unavailable in {table}")
        if not mapping["name"] and not mapping["name_id"]:
            missing.append(f"kernel event name column is unavailable in {table}")
        if missing:
            return EventExtraction(
                [], [], [], inventory, missing, mapping,
                kernel_event_status="FAILED", overall_status="PARTIAL",
            )

        select_keys = [key for key, value in mapping.items() if value]
        query = "SELECT " + ", ".join(
            _quote(str(mapping[key])) for key in select_keys
        ) + f" FROM {_quote(table)} ORDER BY {_quote(str(mapping['start']))}"
        strings = _strings(connection, inventory)
        events = []
        for event_id, raw_values in enumerate(connection.execute(query)):
            raw = dict(zip(select_keys, raw_values))
            start = int(raw["start"])
            end = (
                int(raw["end"])
                if raw.get("end") is not None
                else start + int(raw.get("duration") or 0)
            )
            if mapping["name"]:
                name = str(raw.get("name") or "")
            else:
                raw_id = _int_or_none(raw.get("name_id"))
                name = strings.get(raw_id, str(raw.get("name_id") or "Unknown"))
            classification = classify_kernel(name)
            process_id, thread_id = _process_thread(
                raw.get("process"), raw.get("thread")
            )
            events.append(
                KernelEvent(
                    event_id=event_id,
                    device_id=int(raw.get("device") or 0),
                    context_id=int(raw.get("context") or 0),
                    stream_id=int(raw.get("stream") or 0),
                    start_ns=start,
                    end_ns=end,
                    name=name,
                    category=classification.category,
                    family=base_family(name),
                    process_id=process_id,
                    thread_id=thread_id,
                    correlation_id=_int_or_none(raw.get("correlation")),
                    classification_rule=classification.rule,
                    classification_confidence=classification.confidence,
                    source_table=table,
                )
            )
        warnings = []
        api_events: List[CudaApiEvent] = []
        nvtx_ranges: List[NvtxRange] = []
        nvtx_stats = None
        nvtx_load_status = "NOT_REQUESTED"
        attribution_status = "NOT_REQUESTED"
        overall_status = "PASS"
        counters = _AttributionCounters()
        if include_cuda_api:
            try:
                api_events = _load_cuda_api_events(connection, inventory, strings)
            except (sqlite3.Error, TypeError, ValueError, MemoryError) as exc:
                warnings.append(f"CUDA API enrichment failed: {exc}")
                overall_status = "PARTIAL"
        if include_nvtx:
            try:
                nvtx_result = _load_nvtx_ranges(connection, inventory, strings)
                nvtx_ranges = nvtx_result.ranges
                nvtx_stats = nvtx_result.stats
                warnings.extend(nvtx_result.warnings)
                if nvtx_result.stats.table_name is None:
                    nvtx_load_status = "FAILED"
                    overall_status = "PARTIAL"
                else:
                    nvtx_load_status = (
                        "PASS_WITH_WARNINGS" if nvtx_result.warnings else "PASS"
                    )
                    if nvtx_result.warnings and overall_status == "PASS":
                        overall_status = "PASS_WITH_WARNINGS"
            except (sqlite3.Error, TypeError, ValueError, MemoryError) as exc:
                warnings.append(f"NVTX enrichment failed: {exc}")
                nvtx_load_status = "FAILED"
                overall_status = "PARTIAL"
        if attribute_nvtx:
            if nvtx_load_status == "FAILED":
                attribution_status = "FAILED"
            elif not api_events:
                attribution_status = "PASS_WITH_WARNINGS"
                warnings.append(
                    "CUDA API events are unavailable; NVTX attribution is limited to LOW-confidence fallback"
                )
                events, counters = _attribute_nvtx(events, api_events, nvtx_ranges)
            else:
                events, counters = _attribute_nvtx(events, api_events, nvtx_ranges)
                attribution_status = "PASS"
        if not events:
            missing.append(f"kernel event table {table} contains no events")
        return EventExtraction(
            events,
            api_events,
            nvtx_ranges,
            inventory,
            missing,
            mapping,
            nvtx_stats=nvtx_stats,
            warnings=warnings,
            kernel_event_status="PASS" if events else "FAILED",
            nvtx_load_status=nvtx_load_status,
            nvtx_attribution_status=attribution_status,
            overall_status=overall_status,
            attribution_query_count=counters.queries,
            attribution_candidate_checks=counters.candidate_checks,
        )
    finally:
        connection.close()


def load_device_metadata(path: Path) -> Dict[int, str]:
    connection = _connect_read_only(path)
    try:
        inventory = inspect_schema(connection)
        table = _table(inventory, ("TARGET_INFO_GPU", "GPU_INFO"))
        if not table:
            return {}
        columns = inventory.tables[table]
        device = _column(columns, "deviceId", "id", "device")
        name = _column(columns, "deviceName", "name")
        if not device or not name:
            return {}
        return {
            int(key): str(value)
            for key, value in connection.execute(
                f"SELECT {_quote(device)}, {_quote(name)} FROM {_quote(table)}"
            )
        }
    finally:
        connection.close()


def load_cuda_api_events(path: Path) -> List[CudaApiEvent]:
    connection = _connect_read_only(path)
    try:
        inventory = inspect_schema(connection)
        return _load_cuda_api_events(connection, inventory, _strings(connection, inventory))
    finally:
        connection.close()


def load_nvtx_ranges(path: Path) -> NvtxLoadResult:
    connection = _connect_read_only(path)
    try:
        inventory = inspect_schema(connection)
        return _load_nvtx_ranges(connection, inventory, _strings(connection, inventory))
    finally:
        connection.close()


def load_memory_events(path: Path) -> List[MemoryEvent]:
    connection = _connect_read_only(path)
    try:
        inventory = inspect_schema(connection)
        output: List[MemoryEvent] = []
        event_id = 0
        for table, columns in inventory.tables.items():
            normalized = normalize_header(table)
            if "memcpy" not in normalized and "memset" not in normalized:
                continue
            start = _column(columns, "start", "startNs", "timestamp")
            end = _column(columns, "end", "endNs")
            duration = _column(columns, "duration", "durationNs")
            device = _column(columns, "deviceId", "device", "gpuId")
            kind = _column(columns, "copyKind", "kind", "memoryKind")
            size = _column(columns, "bytes", "size", "numBytes")
            if not start or (not end and not duration) or not device:
                continue
            selected = [start, end or duration, device]
            if kind:
                selected.append(kind)
            if size:
                selected.append(size)
            for row in connection.execute(
                "SELECT " + ", ".join(_quote(value) for value in selected)
                + f" FROM {_quote(table)} ORDER BY {_quote(start)}"
            ):
                raw_start = int(row[0])
                second = int(row[1])
                raw_end = second if end else raw_start + second
                position = 3
                raw_kind = str(row[position]) if kind else table
                if kind:
                    position += 1
                raw_size = _int_or_none(row[position]) if size else None
                output.append(
                    MemoryEvent(
                        event_id, raw_start, raw_end, int(row[2]), raw_kind,
                        raw_size, table,
                    )
                )
                event_id += 1
        return output
    finally:
        connection.close()


def _cross_stream_overlaps(events: Sequence[KernelEvent]) -> List[dict]:
    rows = []
    ordered = sorted(events, key=lambda row: (row.device_id, row.start_ns, row.end_ns))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if right.device_id != left.device_id:
                if right.device_id > left.device_id:
                    break
                continue
            if right.start_ns >= left.end_ns:
                break
            if right.stream_id == left.stream_id:
                continue
            overlap = min(left.end_ns, right.end_ns) - max(left.start_ns, right.start_ns)
            if overlap > 0:
                rows.append(
                    {
                        "device_id": left.device_id,
                        "left_stream_id": left.stream_id,
                        "right_stream_id": right.stream_id,
                        "left_kernel": left.name,
                        "right_kernel": right.name,
                        "overlap_ns": overlap,
                    }
                )
    return rows


def write_event_artifacts(extraction: EventExtraction, output_dir: Path) -> None:
    event_fields = tuple(KernelEvent.__dataclass_fields__)
    rows = (
        {name: getattr(event, name) for name in event_fields}
        for event in extraction.events
    )
    write_csv(output_dir / "kernel_events.csv", event_fields, rows)
    write_csv(
        output_dir / "stream_timeline.csv",
        event_fields,
        (
            {name: getattr(event, name) for name in event_fields}
            for event in extraction.events
        ),
    )
    api_fields = tuple(CudaApiEvent.__dataclass_fields__)
    write_csv(
        output_dir / "cuda_api_events.csv",
        api_fields,
        (
            {name: getattr(row, name) for name in api_fields}
            for row in extraction.api_events
        ),
    )
    nvtx_fields = tuple(NvtxRange.__dataclass_fields__)
    write_csv(
        output_dir / "nvtx_ranges.csv",
        nvtx_fields,
        (
            {name: getattr(row, name) for name in nvtx_fields}
            for row in extraction.nvtx_ranges
        ),
    )
    overlap_fields = (
        "device_id", "left_stream_id", "right_stream_id", "left_kernel",
        "right_kernel", "overlap_ns",
    )
    write_csv(
        output_dir / "cross_stream_overlap.csv",
        overlap_fields,
        _cross_stream_overlaps(extraction.events),
    )
    write_event_metadata(extraction, output_dir)


def write_event_metadata(extraction: EventExtraction, output_dir: Path) -> None:
    status = {
        "base_stats_status": extraction.base_stats_status,
        "kernel_event_status": extraction.kernel_event_status,
        "nvtx_load_status": extraction.nvtx_load_status,
        "nvtx_attribution_status": extraction.nvtx_attribution_status,
        "dependency_analysis_status": extraction.dependency_analysis_status,
        "communication_analysis_status": extraction.communication_analysis_status,
        "overall_status": extraction.overall_status,
    }
    diagnostics = {
        **status,
        "nvtx_load_stats": asdict(extraction.nvtx_stats) if extraction.nvtx_stats else None,
        "warnings": extraction.warnings,
        "attribution_query_count": extraction.attribution_query_count,
        "attribution_candidate_checks": extraction.attribution_candidate_checks,
    }
    atomic_write_json(output_dir / "event_extraction_metadata.json", diagnostics)
    atomic_write_json(
        output_dir / "sqlite_schema.json",
        {
            "tables": {
                name: list(columns) for name, columns in extraction.inventory.tables.items()
            },
            "column_mapping": extraction.column_mapping,
            "missing_capabilities": extraction.missing_capabilities,
            **diagnostics,
        },
    )


def write_kernel_summary_from_events(
    events: Sequence[KernelEvent], output_path: Path
) -> None:
    grouped = defaultdict(list)
    for event in events:
        grouped[event.name].append(event.duration_ns)
    total_all = sum(sum(values) for values in grouped.values())
    rows = []
    for name, durations in grouped.items():
        total = sum(durations)
        rows.append(
            {
                "Time (%)": total / total_all * 100.0 if total_all else 0.0,
                "Total Time (ns)": total,
                "Instances": len(durations),
                "Avg (ns)": total / len(durations),
                "Med (ns)": statistics.median(durations),
                "Min (ns)": min(durations),
                "Max (ns)": max(durations),
                "StdDev (ns)": statistics.pstdev(durations),
                "Name": name,
            }
        )
    rows.sort(key=lambda row: float(row["Total Time (ns)"]), reverse=True)
    fields = (
        "Time (%)", "Total Time (ns)", "Instances", "Avg (ns)", "Med (ns)",
        "Min (ns)", "Max (ns)", "StdDev (ns)", "Name",
    )
    write_csv(output_path, fields, rows)
