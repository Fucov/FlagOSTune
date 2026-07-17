"""Schema-tolerant, read-only extraction of Nsight SQLite GPU events."""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass, replace
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
    event_id: int
    start_ns: int
    end_ns: int
    process_id: Optional[int]
    thread_id: Optional[int]
    correlation_id: Optional[int]
    name: str
    source_table: str


@dataclass(frozen=True)
class NvtxRange:
    range_id: int
    start_ns: int
    end_ns: int
    process_id: Optional[int]
    thread_id: Optional[int]
    text: str
    source_table: str


@dataclass(frozen=True)
class EventExtraction:
    events: List[KernelEvent]
    api_events: List[CudaApiEvent]
    nvtx_ranges: List[NvtxRange]
    inventory: SchemaInventory
    missing_capabilities: List[str]
    column_mapping: Dict[str, Optional[str]]


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
            output.append(
                CudaApiEvent(
                    event_id,
                    int(raw["start"]),
                    int(raw["end"]),
                    _int_or_none(raw.get("process")),
                    _int_or_none(raw.get("thread")),
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
) -> List[NvtxRange]:
    table = _table(inventory, ("NVTX_EVENTS", "NVTX_EVENT", "NVTX_RANGES"))
    if not table:
        return []
    columns = inventory.tables[table]
    mapping = {
        "start": _column(columns, "start", "startNs", "timestamp"),
        "end": _column(columns, "end", "endNs"),
        "process": _column(columns, "globalPid", "processId", "pid"),
        "thread": _column(columns, "globalTid", "threadId", "tid"),
        "text": _column(columns, "text", "message", "name"),
        "text_id": _column(columns, "textId", "messageId", "nameId"),
    }
    if not mapping["start"] or not mapping["end"] or not (
        mapping["text"] or mapping["text_id"]
    ):
        return []
    keys = [key for key, value in mapping.items() if value]
    query = "SELECT " + ", ".join(
        _quote(str(mapping[key])) for key in keys
    ) + f" FROM {_quote(table)} ORDER BY {_quote(str(mapping['start']))}"
    output = []
    for range_id, values in enumerate(connection.execute(query)):
        raw = dict(zip(keys, values))
        if mapping["text"]:
            text = str(raw.get("text") or "")
        else:
            raw_id = _int_or_none(raw.get("text_id"))
            text = strings.get(raw_id, str(raw.get("text_id") or "Unknown"))
        output.append(
            NvtxRange(
                range_id,
                int(raw["start"]),
                int(raw["end"]),
                _int_or_none(raw.get("process")),
                _int_or_none(raw.get("thread")),
                text,
                table,
            )
        )
    return output


def _attribute_nvtx(
    events: Sequence[KernelEvent], ranges: Sequence[NvtxRange]
) -> List[KernelEvent]:
    output = []
    for event in events:
        candidates = []
        for nvtx in ranges:
            if not (nvtx.start_ns <= event.start_ns and nvtx.end_ns >= event.end_ns):
                continue
            if (
                event.process_id is not None
                and nvtx.process_id is not None
                and event.process_id != nvtx.process_id
            ):
                continue
            if (
                event.thread_id is not None
                and nvtx.thread_id is not None
                and event.thread_id != nvtx.thread_id
            ):
                continue
            candidates.append(nvtx)
        if not candidates:
            output.append(event)
            continue
        selected = min(candidates, key=lambda row: row.end_ns - row.start_ns)
        classification = classify_kernel(
            event.name, nvtx=selected.text, module=event.module
        )
        output.append(
            replace(
                event,
                nvtx_range=selected.text,
                category=classification.category,
                classification_rule=classification.rule,
                classification_confidence=classification.confidence,
            )
        )
    return output


def load_kernel_events(path: Path) -> EventExtraction:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        inventory = inspect_schema(connection)
        table = _table(
            inventory,
            ("CUPTI_ACTIVITY_KIND_KERNEL", "CUDA_GPU_KERNEL", "KERNEL_EVENTS"),
        )
        if not table:
            return EventExtraction(
                [], [], [], inventory, ["kernel event table is unavailable"], {}
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
            return EventExtraction([], [], [], inventory, missing, mapping)

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
                    process_id=_int_or_none(raw.get("process")),
                    thread_id=_int_or_none(raw.get("thread")),
                    correlation_id=_int_or_none(raw.get("correlation")),
                    classification_rule=classification.rule,
                    classification_confidence=classification.confidence,
                    source_table=table,
                )
            )
        api_events = _load_cuda_api_events(connection, inventory, strings)
        nvtx_ranges = _load_nvtx_ranges(connection, inventory, strings)
        events = _attribute_nvtx(events, nvtx_ranges)
        if not events:
            missing.append(f"kernel event table {table} contains no events")
        return EventExtraction(
            events, api_events, nvtx_ranges, inventory, missing, mapping
        )
    finally:
        connection.close()


def load_device_metadata(path: Path) -> Dict[int, str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
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
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        inventory = inspect_schema(connection)
        return _load_cuda_api_events(connection, inventory, _strings(connection, inventory))
    finally:
        connection.close()


def load_nvtx_ranges(path: Path) -> List[NvtxRange]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        inventory = inspect_schema(connection)
        return _load_nvtx_ranges(connection, inventory, _strings(connection, inventory))
    finally:
        connection.close()


def load_memory_events(path: Path) -> List[MemoryEvent]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
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
    rows = [event.__dict__ for event in extraction.events]
    write_csv(output_dir / "kernel_events.csv", event_fields, rows)
    write_csv(output_dir / "stream_timeline.csv", event_fields, rows)
    api_fields = tuple(CudaApiEvent.__dataclass_fields__)
    write_csv(
        output_dir / "cuda_api_events.csv",
        api_fields,
        [row.__dict__ for row in extraction.api_events],
    )
    nvtx_fields = tuple(NvtxRange.__dataclass_fields__)
    write_csv(
        output_dir / "nvtx_ranges.csv",
        nvtx_fields,
        [row.__dict__ for row in extraction.nvtx_ranges],
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
    atomic_write_json(
        output_dir / "sqlite_schema.json",
        {
            "tables": {
                name: list(columns) for name, columns in extraction.inventory.tables.items()
            },
            "column_mapping": extraction.column_mapping,
            "missing_capabilities": extraction.missing_capabilities,
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
