"""Read-only, schema-tolerant GPU and process aggregation from Nsight SQLite."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .classify_kernels import base_family, classify_kernel
from .models import DeviceSummary, WarningRecord
from .utils import normalize_header, write_csv


def _tables(connection: sqlite3.Connection) -> Dict[str, str]:
    return {
        normalize_header(name): name
        for (name,) in connection.execute("select name from sqlite_master where type='table'")
    }


def _columns(connection: sqlite3.Connection, table: str) -> Dict[str, str]:
    escaped = table.replace('"', '""')
    return {
        normalize_header(row[1]): row[1]
        for row in connection.execute(f'pragma table_info("{escaped}")')
    }


def _column(columns: Dict[str, str], *aliases: str) -> Optional[str]:
    for alias in aliases:
        value = columns.get(normalize_header(alias))
        if value:
            return value
    return None


def _quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _gpu_names(connection: sqlite3.Connection, tables: Dict[str, str]) -> Dict[int, str]:
    table = tables.get(normalize_header("TARGET_INFO_GPU"))
    if not table:
        return {}
    columns = _columns(connection, table)
    id_column = _column(columns, "id", "deviceId", "device")
    name_column = _column(columns, "name", "deviceName")
    if not id_column or not name_column:
        return {}
    return {
        int(device): str(name)
        for device, name in connection.execute(
            f"select {_quote(id_column)}, {_quote(name_column)} from {_quote(table)}"
        )
    }


def _string_ids(connection: sqlite3.Connection, tables: Dict[str, str]) -> Dict[int, str]:
    table = tables.get(normalize_header("StringIds"))
    if not table:
        return {}
    columns = _columns(connection, table)
    id_column = _column(columns, "id")
    value_column = _column(columns, "value", "string")
    if not id_column or not value_column:
        return {}
    return {
        int(key): str(value)
        for key, value in connection.execute(
            f"select {_quote(id_column)}, {_quote(value_column)} from {_quote(table)}"
        )
    }


def analyze_devices(
    sqlite_path: Path, expected_tp: Optional[int] = None
) -> Tuple[List[DeviceSummary], List[WarningRecord], bool]:
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        tables = _tables(connection)
        table = tables.get(normalize_header("CUPTI_ACTIVITY_KIND_KERNEL"))
        if not table:
            return [], [WarningRecord("devices", "kernel event table is unavailable")], False
        columns = _columns(connection, table)
        device_col = _column(columns, "deviceId", "device")
        pid_col = _column(columns, "globalPid", "pid", "processId")
        start_col = _column(columns, "start", "startNs", "timestamp")
        end_col = _column(columns, "end", "endNs")
        name_col = _column(columns, "name", "kernelName", "nameId")
        if not all((device_col, start_col, end_col, name_col)):
            return [], [WarningRecord("devices", "kernel event table is missing required columns")], False
        selected = [device_col, pid_col or device_col, start_col, end_col, name_col]
        query = f"select {', '.join(_quote(value) for value in selected)} from {_quote(table)}"
        strings = _string_ids(connection, tables)
        gpu_names = _gpu_names(connection, tables)
        by_device = defaultdict(lambda: {"pids": set(), "events": 0, "time": 0.0, "comm": 0.0, "families": defaultdict(float)})
        name_is_id = normalize_header(name_col) == normalize_header("nameId")
        for device, pid, start, end, raw_name in connection.execute(query):
            device_id = int(device)
            duration = max(0.0, float(end) - float(start))
            name = strings.get(int(raw_name), str(raw_name)) if name_is_id else str(raw_name)
            classification = classify_kernel(name)
            values = by_device[device_id]
            values["pids"].add(int(pid))
            values["events"] += 1
            values["time"] += duration
            values["families"][base_family(name)] += duration
            if classification.runtime_communication:
                values["comm"] += duration
        totals = [float(value["time"]) for value in by_device.values()]
        mean = sum(totals) / len(totals) if totals else 0.0
        imbalance = ((max(totals) - min(totals)) / mean) if mean and totals else None
        summaries = []
        for device_id in sorted(by_device):
            values = by_device[device_id]
            total = float(values["time"])
            comm = float(values["comm"])
            top = max(values["families"], key=values["families"].get) if values["families"] else None
            summaries.append(
                DeviceSummary(
                    device_id=device_id,
                    process_count=len(values["pids"]),
                    kernel_events=int(values["events"]),
                    kernel_time_ns=total,
                    compute_time_ns=total - comm,
                    communication_time_ns=comm,
                    communication_percentage=(comm / total * 100.0) if total else None,
                    top_family=top,
                    relative_time=(total / mean) if mean else None,
                    imbalance=imbalance,
                    gpu_name=gpu_names.get(device_id),
                )
            )
        warnings = []
        integrity = True
        if expected_tp is not None and len(summaries) != expected_tp:
            integrity = False
            warnings.append(
                WarningRecord(
                    "devices",
                    f"expected TP{expected_tp} but captured {len(summaries)} GPU(s); check --trace-fork-before-exec=true",
                )
            )
        return summaries, warnings, integrity
    finally:
        connection.close()


def write_device_summary(rows: List[DeviceSummary], output_dir: Path) -> None:
    fields = tuple(DeviceSummary.__dataclass_fields__)
    write_csv(output_dir / "device_summary.csv", fields, [row.__dict__ for row in rows])
