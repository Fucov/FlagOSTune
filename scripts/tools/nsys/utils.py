"""Small filesystem, CSV, and formatting utilities."""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def find_column(
    fieldnames: Sequence[str], aliases: Iterable[str], required: bool = False
) -> Optional[str]:
    normalized = {normalize_header(value): value for value in fieldnames if value}
    for alias in aliases:
        match = normalized.get(normalize_header(alias))
        if match is not None:
            return match
    if required:
        raise ValueError(
            "required column missing; expected one of " + ", ".join(aliases)
        )
    return None


def parse_number(value: object) -> float:
    if value is None:
        raise ValueError("numeric value is missing")
    text = str(value).strip().replace(",", "")
    if not text:
        raise ValueError("numeric value is empty")
    return float(text)


def parse_optional_number(value: object) -> Optional[float]:
    if value is None or not str(value).strip():
        return None
    return parse_number(value)


def parse_duration_ns(value: object, header: str) -> float:
    number = parse_number(value)
    lowered = header.lower()
    if "(s)" in lowered or lowered.endswith(" s"):
        return number * 1_000_000_000
    if "ms" in lowered:
        return number * 1_000_000
    if "us" in lowered or "µs" in lowered:
        return number * 1_000
    return number


def parse_optional_duration_ns(value: object, header: Optional[str]) -> Optional[float]:
    if header is None or value is None or not str(value).strip():
        return None
    return parse_duration_ns(value, header)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def atomic_write_json(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "N/A"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.1f} {unit}"
        number /= 1024
    return f"{number:.1f} TiB"


def read_csv_rows(path: Path) -> List[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    header_index = None
    for index, row in enumerate(rows):
        if len(row) < 2:
            continue
        joined = " ".join(row).strip().lower()
        if joined.startswith("processing [") or " with [" in joined:
            continue
        header_index = index
        break
    if header_index is None:
        return []
    header = [str(value).strip() for value in rows[header_index]]
    output = []
    for row in rows[header_index + 1 :]:
        if not row or not any(str(value).strip() for value in row):
            continue
        if len(row) < len(header):
            row = list(row) + [""] * (len(header) - len(row))
        converted = {}
        for position, name in enumerate(header):
            value = row[position].strip()
            normalized_name = normalize_header(name)
            numeric_header = any(
                token in normalized_name
                for token in (
                    "time", "duration", "instance", "calls", "count", "size",
                    "bytes", "average", "avg", "median", "minimum", "maximum",
                    "stddev", "percent", "deviceid", "contextid", "streamid",
                )
            )
            numeric_text = value.replace(",", "")
            if numeric_header and re.fullmatch(r"[-+]?[0-9]+", numeric_text):
                converted[name] = int(numeric_text)
            elif numeric_header and re.fullmatch(
                r"[-+]?(?:[0-9]+\.[0-9]*|[0-9]*\.[0-9]+)(?:[eE][-+]?[0-9]+)?",
                numeric_text,
            ):
                converted[name] = float(numeric_text)
            else:
                converted[name] = value
        output.append(converted)
    return output
