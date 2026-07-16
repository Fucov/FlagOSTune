"""Normalize native `nsys stats` CSV across minor schema changes."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional

from .models import KernelSummary
from .utils import (
    find_column,
    parse_duration_ns,
    parse_number,
    parse_optional_duration_ns,
)


NAME_ALIASES = ("Name", "Kernel Name", "Kernel", "Range")
TOTAL_ALIASES = (
    "Total Time (ns)", "Total Time (us)", "Total Time (ms)", "Total Time"
)
COUNT_ALIASES = ("Instances", "Num Calls", "Calls", "Count")
AVG_ALIASES = ("Avg (ns)", "Avg (us)", "Average (ns)", "Average", "Avg")
MEDIAN_ALIASES = ("Med (ns)", "Median (ns)", "Median")
MIN_ALIASES = ("Min (ns)", "Min")
MAX_ALIASES = ("Max (ns)", "Max")
STDDEV_ALIASES = ("StdDev (ns)", "StdDev", "Standard Deviation")


def _find_header(reader):
    for candidate in reader:
        if not candidate:
            continue
        try:
            find_column(candidate, NAME_ALIASES, required=True)
            find_column(candidate, TOTAL_ALIASES, required=True)
            find_column(candidate, COUNT_ALIASES, required=True)
            return candidate
        except ValueError:
            continue
    raise ValueError("required column missing from Nsight CSV header")


def load_kernel_summary(path: Path) -> List[KernelSummary]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw_reader = csv.reader(handle)
        fieldnames = _find_header(raw_reader)
        reader = csv.DictReader(handle, fieldnames=fieldnames)
        name_key = find_column(fieldnames, NAME_ALIASES, required=True)
        total_key = find_column(fieldnames, TOTAL_ALIASES, required=True)
        count_key = find_column(fieldnames, COUNT_ALIASES, required=True)
        avg_key = find_column(fieldnames, AVG_ALIASES)
        median_key = find_column(fieldnames, MEDIAN_ALIASES)
        min_key = find_column(fieldnames, MIN_ALIASES)
        max_key = find_column(fieldnames, MAX_ALIASES)
        stddev_key = find_column(fieldnames, STDDEV_ALIASES)
        parsed = []
        for raw in reader:
            name = str(raw.get(name_key, "")).strip()
            if not name:
                continue
            total_ns = parse_duration_ns(raw.get(total_key), total_key)
            parsed.append(
                KernelSummary(
                    name=name,
                    total_ns=total_ns,
                    instances=int(parse_number(raw.get(count_key))),
                    time_percentage=0.0,
                    avg_ns=parse_optional_duration_ns(raw.get(avg_key), avg_key),
                    median_ns=parse_optional_duration_ns(raw.get(median_key), median_key),
                    min_ns=parse_optional_duration_ns(raw.get(min_key), min_key),
                    max_ns=parse_optional_duration_ns(raw.get(max_key), max_key),
                    stddev_ns=parse_optional_duration_ns(raw.get(stddev_key), stddev_key),
                )
            )
    if not parsed:
        raise ValueError("Nsight report contains no CUDA kernel rows")
    denominator = sum(row.total_ns for row in parsed)
    if denominator <= 0:
        raise ValueError("Nsight report has zero total CUDA kernel time")
    normalized = [
        KernelSummary(
            **{
                **row.__dict__,
                "time_percentage": row.total_ns / denominator * 100.0,
            }
        )
        for row in parsed
    ]
    return sorted(normalized, key=lambda row: row.total_ns, reverse=True)
