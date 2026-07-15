#!/usr/bin/env python3
"""Print CUDA kernel, NVTX, and CUDA API summaries from an Nsight report."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SummaryRow:
    name: str
    total_ns: float
    instances: int
    avg_ns: float
    time_percentage: float


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def _number(value: str) -> float:
    text = value.strip().replace(",", "")
    return float(text or 0)


def _find_column(header: Sequence[str], aliases: Iterable[str]) -> Optional[int]:
    normalized = [normalize_header(item) for item in header]
    for alias in aliases:
        wanted = normalize_header(alias)
        if wanted in normalized:
            return normalized.index(wanted)
    return None


def parse_summary_csv(
    output: str,
    *,
    name_aliases: Sequence[str] = ("Name",),
) -> List[SummaryRow]:
    """Parse an Nsight summary CSV, tolerating informational preamble lines."""
    csv_rows = list(csv.reader(output.splitlines()))
    header_index = None
    columns: Optional[Tuple[int, int, int, int, Optional[int]]] = None

    for index, candidate in enumerate(csv_rows):
        name_col = _find_column(candidate, name_aliases)
        total_col = _find_column(candidate, ("Total Time (ns)", "Total Time"))
        calls_col = _find_column(candidate, ("Instances", "Num Calls", "Calls"))
        avg_col = _find_column(candidate, ("Avg (ns)", "Average (ns)", "Avg"))
        pct_col = _find_column(candidate, ("Time (%)", "Time %"))
        if None not in (name_col, total_col, calls_col, avg_col):
            header_index = index
            columns = (
                int(name_col),
                int(total_col),
                int(calls_col),
                int(avg_col),
                pct_col,
            )
            break

    if header_index is None or columns is None:
        raise ValueError("Nsight output does not contain a recognized summary header")

    name_col, total_col, calls_col, avg_col, pct_col = columns
    required_max = max(name_col, total_col, calls_col, avg_col, pct_col or 0)
    parsed: List[SummaryRow] = []
    for raw in csv_rows[header_index + 1 :]:
        if not raw or len(raw) <= required_max:
            continue
        try:
            parsed.append(
                SummaryRow(
                    name=raw[name_col].strip(),
                    total_ns=_number(raw[total_col]),
                    instances=int(_number(raw[calls_col])),
                    avg_ns=_number(raw[avg_col]),
                    time_percentage=_number(raw[pct_col]) if pct_col is not None else 0.0,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value in Nsight summary row: {raw}") from exc

    parsed.sort(key=lambda row: row.total_ns, reverse=True)
    return parsed


def kernel_rows_with_percentage(rows: Sequence[SummaryRow]) -> List[SummaryRow]:
    """Replace Nsight percentages with percentages of all returned kernel time."""
    if not rows:
        raise ValueError("Nsight report contains no CUDA kernel rows")
    total_ns = sum(row.total_ns for row in rows)
    if total_ns <= 0:
        raise ValueError("Nsight report has zero total CUDA kernel time")
    return [
        replace(row, time_percentage=(row.total_ns / total_ns) * 100.0)
        for row in rows
    ]


def run_nsys_report(
    report_path: Path,
    report_name: str,
    *,
    nsys: str = "nsys",
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> str:
    cmd = [
        nsys,
        "stats",
        "--report",
        report_name,
        "--format",
        "csv",
        "--force-export=true",
        str(report_path),
    ]
    try:
        result = runner(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Nsight Systems executable not found: {nsys}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"nsys stats report {report_name!r} failed: {detail}")
    return result.stdout


def render_table(title: str, rows: Sequence[SummaryRow], top: int) -> str:
    selected = list(rows[:top])
    headers = ("Time %", "Total ms", "Calls", "Avg us", "Name")
    values = [
        (
            f"{row.time_percentage:.2f}",
            f"{row.total_ns / 1_000_000:.3f}",
            str(row.instances),
            f"{row.avg_ns / 1_000:.3f}",
            row.name,
        )
        for row in selected
    ]
    if not values:
        return f"\n{title}\n(no rows)"

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in values))
        for index in range(len(headers))
    ]

    def format_row(row: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = "  ".join("-" * width for width in widths)
    body = "\n".join(format_row(row) for row in values)
    return f"\n{title}\n{format_row(headers)}\n{separator}\n{body}"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="input .nsys-rep file")
    parser.add_argument("--top", type=positive_int, default=10, help="rows per table")
    parser.add_argument("--nsys", default="nsys", help="nsys executable path")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report: Path = args.report
    if not report.is_file():
        print(f"[ERROR] Nsight report does not exist: {report}", file=sys.stderr)
        return 2
    if not report.name.endswith(".nsys-rep"):
        print(f"[ERROR] expected a .nsys-rep file: {report}", file=sys.stderr)
        return 2

    try:
        kernel_rows = parse_summary_csv(
            run_nsys_report(report, "cuda_gpu_kern_sum", nsys=args.nsys),
            name_aliases=("Name",),
        )
        kernel_rows = kernel_rows_with_percentage(kernel_rows)
        nvtx_rows = parse_summary_csv(
            run_nsys_report(report, "nvtx_sum", nsys=args.nsys),
            name_aliases=("Range", "Name"),
        )
        api_rows = parse_summary_csv(
            run_nsys_report(report, "cuda_api_sum", nsys=args.nsys),
            name_aliases=("Name",),
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(render_table("Top CUDA Kernels", kernel_rows, args.top))
    print(render_table("NVTX Range Summary", nvtx_rows, args.top))
    print(render_table("CUDA API Summary", api_rows, args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
