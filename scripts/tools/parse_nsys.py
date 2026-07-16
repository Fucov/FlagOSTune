#!/usr/bin/env python3
"""Export and analyze an Nsight Systems report with observable progress."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.tools.nsys.analyze_communication import (
    analyze_communication,
    build_communication_chains,
    build_fusion_candidates,
    write_communication_analysis,
)
from scripts.tools.nsys.analyze_dependencies import build_adjacency, load_trace_csv
from scripts.tools.nsys.analyze_devices import analyze_devices, write_device_summary
from scripts.tools.nsys.analyze_phases import attribute_phase
from scripts.tools.nsys.classify_kernels import classify_kernels, write_classification
from scripts.tools.nsys.collect_stats import (
    CoreReportError,
    collect_reports,
    detect_supported_reports,
    select_reports,
)
from scripts.tools.nsys.export_report import ExportError, resolve_sqlite
from scripts.tools.nsys.models import AnalysisData, KernelSummary, WarningRecord
from scripts.tools.nsys.normalize_stats import load_kernel_summary
from scripts.tools.nsys.progress import ProgressReporter, run_streaming_command
from scripts.tools.nsys.render_markdown import render_markdown
from scripts.tools.nsys.utils import atomic_write_json, normalize_header, read_csv_rows


# Legacy parsing records/helpers remain import-compatible for callers and tests.
SummaryRow = KernelSummary


def _number(value: str) -> float:
    return float(value.strip().replace(",", "") or 0)


def _find_column(header: Sequence[str], aliases: Iterable[str]) -> Optional[int]:
    normalized = [normalize_header(item) for item in header]
    for alias in aliases:
        wanted = normalize_header(alias)
        if wanted in normalized:
            return normalized.index(wanted)
    return None


def parse_summary_csv(
    output: str, *, name_aliases: Sequence[str] = ("Name",)
) -> List[SummaryRow]:
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
            columns = (int(name_col), int(total_col), int(calls_col), int(avg_col), pct_col)
            break
    if header_index is None or columns is None:
        raise ValueError("Nsight output does not contain a recognized summary header")
    name_col, total_col, calls_col, avg_col, pct_col = columns
    required_max = max(name_col, total_col, calls_col, avg_col, pct_col or 0)
    parsed = []
    for raw in csv_rows[header_index + 1 :]:
        if not raw or len(raw) <= required_max:
            continue
        parsed.append(
            SummaryRow(
                name=raw[name_col].strip(),
                total_ns=_number(raw[total_col]),
                instances=int(_number(raw[calls_col])),
                avg_ns=_number(raw[avg_col]),
                time_percentage=_number(raw[pct_col]) if pct_col is not None else 0.0,
            )
        )
    return sorted(parsed, key=lambda row: row.total_ns, reverse=True)


def kernel_rows_with_percentage(rows: Sequence[SummaryRow]) -> List[SummaryRow]:
    if not rows:
        raise ValueError("Nsight report contains no CUDA kernel rows")
    total_ns = sum(row.total_ns for row in rows)
    if total_ns <= 0:
        raise ValueError("Nsight report has zero total CUDA kernel time")
    return [replace(row, time_percentage=row.total_ns / total_ns * 100.0) for row in rows]


def run_nsys_report(
    report_path: Path,
    report_name: str,
    *,
    nsys: str = "nsys",
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> str:
    """Compatibility helper; new orchestration uses streamed CSV files."""
    command = [nsys, "stats", "--report", report_name, "--format", "csv", str(report_path)]
    try:
        result = runner(command, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Nsight Systems executable not found: {nsys}") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"nsys stats report {report_name!r} failed: {detail}")
    return result.stdout


def render_table(title: str, rows: Sequence[SummaryRow], top: int) -> str:
    values = [
        f"{row.time_percentage:.2f}% {row.total_ns / 1_000_000:.3f}ms "
        f"{row.instances} {row.avg_ns / 1_000:.3f}us {row.name}"
        for row in rows[:top]
    ]
    return "\n" + title + "\n" + ("\n".join(values) if values else "(no rows)")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="input .nsys-rep or .sqlite file")
    parser.add_argument("--output-dir", type=Path, help="summary output directory")
    parser.add_argument("--force-export", action="store_true", help="re-export SQLite even when cache is current")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--reuse-sqlite", dest="reuse_sqlite", action="store_true", help="reuse a current adjacent SQLite (default)")
    reuse.add_argument("--no-reuse-sqlite", dest="reuse_sqlite", action="store_false", help="bypass the SQLite cache")
    parser.set_defaults(reuse_sqlite=True)
    parser.add_argument("--top", type=positive_int, default=20, help="rows per Markdown table")
    parser.add_argument("--reports", help="comma-separated native Nsight reports")
    parser.add_argument("--nsys", default="nsys", help="nsys executable path")
    parser.add_argument("--analyze-dependencies", action="store_true")
    parser.add_argument("--analyze-communication", action="store_true")
    parser.add_argument("--phase-log", type=Path)
    parser.add_argument("--phase-metadata", type=Path)
    return parser.parse_args(argv)


def _load_capture_metadata(report: Path, explicit: Optional[Path]) -> dict:
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(
        (
            Path(str(report) + ".metadata.json"),
            report.with_name(report.name.replace(".nsys-rep", "") + ".metadata.json"),
        )
    )
    for candidate in candidates:
        if candidate and candidate.is_file():
            try:
                value = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    value["capture_metadata_path"] = str(candidate)
                    return value
            except (OSError, ValueError) as exc:
                raise ValueError(f"invalid capture metadata {candidate}: {exc}") from exc
    return {}


def _nsys_version(nsys: str, output_dir: Path, progress: ProgressReporter) -> str:
    command = [nsys, "--version"]
    temporary = output_dir / ".nsys-version.txt"
    started = progress.begin("Read Nsight version", command=command, output_path=temporary)
    code = run_streaming_command(command, temporary, output_dir / "progress.log", progress)
    if code != 0:
        progress.finish("Read Nsight version", started, "FAILED", output_path=temporary)
        raise RuntimeError(f"nsys --version failed with exit {code}")
    version = temporary.read_text(encoding="utf-8", errors="replace").strip() or "N/A"
    temporary.unlink(missing_ok=True)
    progress.finish("Read Nsight version", started, detail=version)
    return version


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).astimezone().isoformat()


def _metadata(
    args: argparse.Namespace,
    report: Path,
    sqlite_path: Path,
    version: str,
    capture: dict,
    collection,
    devices,
    integrity: bool,
) -> dict:
    value = dict(capture)
    value.update(
        {
            "input_report": str(report),
            "input_size": report.stat().st_size,
            "input_mtime": _mtime(report),
            "sqlite_path": str(sqlite_path),
            "sqlite_size": sqlite_path.stat().st_size,
            "nsys_version": version,
            "command_line": sys.argv,
            "generated_time": datetime.now(timezone.utc).astimezone().isoformat(),
            "successful_reports": list(collection.successful),
            "unsupported_reports": collection.unsupported,
            "failed_reports": collection.failed,
            "empty_reports": collection.empty,
            "captured_gpu_count": len(devices),
            "captured_devices": [row.__dict__ for row in devices],
            "integrity_ok": integrity,
        }
    )
    return value


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = args.report.absolute()
    output_dir = (args.output_dir or report.parent / "summary").absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "export_sqlite.log").touch(exist_ok=True)
    selected = select_reports(args.reports, args.analyze_dependencies, args.analyze_communication)
    total_stages = 7 + len(selected) + (3 if args.analyze_dependencies or args.analyze_communication else 1)
    progress = ProgressReporter(total_stages, log_path=output_dir / "progress.log")
    try:
        capture = _load_capture_metadata(report, args.phase_metadata)
        sqlite_path = resolve_sqlite(
            report,
            output_dir,
            args.nsys,
            force_export=args.force_export,
            reuse_sqlite=args.reuse_sqlite,
            progress=progress,
        )
        version = _nsys_version(args.nsys, output_dir, progress)
        detection_warnings = []
        try:
            supported = detect_supported_reports(
                args.nsys, output_dir, progress, warnings=detection_warnings
            )
        except CoreReportError as exc:
            message = (
                f"{exc}; falling back to direct report probes against the existing SQLite"
            )
            detection_warnings.append(WarningRecord("detect_reports", message))
            progress.warning(message)
            supported = None
        collection = collect_reports(
            sqlite_path, selected, supported, args.nsys, output_dir, progress
        )
        collection.warnings.extend(detection_warnings)

        started = progress.begin("Normalize and classify kernel reports")
        kernels = load_kernel_summary(collection.successful["cuda_gpu_kern_sum"])
        base_path = collection.successful.get("cuda_gpu_kern_sum:base")
        base_kernels = load_kernel_summary(base_path) if base_path else kernels
        classified = classify_kernels(kernels)
        write_classification(classified, output_dir)
        progress.finish("Normalize and classify kernel reports", started, output_path=output_dir / "kernel_classification.csv")

        expected_tp = capture.get("tp_size") or capture.get("tensor_parallel_size")
        expected_tp = int(expected_tp) if expected_tp not in (None, "", "null") else None
        started = progress.begin("Analyze captured devices", input_path=sqlite_path)
        devices, device_warnings, integrity = analyze_devices(sqlite_path, expected_tp)
        write_device_summary(devices, output_dir)
        progress.finish("Analyze captured devices", started, output_path=output_dir / "device_summary.csv")

        nvtx_rows = read_csv_rows(collection.successful["nvtx_sum"]) if "nvtx_sum" in collection.successful else []
        nvtx_names = [str(row.get("Range") or row.get("Name") or "") for row in nvtx_rows]
        phase = attribute_phase(capture, nvtx_names, args.phase_log)
        events = []
        adjacency = []
        communication_events = []
        chains = []
        candidates = []
        trace_path = collection.successful.get("cuda_gpu_trace:base")
        if (args.analyze_dependencies or args.analyze_communication) and trace_path:
            started = progress.begin("Analyze event dependencies and communication", input_path=trace_path)
            events = [replace(row, phase=phase.phase) for row in load_trace_csv(trace_path)]
            adjacency = build_adjacency(events)
            communication_events = analyze_communication(events)
            chains = build_communication_chains(events, adjacency, communication_events)
            candidates = build_fusion_candidates(communication_events, chains)
            write_communication_analysis(adjacency, communication_events, chains, candidates, output_dir)
            progress.finish("Analyze event dependencies and communication", started, output_path=output_dir / "communication_events.csv")
        elif args.analyze_dependencies or args.analyze_communication:
            warning = WarningRecord("dependencies", "cuda_gpu_trace:base is unavailable; event analysis was not generated")
            collection.warnings.append(warning)
            progress.warning(warning.message)

        native_tables = {
            name: read_csv_rows(path)
            for name, path in collection.successful.items()
            if name not in ("cuda_gpu_kern_sum", "cuda_gpu_kern_sum:base")
        }
        metadata = _metadata(args, report, sqlite_path, version, capture, collection, devices, integrity)
        metadata["profile_phase_attribution"] = phase.__dict__
        metadata["warnings"] = [row.__dict__ for row in collection.warnings + device_warnings]
        started = progress.begin("Write metadata", output_path=output_dir / "metadata.json")
        atomic_write_json(output_dir / "metadata.json", metadata)
        progress.finish("Write metadata", started, output_path=output_dir / "metadata.json")

        data = AnalysisData(
            metadata=metadata,
            reports=collection,
            kernels=kernels,
            base_kernels=base_kernels,
            classified=classified,
            devices=devices,
            adjacency=adjacency,
            communication_events=communication_events,
            communication_chains=chains,
            fusion_candidates=candidates,
            phase_attribution=phase,
            native_tables=native_tables,
            warnings=device_warnings,
        )
        started = progress.begin("Render Markdown report", output_path=output_dir / "nsys_analysis.md")
        markdown = render_markdown(data, args.top, output_dir / "nsys_analysis.md")
        progress.finish("Render Markdown report", started, output_path=output_dir / "nsys_analysis.md")
        sys.stdout.write(markdown)
        return 0
    except KeyboardInterrupt:
        progress.emit("[ERROR] interrupted by user; active child terminated; original report preserved")
        return 130
    except (CoreReportError, ExportError, OSError, RuntimeError, ValueError) as exc:
        progress.emit(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
