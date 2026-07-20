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
from scripts.tools.nsys.classify_kernels import (
    build_operator_hotspots,
    classify_kernel,
    classify_kernels,
    write_classification,
    write_operator_hotspots,
)
from scripts.tools.nsys.collect_stats import (
    CoreReportError,
    collect_reports,
    detect_supported_reports,
    select_reports,
)
from scripts.tools.nsys.export_report import ExportError, resolve_sqlite
from scripts.tools.nsys.evaluate_integrity import IntegrityInputs, evaluate_integrity
from scripts.tools.nsys.models import AnalysisData, KernelSummary, WarningRecord
from scripts.tools.nsys.normalize_stats import load_kernel_summary
from scripts.tools.nsys.progress import ProgressReporter, run_streaming_command
from scripts.tools.nsys.render_markdown import render_markdown
from scripts.tools.nsys.sqlite_events import (
    load_kernel_events,
    load_memory_events,
    write_event_artifacts,
    write_event_metadata,
    write_kernel_summary_from_events,
)
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


def _optional_float(value: object) -> Optional[float]:
    if value in (None, "", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    parser.add_argument(
        "--analyze-nvtx-attribution",
        action="store_true",
        help="load NVTX intervals and attribute kernels through CUDA API correlation",
    )
    parser.add_argument(
        "--reuse-existing-stats",
        action="store_true",
        help="reuse valid CSV reports already present in --output-dir",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume analysis and reuse valid existing CSV reports",
    )
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
            "report_sources": collection.selected_sources,
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
        if args.resume:
            started = progress.begin("Detect supported reports")
            supported = None
            progress.finish(
                "Detect supported reports",
                started,
                "REUSED",
                detail="resume mode probes only missing CSV reports",
            )
        else:
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
            sqlite_path,
            selected,
            supported,
            args.nsys,
            output_dir,
            progress,
            allow_core_fallback=True,
            reuse_existing=args.reuse_existing_stats or args.resume,
        )
        collection.warnings.extend(detection_warnings)

        started = progress.begin("Inspect SQLite event schema", input_path=sqlite_path)
        request_event_enrichment = bool(
            args.analyze_dependencies
            or args.analyze_communication
            or args.analyze_nvtx_attribution
        )
        extraction = load_kernel_events(
            sqlite_path,
            include_cuda_api=request_event_enrichment,
            include_nvtx=request_event_enrichment,
            attribute_nvtx=request_event_enrichment,
        )
        base_stats_status = (
            "REUSED"
            if collection.successful
            and all(
                collection.selected_sources.get(name) == "existing CSV"
                for name in collection.successful
            )
            else "PASS"
        )
        extraction = replace(extraction, base_stats_status=base_stats_status)
        write_event_artifacts(extraction, output_dir)
        for message in extraction.missing_capabilities:
            collection.warnings.append(WarningRecord("sqlite_events", message))
            progress.warning(message)
        for message in extraction.warnings:
            collection.warnings.append(WarningRecord("sqlite_events", message))
            progress.warning(message)
        progress.finish(
            "Inspect SQLite event schema",
            started,
            "WARNING" if extraction.missing_capabilities or extraction.warnings else "SUCCESS",
            output_path=output_dir / "sqlite_schema.json",
        )
        if "cuda_gpu_kern_sum" not in collection.successful:
            if not extraction.events:
                raise CoreReportError(
                    "cuda_gpu_kern_sum is unavailable and direct SQLite query "
                    "did not find kernel events"
                )
            sqlite_summary = output_dir / "cuda_gpu_kern_sum.csv"
            write_kernel_summary_from_events(extraction.events, sqlite_summary)
            collection.successful["cuda_gpu_kern_sum"] = sqlite_summary
            collection.selected_sources["cuda_gpu_kern_sum"] = "direct SQLite query"
            collection.failed.pop("cuda_gpu_kern_sum", None)
            warning = WarningRecord(
                "collect_stats",
                "cuda_gpu_kern_sum is unavailable; generated core summary from direct SQLite query",
            )
            collection.warnings.append(warning)
            progress.warning(warning.message)

        started = progress.begin("Normalize and classify kernel reports")
        kernels = load_kernel_summary(collection.successful["cuda_gpu_kern_sum"])
        base_path = collection.successful.get("cuda_gpu_kern_sum_base")
        base_kernels = load_kernel_summary(base_path) if base_path else kernels
        classified = classify_kernels(kernels)
        operator_hotspots = build_operator_hotspots(kernels)
        write_classification(classified, output_dir)
        write_operator_hotspots(operator_hotspots, output_dir)
        progress.finish("Normalize and classify kernel reports", started, output_path=output_dir / "operator_hotspots.csv")

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
        trace_path = collection.successful.get("cuda_gpu_trace")
        if args.analyze_dependencies or args.analyze_communication:
            event_source = trace_path or sqlite_path
            started = progress.begin("Analyze event dependencies and communication", input_path=event_source)
            raw_events = load_trace_csv(trace_path) if trace_path else extraction.events
            events = [replace(row, phase=phase.phase) for row in raw_events]
        if (args.analyze_dependencies or args.analyze_communication) and events:
            adjacency = build_adjacency(events)
            communication_events = analyze_communication(events)
            chains = build_communication_chains(events, adjacency, communication_events)
            candidates = build_fusion_candidates(communication_events, chains)
            write_communication_analysis(adjacency, communication_events, chains, candidates, output_dir)
            progress.finish("Analyze event dependencies and communication", started, output_path=output_dir / "communication_events.csv")
        elif args.analyze_dependencies or args.analyze_communication:
            warning = WarningRecord(
                "dependencies",
                "native CUDA trace and SQLite kernel events are unavailable; event analysis was not generated",
            )
            collection.warnings.append(warning)
            progress.warning(warning.message)
            progress.finish(
                "Analyze event dependencies and communication",
                started,
                "WARNING",
                detail=warning.message,
            )

        dependency_status = (
            "PASS" if args.analyze_dependencies and events else
            "FAILED" if args.analyze_dependencies else "NOT_REQUESTED"
        )
        communication_status = (
            "PASS" if args.analyze_communication and events else
            "FAILED" if args.analyze_communication else "NOT_REQUESTED"
        )
        overall_status = extraction.overall_status
        if "FAILED" in (dependency_status, communication_status):
            overall_status = "PARTIAL"
        extraction = replace(
            extraction,
            dependency_analysis_status=dependency_status,
            communication_analysis_status=communication_status,
            overall_status=overall_status,
        )
        write_event_metadata(extraction, output_dir)

        memory_events = load_memory_events(sqlite_path)
        memory_time_ns = sum(row.end_ns - row.start_ns for row in memory_events)
        h2d_time_ns = sum(
            row.end_ns - row.start_ns
            for row in memory_events
            if "h2d" in row.kind.lower()
            or "htod" in row.kind.lower()
            or row.kind.strip() == "1"
        )
        all_events = events or extraction.events
        runtime_collectives = [
            row for row in all_events if classify_kernel(row.name).runtime_communication
        ]
        requested_phase = str(
            capture.get("requested_phase") or capture.get("profile_phase") or ""
        )
        capture_log_text = ""
        capture_log_path = capture.get("server_log") or capture.get("nsys_log")
        if capture_log_path:
            candidate_log = Path(str(capture_log_path))
            if candidate_log.is_file():
                capture_log_text = candidate_log.read_text(
                    encoding="utf-8", errors="replace"
                ).lower()
        deepgemm_jit_detected = bool(
            capture.get("deepgemm_jit_detected")
            or ("deepgemm" in capture_log_text and "jit" in capture_log_text)
            or any(
                "deep_gemm" in row.name.lower() and "jit" in row.name.lower()
                for row in all_events
            )
        )
        moe_config_fallback_detected = bool(
            capture.get("moe_config_fallback_detected")
            or ("moe" in capture_log_text and "fallback" in capture_log_text)
        )
        invalid_timestamps = sum(row.end_ns < row.start_ns for row in all_events)
        integrity_result = evaluate_integrity(
            IntegrityInputs(
                report_size=report.stat().st_size,
                sqlite_size=sqlite_path.stat().st_size,
                kernel_event_count=len(all_events),
                invalid_timestamp_count=invalid_timestamps,
                requested_dependencies=args.analyze_dependencies,
                requested_communication=args.analyze_communication,
                event_trace_available=bool(all_events),
                communication_capability=bool(all_events),
                expected_tp=expected_tp,
                captured_devices=tuple(row.device_id for row in devices),
                requested_phase=requested_phase,
                detected_phase=phase.phase,
                initialization_only=bool(all_events) and all(
                    row.category == "Communication Init" for row in all_events
                ),
                runtime_collective_count=len(runtime_collectives),
                capture_duration_seconds=_optional_float(
                    capture.get("capture_duration_seconds")
                ),
                benchmark_duration_seconds=_optional_float(
                    capture.get("benchmark_duration_seconds")
                ),
                kernel_time_ns=sum(row.duration_ns for row in all_events),
                h2d_time_ns=h2d_time_ns,
                memory_time_ns=memory_time_ns,
                largest_nvtx=nvtx_names[0] if nvtx_names else "",
                capture_mode=str(capture.get("capture_mode") or "unknown"),
                deepgemm_jit_detected=deepgemm_jit_detected,
            )
        )

        native_tables = {
            name: read_csv_rows(path)
            for name, path in collection.successful.items()
            if name not in ("cuda_gpu_kern_sum", "cuda_gpu_kern_sum_base")
        }
        metadata = _metadata(args, report, sqlite_path, version, capture, collection, devices, integrity)
        metadata["profile_phase_attribution"] = phase.__dict__
        metadata["requested_phase"] = requested_phase or None
        metadata["detected_phase"] = phase.phase
        metadata["phase_confidence"] = phase.confidence
        metadata["phase_evidence"] = phase.evidence
        metadata["deepgemm_jit_detected"] = deepgemm_jit_detected
        metadata["moe_config_fallback_detected"] = moe_config_fallback_detected
        metadata["event_trace_source"] = (
            collection.selected_sources.get("cuda_gpu_trace")
            if trace_path
            else "direct SQLite query" if extraction.events else None
        )
        metadata["sqlite_missing_capabilities"] = extraction.missing_capabilities
        metadata["base_stats_status"] = extraction.base_stats_status
        metadata["kernel_event_status"] = extraction.kernel_event_status
        metadata["nvtx_load_status"] = extraction.nvtx_load_status
        metadata["nvtx_attribution_status"] = extraction.nvtx_attribution_status
        metadata["dependency_analysis_status"] = extraction.dependency_analysis_status
        metadata["communication_analysis_status"] = extraction.communication_analysis_status
        metadata["overall_status"] = extraction.overall_status
        metadata["nvtx_load_stats"] = (
            extraction.nvtx_stats.__dict__ if extraction.nvtx_stats else None
        )
        metadata["event_extraction_warnings"] = extraction.warnings
        metadata["nvtx_attribution_query_count"] = extraction.attribution_query_count
        metadata["nvtx_attribution_candidate_checks"] = extraction.attribution_candidate_checks
        metadata["raw_report_integrity"] = integrity_result.raw_report_integrity
        metadata["analysis_completeness"] = integrity_result.analysis_completeness
        metadata["raw_integrity_reasons"] = list(integrity_result.raw_reasons)
        metadata["analysis_completeness_reasons"] = list(integrity_result.analysis_reasons)
        metadata["capture_sanity_checks"] = list(integrity_result.sanity_checks)
        metadata.update(integrity_result.flags)
        metadata["integrity_ok"] = (
            integrity_result.raw_report_integrity == "PASS"
            and integrity_result.analysis_completeness == "PASS"
        )
        metadata["warnings"] = [row.__dict__ for row in collection.warnings + device_warnings]
        started = progress.begin("Write metadata", output_path=output_dir / "metadata.json")
        atomic_write_json(output_dir / "metadata.json", metadata)
        progress.finish("Write metadata", started, output_path=output_dir / "metadata.json")

        data = AnalysisData(
            metadata=metadata,
            reports=collection,
            kernels=kernels,
            base_kernels=base_kernels,
            operator_hotspots=operator_hotspots,
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
