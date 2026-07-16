"""Detect and collect native Nsight statistics from an exported SQLite file."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

from .models import ReportCollection, WarningRecord
from .progress import ProgressReporter, run_streaming_command


class CoreReportError(RuntimeError):
    pass


CORE_REPORT = "cuda_gpu_kern_sum"
BASE_REPORTS = (
    CORE_REPORT,
    "cuda_gpu_kern_sum:base",
    "cuda_gpu_kern_gb_sum",
    "cuda_kern_exec_sum:base",
    "cuda_api_sum",
    "nvtx_sum",
    "nvtx_gpu_proj_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
)
TRACE_REPORTS = (
    "cuda_gpu_trace:base",
    "cuda_kern_exec_trace:base",
    "nvtx_kern_sum:base",
    "nvtx_gpu_proj_trace",
)
KNOWN_REPORTS = BASE_REPORTS + TRACE_REPORTS


def report_filename(report_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", report_name).replace(":", "_") + ".csv"


def select_reports(
    explicit: Optional[str], analyze_dependencies: bool, analyze_communication: bool
) -> List[str]:
    if explicit:
        requested = [item.strip() for item in explicit.split(",") if item.strip()]
        selected = [CORE_REPORT] + [item for item in requested if item != CORE_REPORT]
    else:
        selected = list(BASE_REPORTS)
        if analyze_dependencies or analyze_communication:
            selected.extend(TRACE_REPORTS)
    seen = set()
    return [item for item in selected if not (item in seen or seen.add(item))]


def detect_supported_reports(
    nsys_path: str, output_dir: Path, progress: ProgressReporter
) -> Set[str]:
    command = [nsys_path, "stats", "--help-reports"]
    help_path = output_dir / ".help-reports.txt"
    started = progress.begin("Detect supported reports", command=command, output_path=help_path)
    code = run_streaming_command(
        command, help_path, output_dir / "progress.log", progress, monitored_output=help_path
    )
    if code != 0:
        progress.finish("Detect supported reports", started, "FAILED", output_path=help_path)
        help_path.unlink(missing_ok=True)
        raise CoreReportError(f"nsys stats --help-reports failed with exit {code}")
    text = help_path.read_text(encoding="utf-8", errors="replace")
    supported = {name for name in KNOWN_REPORTS if name in text}
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_]*(?::[A-Za-z0-9_]+)?", text):
        if token.startswith(("cuda_", "nvtx_")):
            supported.add(token)
    help_path.unlink(missing_ok=True)
    progress.finish(
        "Detect supported reports", started, detail=f"supported={len(supported)}"
    )
    if CORE_REPORT not in supported:
        raise CoreReportError(f"installed Nsight does not support core report {CORE_REPORT}")
    return supported


def collect_reports(
    sqlite_path: Path,
    report_names: Sequence[str],
    supported_reports: Set[str],
    nsys_path: str,
    output_dir: Path,
    progress: ProgressReporter,
) -> ReportCollection:
    output_dir.mkdir(parents=True, exist_ok=True)
    collection = ReportCollection()
    for report_name in report_names:
        if report_name not in supported_reports:
            message = f"optional Nsight report unsupported: {report_name}"
            if report_name == CORE_REPORT:
                raise CoreReportError(message)
            collection.unsupported.append(report_name)
            collection.warnings.append(WarningRecord("collect_stats", message))
            progress.warning(message)
            continue

        final_path = output_dir / report_filename(report_name)
        temporary = final_path.with_name(final_path.name + ".tmp")
        temporary.unlink(missing_ok=True)
        command = [
            nsys_path,
            "stats",
            "--report",
            report_name,
            "--format",
            "csv",
            str(sqlite_path),
        ]
        stage_name = f"Generate {report_name}"
        started = progress.begin(
            stage_name, command=command, input_path=sqlite_path, output_path=temporary
        )
        code = run_streaming_command(
            command,
            temporary,
            output_dir / "progress.log",
            progress,
            monitored_output=temporary,
        )
        if code != 0:
            temporary.unlink(missing_ok=True)
            message = f"Nsight report {report_name} failed with exit {code}"
            progress.finish(stage_name, started, "FAILED", output_path=final_path)
            if report_name == CORE_REPORT:
                raise CoreReportError(message)
            collection.failed[report_name] = message
            collection.warnings.append(WarningRecord("collect_stats", message))
            progress.warning(message)
            continue
        if not temporary.is_file() or temporary.stat().st_size == 0:
            temporary.unlink(missing_ok=True)
            message = f"Nsight report {report_name} produced no data"
            progress.finish(stage_name, started, "WARNING", output_path=final_path)
            if report_name == CORE_REPORT:
                raise CoreReportError(message)
            collection.empty.append(report_name)
            collection.warnings.append(WarningRecord("collect_stats", message))
            progress.warning(message)
            continue
        os.replace(str(temporary), str(final_path))
        collection.successful[report_name] = final_path
        progress.finish(stage_name, started, output_path=final_path)
    return collection
