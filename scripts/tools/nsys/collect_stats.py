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
    "cuda_gpu_kern_gb_sum",
    "cuda_kern_exec_sum",
    "cuda_api_sum",
    "nvtx_sum",
    "nvtx_gpu_proj_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_mem_size_sum",
)
TRACE_REPORTS = (
    "cuda_gpu_trace",
    "cuda_kern_exec_trace",
    "nvtx_kern_sum",
    "nvtx_gpu_proj_trace",
)
KNOWN_REPORTS = BASE_REPORTS + TRACE_REPORTS

REPORT_FALLBACKS = {
    "cuda_gpu_trace": ("cuda_gpu_trace:nvtx-name", "cuda_gpu_trace"),
    "cuda_kern_exec_trace": (
        "cuda_kern_exec_trace:nvtx-name",
        "cuda_kern_exec_trace",
    ),
    "nvtx_kern_sum": ("nvtx_kern_sum",),
}


def report_candidates(report_name: str) -> tuple[str, ...]:
    return REPORT_FALLBACKS.get(report_name, (report_name,))


def parse_help_report_names(text: str) -> Set[str]:
    """Extract concrete report names while ignoring help grammar suffixes."""
    pattern = r"\b((?:cuda|nvtx)_[a-z0-9_]+(?::(?:base|mangled|nvtx-name))?)(?=\[|\s|,|$)"
    return {match.lower() for match in re.findall(pattern, text, re.IGNORECASE)}


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
    nsys_path: str,
    output_dir: Path,
    progress: ProgressReporter,
    warnings: Optional[List[WarningRecord]] = None,
) -> Set[str]:
    command = [nsys_path, "stats", "--help-reports"]
    help_path = output_dir / ".help-reports.txt"
    started = progress.begin("Detect supported reports", command=command, output_path=help_path)
    code = run_streaming_command(
        command,
        help_path,
        output_dir / "progress.log",
        progress,
        monitored_output=help_path,
        merge_stderr=True,
    )
    text = help_path.read_text(encoding="utf-8", errors="replace")
    supported = parse_help_report_names(text)
    degraded_help_is_valid = (
        CORE_REPORT in supported
        and "cuda_api_sum" in supported
        and len(supported) >= 2
    )
    if code != 0 and not degraded_help_is_valid:
        progress.finish("Detect supported reports", started, "FAILED", output_path=help_path)
        help_path.unlink(missing_ok=True)
        raise CoreReportError(
            f"nsys stats --help-reports failed with exit {code} and no valid help body"
        )
    help_path.unlink(missing_ok=True)
    if CORE_REPORT not in supported:
        progress.finish(
            "Detect supported reports", started, "FAILED", detail="core report missing"
        )
        raise CoreReportError(f"installed Nsight does not support core report {CORE_REPORT}")
    if code != 0:
        message = (
            f"nsys stats --help-reports returned exit {code}, but its help body is valid; continuing"
        )
        if warnings is not None:
            warnings.append(WarningRecord("detect_reports", message))
        progress.finish(
            "Detect supported reports", started, "WARNING", detail=f"supported={len(supported)}"
        )
        progress.warning(message)
    else:
        progress.finish(
            "Detect supported reports", started, detail=f"supported={len(supported)}"
        )
    return supported


def collect_reports(
    sqlite_path: Path,
    report_names: Sequence[str],
    supported_reports: Optional[Set[str]],
    nsys_path: str,
    output_dir: Path,
    progress: ProgressReporter,
    allow_core_fallback: bool = False,
) -> ReportCollection:
    output_dir.mkdir(parents=True, exist_ok=True)
    collection = ReportCollection()
    for report_name in report_names:
        candidates = report_candidates(report_name)
        is_supported = supported_reports is None or any(
            candidate in supported_reports or report_name in supported_reports
            for candidate in candidates
        )
        if not is_supported:
            message = f"optional Nsight report unsupported: {report_name}"
            if report_name == CORE_REPORT:
                raise CoreReportError(message)
            collection.unsupported.append(report_name)
            collection.warnings.append(WarningRecord("collect_stats", message))
            progress.warning(message)
            continue

        final_path = output_dir / report_filename(report_name)
        candidate_errors = []
        selected = None
        for candidate in candidates:
            if (
                supported_reports is not None
                and candidate not in supported_reports
                and report_name not in supported_reports
            ):
                continue
            temporary = final_path.with_name(final_path.name + ".tmp")
            temporary.unlink(missing_ok=True)
            command = [
                nsys_path,
                "stats",
                "--report",
                candidate,
                "--format",
                "csv",
                str(sqlite_path),
            ]
            stage_name = f"Generate {candidate}"
            started = progress.begin(
                stage_name,
                command=command,
                input_path=sqlite_path,
                output_path=temporary,
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
                candidate_errors.append(f"{candidate}: exit {code}")
                progress.finish(stage_name, started, "FAILED", output_path=final_path)
                continue
            if not temporary.is_file() or temporary.stat().st_size == 0:
                temporary.unlink(missing_ok=True)
                candidate_errors.append(f"{candidate}: no data")
                progress.finish(stage_name, started, "WARNING", output_path=final_path)
                continue
            os.replace(str(temporary), str(final_path))
            collection.successful[report_name] = final_path
            collection.selected_sources[report_name] = candidate
            selected = candidate
            progress.finish(stage_name, started, output_path=final_path)
            break
        if selected is not None:
            if selected != candidates[0]:
                message = (
                    f"Nsight report {report_name} used fallback {selected}; "
                    + "; ".join(candidate_errors)
                )
                collection.warnings.append(WarningRecord("collect_stats", message))
                progress.warning(message)
            continue
        message = (
            f"Nsight report {report_name} failed: "
            + ("; ".join(candidate_errors) or "no supported candidate")
        )
        if report_name == CORE_REPORT and not allow_core_fallback:
            raise CoreReportError(message)
        collection.failed[report_name] = message
        collection.warnings.append(WarningRecord("collect_stats", message))
        progress.warning(message)
    return collection
