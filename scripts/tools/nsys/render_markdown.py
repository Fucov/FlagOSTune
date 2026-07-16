"""Render the stable mentor-style Nsight analysis Markdown contract."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .models import AnalysisData, KernelSummary
from .utils import atomic_write_text, format_bytes


SECTION_NAMES = (
    "Execution Summary",
    "Experiment Environment",
    "Workload and Benchmark",
    "Data Integrity",
    "Top Kernel Families",
    "Top Kernel Variants",
    "Kernel Classification",
    "Multi-GPU and TP Rank",
    "CUDA API and Launch/Execution",
    "NVTX and Module Attribution",
    "Communication Analysis",
    "Communication–Compute Candidate Chains",
    "Triton/TLE Fusion Candidates",
    "Prefill/Decode/Mixed Auxiliary Attribution",
    "Torch Profiler Cross-Validation Interface",
    "Data Scope and Limitations",
)
SECTION_TITLES = tuple(f"## {index}. {name}" for index, name in enumerate(SECTION_NAMES, 1))


def _escape(value: object) -> str:
    if value is None or value == "":
        return "N/A"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _mapping_table(rows: Sequence[Mapping[str, object]], top: int) -> str:
    if not rows:
        return "N/A"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(_escape(value) for value in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows[:top]:
        lines.append("| " + " | ".join(_escape(row.get(key)) for key in headers) + " |")
    return "\n".join(lines)


def _kernel_table(rows: Sequence[KernelSummary], top: int) -> str:
    values = []
    for row in rows[:top]:
        values.append(
            {
                "Time (%)": f"{row.time_percentage:.2f}",
                "Total (ms)": f"{row.total_ns / 1_000_000:.3f}",
                "Instances": row.instances,
                "Avg (us)": f"{row.avg_ns / 1_000:.3f}" if row.avg_ns is not None else "N/A",
                "Name": row.name,
            }
        )
    return _mapping_table(values, top)


def _native(data: AnalysisData, report: str, top: int) -> str:
    return _mapping_table(data.native_tables.get(report, []), top)


def render_markdown(
    data: AnalysisData, top: int = 20, output_path: Optional[Path] = None
) -> str:
    metadata = data.metadata
    kernels_count = sum(row.instances for row in data.kernels) if data.kernels else None
    warning_rows = list(data.warnings) + list(data.reports.warnings)
    warning_rows.extend(
        type("WarningView", (), {"stage": "reports", "message": f"unsupported report: {name}"})()
        for name in data.reports.unsupported
    )
    warnings = "\n".join(f"- `{row.stage}`: {row.message}" for row in warning_rows) or "- None"
    device_rows = [row.__dict__ for row in data.devices]
    classification_rows = [row.__dict__ for row in data.classified]
    communication_rows = [row.__dict__ for row in data.communication_events]
    chain_rows = [row.__dict__ for row in data.communication_chains]
    candidate_rows = [row.__dict__ for row in data.fusion_candidates]
    phase = data.phase_attribution

    sections = [
        "# FlagOSTune Nsight Systems Analysis",
        SECTION_TITLES[0],
        "\n".join(
            (
                "| Field | Value |",
                "| --- | --- |",
                f"| Input report | {_escape(metadata.get('input_report'))} |",
                f"| Input size | {format_bytes(metadata.get('input_size'))} |",
                f"| Input mtime | {_escape(metadata.get('input_mtime'))} |",
                f"| SQLite | {_escape(metadata.get('sqlite_path'))} |",
                f"| SQLite size | {format_bytes(metadata.get('sqlite_size'))} |",
                f"| Kernel instances | {_escape(kernels_count)} |",
            )
        ),
        SECTION_TITLES[1],
        f"Nsight Systems version: `{_escape(metadata.get('nsys_version'))}`.\n\nCaptured GPUs: `{len(data.devices) if data.devices else 'N/A'}`.",
        SECTION_TITLES[2],
        "\n".join(
            f"- {key}: `{_escape(metadata.get(key))}`"
            for key in ("model", "scenario", "workload", "tp_size", "visible_devices", "capture_mode", "profile_phase")
        ),
        SECTION_TITLES[3],
        f"Integrity status: **{'PASS' if metadata.get('integrity_ok') is True else 'N/A' if metadata.get('integrity_ok') is None else 'FAILED'}**.\n\nWarnings:\n{warnings}",
        SECTION_TITLES[4],
        f"Top {top} base kernel families:\n\n{_kernel_table(data.base_kernels, top)}\n\nKernel grid/block summary (launch-shape proxy):\n\n{_native(data, 'cuda_gpu_kern_gb_sum', top)}",
        SECTION_TITLES[5],
        f"Top {top} complete CUDA kernel names:\n\n{_kernel_table(data.kernels, top)}",
        SECTION_TITLES[6],
        f"{_mapping_table(classification_rows, top)}\n\nUnknown kernel rows remain Unknown and are also written to `unknown_kernels.csv`.",
        SECTION_TITLES[7],
        _mapping_table(device_rows, top),
        SECTION_TITLES[8],
        f"CUDA API Top:\n\n{_native(data, 'cuda_api_sum', top)}\n\nCUDA launch/API/queue/kernel time:\n\n{_native(data, 'cuda_kern_exec_sum:base', top)}",
        SECTION_TITLES[9],
        f"NVTX range summary:\n\n{_native(data, 'nvtx_sum', top)}\n\nNVTX GPU projection:\n\n{_native(data, 'nvtx_gpu_proj_sum', top)}",
        SECTION_TITLES[10],
        f"Communication events:\n\n{_mapping_table(communication_rows, top)}\n\nGPU memory operation time:\n\n{_native(data, 'cuda_gpu_mem_time_sum', top)}\n\nGPU memory operation size:\n\n{_native(data, 'cuda_gpu_mem_size_sum', top)}",
        SECTION_TITLES[11],
        _mapping_table(chain_rows, top),
        SECTION_TITLES[12],
        f"{_mapping_table(candidate_rows, top)}\n\nCandidate scores are screening heuristics, not predicted speedups. NCCL collectives require a distributed primitive; ordinary Triton does not replace NCCL. Unchecked TLE feasibility is UNKNOWN.",
        SECTION_TITLES[13],
        (
            f"Phase: **{phase.phase}**; source: `{phase.source}`; confidence: `{phase.confidence}`; evidence: {_escape(phase.evidence)}."
            if phase else "Phase: **UNKNOWN**; attribution evidence is unavailable."
        ),
        SECTION_TITLES[14],
        "This report preserves an interface for later Torch Profiler cross-validation. It does not modify or invoke the existing Torch Profiler workflow.",
        SECTION_TITLES[15],
        "\n".join(
            (
                "- `cuda_gpu_kern_sum` Time (%) uses the **sum of all kernel durations** as its denominator.",
                "- In TP multi-GPU execution, summed GPU kernel time is **not wall-clock** time.",
                "- CUDA API Time (%) uses the **sum of CUDA API durations** as its denominator.",
                "- Summed NCCL time is **not critical-path communication overhead**.",
                "- Same-stream temporal adjacency is **not a Tensor data dependency**.",
                "- Kernel grid/block is a launch-shape proxy, **not Tensor shape**.",
                "- Exposed communication is a **FlagOSTune-derived timeline metric**, not an Nsight critical-path metric.",
                "- A full-run trace is not decode-only and must not be described as a decode-only trace.",
                "- CUDA-graph-disabled measurements do not represent a graph-enabled production path.",
                "- Conclusions apply to the **captured workload only**.",
            )
        ),
    ]
    markdown = "\n\n".join(sections).rstrip() + "\n"
    if output_path is not None:
        atomic_write_text(output_path, markdown)
    return markdown
