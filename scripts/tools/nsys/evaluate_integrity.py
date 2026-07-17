"""Separate raw report validity from requested analysis completeness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple


@dataclass(frozen=True)
class IntegrityInputs:
    report_size: int
    sqlite_size: int
    kernel_event_count: int
    invalid_timestamp_count: int
    requested_dependencies: bool
    requested_communication: bool
    event_trace_available: bool
    communication_capability: bool
    expected_tp: Optional[int]
    captured_devices: Tuple[int, ...]
    requested_phase: str
    detected_phase: str
    initialization_only: bool
    runtime_collective_count: int
    capture_duration_seconds: Optional[float]
    benchmark_duration_seconds: Optional[float]
    kernel_time_ns: float
    h2d_time_ns: float
    memory_time_ns: float
    largest_nvtx: str
    capture_mode: str
    deepgemm_jit_detected: bool


@dataclass(frozen=True)
class IntegrityResult:
    raw_report_integrity: str
    analysis_completeness: str
    raw_reasons: Tuple[str, ...] = field(default_factory=tuple)
    analysis_reasons: Tuple[str, ...] = field(default_factory=tuple)
    sanity_checks: Tuple[str, ...] = field(default_factory=tuple)
    flags: Dict[str, bool] = field(default_factory=dict)


def evaluate_integrity(inputs: IntegrityInputs) -> IntegrityResult:
    raw_reasons = []
    analysis_reasons = []
    sanity = []

    if inputs.report_size <= 0:
        raw_reasons.append("input report is missing or empty")
    if inputs.sqlite_size <= 0:
        raw_reasons.append("SQLite export is missing or empty")
    if inputs.invalid_timestamp_count:
        raw_reasons.append(
            f"{inputs.invalid_timestamp_count} kernel event(s) have invalid timestamps"
        )
    if raw_reasons:
        raw_state = "FAIL"
    elif inputs.kernel_event_count <= 0:
        raw_state = "PARTIAL"
        raw_reasons.append("kernel event table is unavailable or contains no events")
    else:
        raw_state = "PASS"

    if inputs.requested_dependencies and not inputs.event_trace_available:
        analysis_reasons.append(
            "dependency analysis was requested but event trace is unavailable"
        )
    if inputs.requested_communication and not inputs.communication_capability:
        analysis_reasons.append(
            "communication analysis was requested but no communication-capable event table is available"
        )
    if inputs.expected_tp and inputs.expected_tp > 1:
        if len(inputs.captured_devices) != inputs.expected_tp:
            analysis_reasons.append(
                f"expected TP{inputs.expected_tp} but captured devices are "
                f"{list(inputs.captured_devices) or 'N/A'}"
            )
    requested_phase = inputs.requested_phase.lower()
    if requested_phase in ("startup", "prefill", "decode", "full"):
        if inputs.detected_phase.upper() == "UNKNOWN":
            analysis_reasons.append(
                f"requested phase {requested_phase} has UNKNOWN attribution"
            )
        elif inputs.detected_phase.upper() != requested_phase.upper():
            analysis_reasons.append(
                f"requested phase {requested_phase} was detected as "
                f"{inputs.detected_phase.upper()}"
            )
    if inputs.initialization_only:
        analysis_reasons.append("kernel events contain initialization only")
        sanity.append("initialization-only kernel events detected")
    if (
        inputs.requested_communication
        and inputs.runtime_collective_count <= 0
    ):
        analysis_reasons.append("runtime collective count is zero")

    if (
        inputs.capture_duration_seconds
        and inputs.benchmark_duration_seconds
        and inputs.capture_duration_seconds > 0
        and inputs.benchmark_duration_seconds > 0
    ):
        ratio = inputs.capture_duration_seconds / inputs.benchmark_duration_seconds
        if ratio < 0.5 or ratio > 2.0:
            sanity.append(
                "capture duration and benchmark duration materially disagree "
                f"(ratio={ratio:.3f})"
            )
    if inputs.capture_duration_seconds and inputs.capture_duration_seconds > 0:
        window_ns = inputs.capture_duration_seconds * 1_000_000_000
        if inputs.kernel_time_ns < window_ns * 0.01:
            sanity.append(
                "summed kernel time is suspiciously small relative to capture window"
            )
    if requested_phase == "decode":
        if "ncclcomminitrank" in inputs.largest_nvtx.lower():
            sanity.append("decode capture is dominated by communication initialization")
        if inputs.runtime_collective_count <= 0:
            sanity.append("decode capture has no runtime collective")
        if inputs.memory_time_ns > 0 and inputs.h2d_time_ns / inputs.memory_time_ns > 0.5:
            sanity.append("decode capture is dominated by H2D model-loading traffic")

    analysis_reasons.extend(sanity)
    if raw_state == "FAIL":
        analysis_state = "FAIL"
    elif analysis_reasons or raw_state == "PARTIAL":
        analysis_state = "PARTIAL"
    else:
        analysis_state = "PASS"

    return IntegrityResult(
        raw_report_integrity=raw_state,
        analysis_completeness=analysis_state,
        raw_reasons=tuple(raw_reasons),
        analysis_reasons=tuple(analysis_reasons),
        sanity_checks=tuple(sanity),
        flags={
            "startup_contaminated": bool(
                inputs.capture_mode == "full-offline"
                and inputs.deepgemm_jit_detected
            )
        },
    )
