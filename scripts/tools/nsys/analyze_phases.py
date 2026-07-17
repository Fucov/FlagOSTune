"""Conservative phase attribution with explicit evidence priority."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional

from .models import PhaseAttribution


def _phase_from_text(value: str) -> Optional[str]:
    lowered = value.lower()
    has_prefill = any(token in lowered for token in ("prefill", "extend"))
    has_decode = any(token in lowered for token in ("decode", "generation"))
    if has_prefill and has_decode:
        return "MIXED"
    if has_prefill:
        return "PREFILL"
    if has_decode:
        return "DECODE"
    if "startup" in lowered or "initialization" in lowered:
        return "STARTUP"
    if "full" in lowered:
        return "FULL"
    return None


def attribute_phase(
    metadata: Mapping[str, object],
    nvtx_ranges: Iterable[str],
    phase_log: Optional[Path] = None,
) -> PhaseAttribution:
    joined_nvtx = " ".join(str(value) for value in nvtx_ranges)
    phase = _phase_from_text(joined_nvtx)
    if phase:
        return PhaseAttribution(phase, "NVTX", joined_nvtx[:500], "HIGH")
    if phase_log and phase_log.is_file():
        log_text = phase_log.read_text(encoding="utf-8", errors="replace")
        phase = _phase_from_text(log_text)
        if phase:
            return PhaseAttribution(phase, "timestamped SGLang log", str(phase_log), "MEDIUM")
    metadata_phase = str(metadata.get("profile_phase") or "")
    phase = _phase_from_text(metadata_phase)
    if phase:
        metadata_evidence = str(metadata.get("phase_evidence") or metadata_phase)
        metadata_confidence = str(metadata.get("phase_confidence") or "MEDIUM").upper()
        if metadata_confidence not in ("HIGH", "MEDIUM", "LOW"):
            metadata_confidence = "MEDIUM"
        return PhaseAttribution(
            phase, "workflow metadata", metadata_evidence, metadata_confidence
        )
    return PhaseAttribution(
        "UNKNOWN", "insufficient evidence", "no explicit NVTX/log/metadata phase evidence", "LOW"
    )
