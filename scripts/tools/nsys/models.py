"""Shared records used by the Nsight analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class WarningRecord:
    stage: str
    message: str


@dataclass(frozen=True)
class KernelSummary:
    name: str
    total_ns: float
    instances: int
    time_percentage: float
    avg_ns: Optional[float] = None
    median_ns: Optional[float] = None
    min_ns: Optional[float] = None
    max_ns: Optional[float] = None
    stddev_ns: Optional[float] = None
    grid: Optional[str] = None
    block: Optional[str] = None


@dataclass(frozen=True)
class ClassifiedKernel:
    name: str
    base_family: str
    category: str
    rule: str
    total_ns: float
    instances: int
    time_percentage: float


@dataclass(frozen=True)
class DeviceSummary:
    device_id: int
    process_count: int
    kernel_events: int
    kernel_time_ns: float
    compute_time_ns: float
    communication_time_ns: float
    communication_percentage: Optional[float]
    top_family: Optional[str]
    relative_time: Optional[float]
    imbalance: Optional[float]
    gpu_name: Optional[str] = None


@dataclass(frozen=True)
class KernelEvent:
    event_id: int
    device_id: int
    context_id: int
    stream_id: int
    start_ns: int
    end_ns: int
    name: str
    category: str = "Unknown"
    family: str = "Unknown"
    phase: str = "UNKNOWN"
    module: str = "N/A"

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)


@dataclass(frozen=True)
class AdjacencyRecord:
    device_id: int
    context_id: int
    stream_id: int
    previous_kernel: str
    next_kernel: str
    gap_ns: int
    relation_type: str = "temporal_adjacency"
    evidence: str = "same stream and temporal order"
    confidence: str = "LOW"


@dataclass(frozen=True)
class CommunicationEvent:
    event_id: int
    device_id: int
    name: str
    family: str
    start_ns: int
    end_ns: int
    duration_ns: int
    overlap_compute_ns: int
    exposed_communication_ns: int
    phase: str
    module: str


@dataclass(frozen=True)
class CommunicationChain:
    relation_type: str
    device_id: int
    phase: str
    module: str
    previous_family: str
    communication_family: str
    next_family: str
    count: int
    denominator: int
    adjacency_rate: float
    evidence: str
    confidence: str


@dataclass(frozen=True)
class FusionCandidate:
    candidate: str
    importance_score: float
    exposed_score: float
    frequency_score: float
    adjacency_score: float
    attribution_score: float
    feasibility_score: float
    total_score: float
    required_primitive: str
    tle_feasibility: str = "UNKNOWN"


@dataclass
class ReportCollection:
    successful: Dict[str, Path] = field(default_factory=dict)
    unsupported: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)
    empty: List[str] = field(default_factory=list)
    warnings: List[WarningRecord] = field(default_factory=list)


@dataclass(frozen=True)
class PhaseAttribution:
    phase: str
    source: str
    evidence: str
    confidence: str


@dataclass
class AnalysisData:
    metadata: Dict[str, object]
    reports: ReportCollection
    kernels: List[KernelSummary] = field(default_factory=list)
    base_kernels: List[KernelSummary] = field(default_factory=list)
    classified: List[ClassifiedKernel] = field(default_factory=list)
    devices: List[DeviceSummary] = field(default_factory=list)
    adjacency: List[AdjacencyRecord] = field(default_factory=list)
    communication_events: List[CommunicationEvent] = field(default_factory=list)
    communication_chains: List[CommunicationChain] = field(default_factory=list)
    fusion_candidates: List[FusionCandidate] = field(default_factory=list)
    phase_attribution: Optional[PhaseAttribution] = None
    native_tables: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    warnings: List[WarningRecord] = field(default_factory=list)
