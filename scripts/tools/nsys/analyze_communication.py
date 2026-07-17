"""Derived communication overlap, chain, and candidate analysis."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .models import (
    AdjacencyRecord,
    CommunicationChain,
    CommunicationEvent,
    FusionCandidate,
    KernelEvent,
)
from .utils import write_csv
from .classify_kernels import classify_kernel


def _percentile(values: Sequence[int], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_communication(
    communication: Sequence[CommunicationEvent],
) -> List[dict]:
    grouped = defaultdict(list)
    for event in communication:
        classification = classify_kernel(event.name)
        provider = "NCCL" if classification.category.startswith("NCCL ") else "Custom"
        if classification.category == "P2P Send/Recv" and "nccl" in event.name.lower():
            provider = "NCCL"
        grouped[(event.device_id, provider, classification.category)].append(event)
    output = []
    for (device, provider, collective), values in sorted(grouped.items()):
        durations = [row.duration_ns for row in values]
        total = sum(durations)
        overlap = sum(row.overlap_compute_ns for row in values)
        exposed = sum(row.exposed_communication_ns for row in values)
        output.append(
            {
                "device_id": device,
                "provider": provider,
                "collective": collective,
                "count": len(values),
                "total_duration_ns": total,
                "average_ns": total / len(values),
                "p50_ns": _percentile(durations, 0.50),
                "p95_ns": _percentile(durations, 0.95),
                "compute_overlap_ns": overlap,
                "exposed_communication_ns": exposed,
                "exposed_ratio": exposed / total if total else 0.0,
            }
        )
    return output


def summarize_arrival_skew(
    communication: Sequence[CommunicationEvent],
) -> List[dict]:
    by_collective_device = defaultdict(lambda: defaultdict(list))
    for event in communication:
        collective = classify_kernel(event.name).category
        by_collective_device[collective][event.device_id].append(event)
    output = []
    for collective, by_device in sorted(by_collective_device.items()):
        if len(by_device) < 2:
            continue
        ordered = {
            device: sorted(values, key=lambda row: (row.start_ns, row.event_id))
            for device, values in by_device.items()
        }
        counts = {len(values) for values in ordered.values()}
        if len(counts) != 1:
            continue
        count = counts.pop()
        for occurrence in range(count):
            arrivals = {
                device: values[occurrence].start_ns
                for device, values in ordered.items()
            }
            output.append(
                {
                    "collective": collective,
                    "occurrence": occurrence,
                    "device_count": len(arrivals),
                    "earliest_start_ns": min(arrivals.values()),
                    "latest_start_ns": max(arrivals.values()),
                    "arrival_skew_ns": max(arrivals.values()) - min(arrivals.values()),
                    "evidence": "same collective and occurrence order across devices",
                    "confidence": "MEDIUM",
                }
            )
    return output


def _union_length(intervals: Iterable[Tuple[int, int]], lower: int, upper: int) -> int:
    clipped = sorted((max(lower, start), min(upper, end)) for start, end in intervals if end > lower and start < upper)
    if not clipped:
        return 0
    total = 0
    current_start, current_end = clipped[0]
    for start, end in clipped[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += max(0, current_end - current_start)
            current_start, current_end = start, end
    total += max(0, current_end - current_start)
    return min(max(0, total), max(0, upper - lower))


def analyze_communication(events: Iterable[KernelEvent]) -> List[CommunicationEvent]:
    materialized = list(events)
    computes = defaultdict(list)
    for event in materialized:
        if not classify_kernel(event.name).runtime_communication:
            computes[event.device_id].append((event.start_ns, event.end_ns))
    output = []
    for event in materialized:
        if not classify_kernel(event.name).runtime_communication:
            continue
        duration = event.duration_ns
        overlap = _union_length(computes[event.device_id], event.start_ns, event.end_ns)
        exposed = min(duration, max(0, duration - overlap))
        output.append(
            CommunicationEvent(
                event_id=event.event_id,
                device_id=event.device_id,
                name=event.name,
                family=event.family,
                start_ns=event.start_ns,
                end_ns=event.end_ns,
                duration_ns=duration,
                overlap_compute_ns=overlap,
                exposed_communication_ns=exposed,
                phase=event.phase,
                module=event.module,
            )
        )
    return output


def build_communication_chains(
    events: Sequence[KernelEvent],
    adjacency: Sequence[AdjacencyRecord],
    communication: Sequence[CommunicationEvent],
) -> List[CommunicationChain]:
    del adjacency  # Event IDs and stream order are more reliable than name-only rows.
    streams = defaultdict(list)
    for event in events:
        streams[(event.device_id, event.context_id, event.stream_id)].append(event)
    comm_by_id = {row.event_id: row for row in communication}
    denominators = Counter((row.device_id, row.phase, row.module, row.family) for row in communication)
    aggregate = Counter()
    evidence = "same-stream temporal adjacency; not a Tensor data dependency"
    for values in streams.values():
        ordered = sorted(values, key=lambda row: (row.start_ns, row.end_ns, row.event_id))
        for index, event in enumerate(ordered):
            comm = comm_by_id.get(event.event_id)
            if comm is None:
                continue
            previous = ordered[index - 1] if index > 0 and not classify_kernel(ordered[index - 1].name).runtime_communication else None
            following = ordered[index + 1] if index + 1 < len(ordered) and not classify_kernel(ordered[index + 1].name).runtime_communication else None
            if previous:
                aggregate[("Compute→Communication", comm.device_id, comm.phase, comm.module, previous.family, comm.family, "N/A")] += 1
            if following:
                aggregate[("Communication→Compute", comm.device_id, comm.phase, comm.module, "N/A", comm.family, following.family)] += 1
            if previous and following:
                aggregate[("Compute→Communication→Compute", comm.device_id, comm.phase, comm.module, previous.family, comm.family, following.family)] += 1
    output = []
    for key, count in aggregate.items():
        relation, device, phase, module, previous, comm_family, following = key
        denominator = denominators[(device, phase, module, comm_family)]
        output.append(
            CommunicationChain(
                relation, device, phase, module, previous, comm_family, following,
                count, denominator, count / denominator if denominator else 0.0,
                evidence, "LOW",
            )
        )
    return sorted(output, key=lambda row: (-row.count, row.relation_type))


def build_fusion_candidates(
    communication: Sequence[CommunicationEvent], chains: Sequence[CommunicationChain]
) -> List[FusionCandidate]:
    by_family = defaultdict(list)
    for event in communication:
        by_family[event.family].append(event)
    output = []
    total_duration = sum(event.duration_ns for event in communication) or 1
    max_count = max((len(values) for values in by_family.values()), default=1)
    for family, values in by_family.items():
        duration = sum(value.duration_ns for value in values)
        exposed = sum(value.exposed_communication_ns for value in values)
        relevant = [row for row in chains if row.communication_family == family]
        importance_score = duration / total_duration
        exposed_score = exposed / duration if duration else 0.0
        frequency_score = len(values) / max_count
        adjacency_score = max((row.adjacency_rate for row in relevant), default=0.0)
        attribution_score = sum(value.module != "N/A" for value in values) / len(values)
        feasibility_score = 0.25
        total = (
            0.25 * importance_score + 0.25 * exposed_score + 0.15 * frequency_score
            + 0.15 * adjacency_score + 0.1 * attribution_score + 0.1 * feasibility_score
        )
        output.append(
            FusionCandidate(
                candidate=f"screen {family} adjacent compute for collective-aware fusion",
                importance_score=importance_score,
                exposed_score=exposed_score,
                frequency_score=frequency_score,
                adjacency_score=adjacency_score,
                attribution_score=attribution_score,
                feasibility_score=feasibility_score,
                total_score=total,
                required_primitive="distributed collective primitive",
            )
        )
    return sorted(output, key=lambda row: row.total_score, reverse=True)


def write_communication_analysis(
    adjacency: Sequence[AdjacencyRecord],
    communication: Sequence[CommunicationEvent],
    chains: Sequence[CommunicationChain],
    candidates: Sequence[FusionCandidate],
    output_dir: Path,
) -> None:
    outputs = (
        ("kernel_adjacency.csv", AdjacencyRecord, adjacency),
        ("communication_events.csv", CommunicationEvent, communication),
        ("communication_chains.csv", CommunicationChain, chains),
        ("fusion_candidates.csv", FusionCandidate, candidates),
    )
    for filename, record_type, rows in outputs:
        write_csv(
            output_dir / filename,
            tuple(record_type.__dataclass_fields__),
            [row.__dict__ for row in rows],
        )
    summaries = summarize_communication(communication)
    summary_fields = (
        "device_id", "provider", "collective", "count", "total_duration_ns",
        "average_ns", "p50_ns", "p95_ns", "compute_overlap_ns",
        "exposed_communication_ns", "exposed_ratio",
    )
    write_csv(output_dir / "communication_summary.csv", summary_fields, summaries)
    skew = summarize_arrival_skew(communication)
    skew_fields = (
        "collective", "occurrence", "device_count", "earliest_start_ns",
        "latest_start_ns", "arrival_skew_ns", "evidence", "confidence",
    )
    write_csv(output_dir / "communication_arrival_skew.csv", skew_fields, skew)
