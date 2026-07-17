"""Conservative same-stream temporal adjacency analysis."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple

from .models import AdjacencyRecord, KernelEvent
from .utils import write_csv


def build_adjacency(events: Iterable[KernelEvent]) -> List[AdjacencyRecord]:
    grouped = defaultdict(list)
    for event in events:
        grouped[(event.device_id, event.context_id, event.stream_id)].append(event)
    output = []
    for (device, context, stream), values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda row: (row.start_ns, row.end_ns, row.event_id))
        for previous, following in zip(ordered, ordered[1:]):
            output.append(
                AdjacencyRecord(
                    device_id=device,
                    context_id=context,
                    stream_id=stream,
                    previous_kernel=previous.name,
                    next_kernel=following.name,
                    gap_ns=following.start_ns - previous.end_ns,
                )
            )
    return output


def load_trace_csv(path: Path) -> List[KernelEvent]:
    """Load the normalized subset of a native `cuda_gpu_trace` CSV."""
    aliases = {
        "start": ("Start (ns)", "Start", "Start Ns"),
        "duration": ("Duration (ns)", "Duration", "Duration Ns"),
        "device": ("Device", "Device ID", "DeviceId"),
        "context": ("Context", "Context ID", "ContextId"),
        "stream": ("Stream", "Stream ID", "StreamId"),
        "name": ("Name", "Kernel Name"),
    }
    from .classify_kernels import base_family, classify_kernel
    from .utils import find_column, parse_duration_ns, parse_number, read_csv_rows

    rows = read_csv_rows(path)
    fields = list(rows[0]) if rows else []
    keys = {name: find_column(fields, values, required=True) for name, values in aliases.items()}
    events = []
    for index, row in enumerate(rows):
        start = int(parse_duration_ns(row[keys["start"]], keys["start"]))
        duration = int(parse_duration_ns(row[keys["duration"]], keys["duration"]))
        name = row[keys["name"]].strip()
        classification = classify_kernel(name)
        events.append(
            KernelEvent(
                index,
                int(parse_number(row[keys["device"]])),
                int(parse_number(row[keys["context"]])),
                int(parse_number(row[keys["stream"]])),
                start,
                start + duration,
                name,
                classification.category,
                base_family(name),
            )
        )
    return events


def write_adjacency(rows: Iterable[AdjacencyRecord], output_dir: Path) -> None:
    materialized = list(rows)
    write_csv(
        output_dir / "kernel_adjacency.csv",
        tuple(AdjacencyRecord.__dataclass_fields__),
        [row.__dict__ for row in materialized],
    )
