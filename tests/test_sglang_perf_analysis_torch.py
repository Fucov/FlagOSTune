import json
import tempfile
import unittest
from pathlib import Path

from scripts.tools.sglang_perf_analysis_torch import (
    DistributedOpKind,
    classify_distributed_op,
    extract_rank,
    parse_profile_dir_by_rank,
)


def write_trace(path: Path, events: list[dict]) -> None:
    path.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")


def kernel_event(name: str, dur: float, external_id: int | None = None) -> dict:
    args = {}
    if external_id is not None:
        args["External id"] = external_id
    return {
        "ph": "X",
        "cat": "kernel",
        "name": name,
        "ts": 10.0,
        "dur": dur,
        "tid": 7,
        "args": args,
    }


def cpu_op_event(name: str, external_id: int) -> dict:
    return {
        "ph": "X",
        "cat": "cpu_op",
        "name": name,
        "ts": 1.0,
        "dur": 1.0,
        "tid": 7,
        "args": {"External id": external_id},
    }


class SGLangPerfAnalysisTorchTest(unittest.TestCase):
    def test_classify_distributed_op_covers_common_collectives(self) -> None:
        self.assertEqual(
            classify_distributed_op("ncclDevKernel_AllReduce_RING_LL", ""),
            DistributedOpKind.ALL_REDUCE,
        )
        self.assertEqual(
            classify_distributed_op("ncclDevKernel_AllGather_RING_LL", ""),
            DistributedOpKind.ALL_GATHER,
        )
        self.assertEqual(
            classify_distributed_op("ncclDevKernel_ReduceScatter_RING_LL", ""),
            DistributedOpKind.REDUCE_SCATTER,
        )
        self.assertEqual(
            classify_distributed_op("ncclDevKernel_AllToAll", ""),
            DistributedOpKind.ALL_TO_ALL,
        )
        self.assertEqual(
            classify_distributed_op("void ncclBroadcastKernel", ""),
            DistributedOpKind.BROADCAST,
        )
        self.assertEqual(
            classify_distributed_op("ncclKernel_SendRecv", ""),
            DistributedOpKind.SEND_RECV,
        )
        self.assertEqual(
            classify_distributed_op("sglang::barrier_wait", ""),
            DistributedOpKind.BARRIER,
        )
        self.assertIsNone(classify_distributed_op("sglang::launch_attention", ""))

    def test_extract_rank_handles_sglang_and_torch_profiler_names(self) -> None:
        self.assertEqual(extract_rank(Path("worker-rank0.pt.trace.json")), 0)
        self.assertEqual(extract_rank(Path("profile-TP-3-DP-0-PP-0-EP-0.trace.json")), 3)
        self.assertEqual(extract_rank(Path("profiler_out_7.txt")), 7)
        self.assertIsNone(extract_rank(Path("no-rank.pt.trace.json")))

    def test_parse_profile_dir_by_rank_aggregates_distributed_ops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report-sglang"
            report_dir.mkdir()
            write_trace(
                report_dir / "worker-rank0.pt.trace.json",
                [
                    cpu_op_event("torch.distributed.all_reduce", 1),
                    kernel_event("ncclDevKernel_AllReduce_RING_LL", 30.0, 1),
                    kernel_event("sglang_attention_kernel", 70.0, 2),
                ],
            )
            write_trace(
                report_dir / "worker-rank1.pt.trace.json",
                [
                    cpu_op_event("torch.distributed.all_reduce", 1),
                    kernel_event("ncclDevKernel_AllReduce_RING_LL", 50.0, 1),
                    kernel_event("ncclDevKernel_AllGather_RING_LL", 20.0, 2),
                ],
            )

            profile = parse_profile_dir_by_rank(report_dir, rank_selector="all")

        self.assertEqual(sorted(profile.rank_stats.keys()), [0, 1])
        self.assertEqual(profile.rank_stats[0].distributed[DistributedOpKind.ALL_REDUCE].calls, 1)
        self.assertEqual(profile.rank_stats[0].distributed[DistributedOpKind.ALL_REDUCE].total_us, 30.0)
        self.assertEqual(profile.rank_stats[1].distributed[DistributedOpKind.ALL_REDUCE].total_us, 50.0)
        self.assertEqual(profile.rank_stats[1].distributed[DistributedOpKind.ALL_GATHER].total_us, 20.0)
        self.assertEqual(profile.total_kernel_us, 170.0)

    def test_parse_profile_dir_by_rank_filters_single_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report-sglang"
            report_dir.mkdir()
            write_trace(report_dir / "worker-rank0.pt.trace.json", [kernel_event("ncclDevKernel_AllReduce", 30.0)])
            write_trace(report_dir / "worker-rank1.pt.trace.json", [kernel_event("ncclDevKernel_AllReduce", 50.0)])

            profile = parse_profile_dir_by_rank(report_dir, rank_selector="1")

        self.assertEqual(sorted(profile.rank_stats.keys()), [1])
        self.assertEqual(profile.total_kernel_us, 50.0)


if __name__ == "__main__":
    unittest.main()
