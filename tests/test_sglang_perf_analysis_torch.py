from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.tools.sglang_perf_analysis_torch import (
    DistributedOpKind,
    OpKind,
    build_markdown,
    classify_distributed_op,
    classify_op_kind,
    extract_rank,
    iter_trace_events,
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


def profiler_event(name: str, cat: str, dur: float, external_id: int | None = None) -> dict:
    args = {}
    if external_id is not None:
        args["External id"] = external_id
    return {
        "ph": "X",
        "cat": cat,
        "name": name,
        "ts": 2.0,
        "dur": dur,
        "tid": 7,
        "args": args,
    }


class SGLangPerfAnalysisTorchTest(unittest.TestCase):
    def test_classify_distributed_op_covers_common_collectives(self) -> None:
        self.assertEqual(
            classify_distributed_op("ncclDevKernel_AllReduce_RING_LL", ""),
            DistributedOpKind.DISTRIBUTED_ALL_REDUCE,
        )
        self.assertEqual(
            classify_distributed_op("ncclDevKernel_AllGather_RING_LL", ""),
            DistributedOpKind.DISTRIBUTED_ALL_GATHER,
        )
        self.assertEqual(
            classify_distributed_op("ncclDevKernel_ReduceScatter_RING_LL", ""),
            DistributedOpKind.DISTRIBUTED_REDUCE_SCATTER,
        )
        self.assertEqual(
            classify_distributed_op("ncclDevKernel_AllToAll", ""),
            DistributedOpKind.DISTRIBUTED_ALL_TO_ALL,
        )
        self.assertEqual(
            classify_distributed_op("void ncclBroadcastKernel", ""),
            DistributedOpKind.DISTRIBUTED_BROADCAST,
        )
        self.assertEqual(
            classify_distributed_op("ncclKernel_SendRecv", ""),
            DistributedOpKind.DISTRIBUTED_P2P,
        )
        self.assertIsNone(classify_distributed_op("sglang::launch_attention", ""))

    def test_classify_op_kind_keeps_cutlass_flash_attention_non_distributed(self) -> None:
        flash_kernel = "cutlass::device_kernel<flash::FlashAttnFwdSm90<CollectiveMainloopFwdSm90, CollectiveEpilogueFwd>>"
        self.assertEqual(classify_op_kind(flash_kernel, ""), OpKind.ATTENTION)
        self.assertIsNone(classify_distributed_op(flash_kernel, ""))
        self.assertEqual(classify_op_kind("flash::prepare_varlen_num_blocks_kernel", ""), OpKind.ATTENTION)

    def test_classify_op_kind_covers_requested_distributed_variants(self) -> None:
        self.assertEqual(
            classify_op_kind("ncclDevKernel_AllReduce_RING_LL", ""),
            OpKind.DISTRIBUTED_ALL_REDUCE,
        )
        self.assertEqual(
            classify_op_kind("some_kernel", "torch.ops._C_custom_ar::all_reduce"),
            OpKind.DISTRIBUTED_ALL_REDUCE,
        )
        self.assertEqual(
            classify_op_kind("ncclDevKernel_AllGather_RING_LL", ""),
            OpKind.DISTRIBUTED_ALL_GATHER,
        )
        self.assertEqual(
            classify_op_kind("sglang_reduce_scatter_kernel", ""),
            OpKind.DISTRIBUTED_REDUCE_SCATTER,
        )
        self.assertEqual(
            classify_op_kind("sglang::all_to_all_kernel", ""),
            OpKind.DISTRIBUTED_ALL_TO_ALL,
        )
        self.assertEqual(
            classify_op_kind("sglang::alltoall_kernel", ""),
            OpKind.DISTRIBUTED_ALL_TO_ALL,
        )

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
        self.assertEqual(profile.rank_stats[0].distributed[DistributedOpKind.DISTRIBUTED_ALL_REDUCE].calls, 1)
        self.assertEqual(profile.rank_stats[0].distributed[DistributedOpKind.DISTRIBUTED_ALL_REDUCE].total_us, 30.0)
        self.assertEqual(profile.rank_stats[1].distributed[DistributedOpKind.DISTRIBUTED_ALL_REDUCE].total_us, 50.0)
        self.assertEqual(profile.rank_stats[1].distributed[DistributedOpKind.DISTRIBUTED_ALL_GATHER].total_us, 20.0)
        self.assertEqual(profile.total_kernel_us, 170.0)

    def test_tp1_flash_attention_report_has_zero_distributed_and_sanity_hint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report-sglang"
            report_dir.mkdir()
            write_trace(
                report_dir / "worker-rank0.pt.trace.json",
                [
                    kernel_event(
                        "cutlass::device_kernel<flash::FlashAttnFwdSm90<CollectiveMainloopFwdSm90, CollectiveEpilogueFwd>>",
                        120.0,
                    ),
                ],
            )

            profile = parse_profile_dir_by_rank(report_dir, rank_selector="all")
            markdown, _ = build_markdown(profile, "all")

        self.assertEqual(profile.distributed_total_us, 0.0)
        self.assertIn("distributed kernel time(ms) | 0.000", markdown)
        self.assertIn("当前 profile 未检测到真实多卡通信，可能是 TP=1 smoke test", markdown)
        self.assertIn("attention", markdown)

    def test_parse_profile_dir_by_rank_filters_single_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report-sglang"
            report_dir.mkdir()
            write_trace(report_dir / "worker-rank0.pt.trace.json", [kernel_event("ncclDevKernel_AllReduce", 30.0)])
            write_trace(report_dir / "worker-rank1.pt.trace.json", [kernel_event("ncclDevKernel_AllReduce", 50.0)])

            profile = parse_profile_dir_by_rank(report_dir, rank_selector="1")

        self.assertEqual(sorted(profile.rank_stats.keys()), [1])
        self.assertEqual(profile.total_kernel_us, 50.0)

    def test_iter_trace_events_streams_plain_json_trace_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "worker-rank0.pt.trace.json"
            write_trace(trace, [kernel_event("kernel_a", 1.0), kernel_event("kernel_b", 2.0)])

            names = [event["name"] for event in iter_trace_events(trace)]

        self.assertEqual(names, ["kernel_a", "kernel_b"])

    def test_markdown_separates_gpu_kernels_from_profiler_hotspots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report-sglang"
            report_dir.mkdir()
            write_trace(
                report_dir / "worker-rank0.pt.trace.json",
                [
                    cpu_op_event("aten::mm", 1),
                    profiler_event("sglang.forward", "python_function", 500.0, 1),
                    profiler_event("Model.layers.0", "nn_module", 400.0, 1),
                    kernel_event("void cutlass_gemm_kernel", 80.0, 1),
                ],
            )

            profile = parse_profile_dir_by_rank(
                report_dir,
                rank_selector="0",
                progress_every=0,
                use_cache=False,
            )
            markdown, _ = build_markdown(profile, "0", model_name="Qwen3.6-35B-A3B-FP8-TP4-P32768D1024C1")

        kernel_section = markdown.split("## Profiler Event 热点", 1)[0]
        self.assertIn("void cutlass_gemm_kernel", kernel_section)
        self.assertNotIn("sglang.forward", kernel_section)
        self.assertIn("## Profiler Event 热点（按总时间排序）", markdown)
        self.assertIn("sglang.forward", markdown)
        self.assertIn("FlagOSTune Torch Profiling 之 SGLang Qwen3.6-35B-A3B-FP8 TP4", markdown)

    def test_parse_profile_dir_by_rank_reuses_cache_when_trace_metadata_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_dir = root / "report-sglang"
            output_dir = root / "reports" / "model"
            report_dir.mkdir()
            write_trace(report_dir / "worker-rank0.pt.trace.json", [kernel_event("ncclDevKernel_AllReduce", 30.0)])

            first = parse_profile_dir_by_rank(
                report_dir,
                rank_selector="0",
                output_dir=output_dir,
                progress_every=0,
            )
            cache_file = output_dir / "cache" / "rank0_kernel_agg.json"
            self.assertTrue(cache_file.exists())
            first_mtime = os.path.getmtime(cache_file)

            second = parse_profile_dir_by_rank(
                report_dir,
                rank_selector="0",
                output_dir=output_dir,
                progress_every=0,
            )

            self.assertEqual(first.total_kernel_us, second.total_kernel_us)
            self.assertEqual(first_mtime, os.path.getmtime(cache_file))


if __name__ == "__main__":
    unittest.main()
