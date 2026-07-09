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
    is_gpu_kernel_event,
    extract_rank,
    iter_trace_events,
    parse_profile_dir_by_rank,
    report_type_for,
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


def cuda_runtime_event(name: str, dur: float, correlation: int, external_id: int | None = None) -> dict:
    event = profiler_event(name, "cuda_runtime", dur, external_id)
    event["args"]["correlation"] = correlation
    return event


def correlated_kernel_event(name: str, dur: float, correlation: int) -> dict:
    event = kernel_event(name, dur)
    event["args"]["correlation"] = correlation
    return event


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

        kernel_section = markdown.split("## True GPU Kernel", 1)[1].split("## Profiler Event 热点", 1)[0]
        self.assertIn("void cutlass_gemm_kernel", kernel_section)
        self.assertNotIn("sglang.forward", kernel_section)
        self.assertIn("## Profiler Event 热点（按总时间排序）", markdown)
        self.assertIn("sglang.forward", markdown)
        self.assertIn("FlagOSTune Torch Profiling 之 SGLang Qwen3.6-35B-A3B-FP8 TP4", markdown)

    def test_markdown_front_section_matches_mentor_profile_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report-sglang"
            report_dir.mkdir()
            write_trace(
                report_dir / "worker-rank0.pt.trace.json",
                [
                    cpu_op_event("aten::scaled_dot_product_attention", 1),
                    profiler_event("scheduler.run_batch", "user_annotation", 500.0, 1),
                    kernel_event("flashinfer_attention_kernel", 80.0, 1),
                ],
            )

            profile = parse_profile_dir_by_rank(
                report_dir,
                rank_selector="0",
                progress_every=0,
                use_cache=False,
            )
            markdown, _ = build_markdown(profile, "0", model_name="Qwen3.6-35B-A3B-FP8-TP4-P32768D1024C1")

        env_idx = markdown.index("# 环境")
        op_idx = markdown.index("## 算子数据")
        cuda_idx = markdown.index("## CUDA kernel（按总时间排序）")
        credibility_idx = markdown.index("# 数据可信度说明")

        self.assertLess(env_idx, op_idx)
        self.assertLess(op_idx, cuda_idx)
        self.assertLess(cuda_idx, credibility_idx)

        front_cuda_section = markdown.split("## CUDA kernel（按总时间排序）", 1)[1].split("# 数据可信度说明", 1)[0]
        self.assertIn("| source file | op_name | kernel_name | 调用次数 | 总时间(ms) | 平均时间(us) | 占比 |", front_cuda_section)
        self.assertIn("flashinfer_attention_kernel", front_cuda_section)
        self.assertNotIn("scheduler.run_batch", front_cuda_section)
        self.assertNotIn("source_type", front_cuda_section)
        self.assertNotIn("overall_pct", front_cuda_section)
        self.assertIn("## True GPU Kernel 明细", markdown)

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

    def test_true_gpu_kernel_filter_excludes_profiler_annotations(self) -> None:
        excluded = [
            profiler_event("scheduler.run_batch", "user_annotation", 1000.0),
            profiler_event("step[DECODE bs=1]", "user_annotation", 900.0),
            profiler_event("step[EXTEND bs=1 toks=16128]", "user_annotation", 800.0),
            profiler_event("## Call CompiledFxGraph", "cpu_op", 700.0),
            profiler_event("cudaLaunchKernel", "cuda_runtime", 10.0),
        ]
        for event in excluded:
            self.assertFalse(is_gpu_kernel_event(event), event["name"])

        included = [
            kernel_event("ncclDevKernel_AllGather_RING_LL", 20.0),
            kernel_event("void my_kernel(float*)", 30.0),
            kernel_event("triton__kernel", 40.0),
            kernel_event("cutlass::device_kernel<flashinfer::foo>", 50.0),
            profiler_event("Memcpy DtoH", "gpu_memcpy", 5.0),
        ]
        for event in included:
            self.assertTrue(is_gpu_kernel_event(event), event["name"])

    def test_report_type_classifies_attention_norm_quant_and_linear_attention(self) -> None:
        self.assertEqual(
            report_type_for(OpKind.ATTENTION.value, "", "radix_attention_kernel", ""),
            "Attention",
        )
        self.assertEqual(
            report_type_for(OpKind.MAMBA_OR_LINEAR_ATTENTION.value, "", "hybrid_linear_attn_gated_delta_kernel", ""),
            "Linear Attention / Mamba",
        )
        self.assertEqual(
            report_type_for(OpKind.NORM.value, "", "kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnorm", ""),
            "Norm/Fused Norm",
        )
        self.assertEqual(
            report_type_for(OpKind.NON_DISTRIBUTED.value, "", "per_token_group_quant_8bit_v2", ""),
            "Quantization/Dequantization",
        )

    def test_parser_uses_correlation_metadata_and_does_not_count_comm_annotations_as_gpu_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report-sglang"
            report_dir.mkdir()
            write_trace(
                report_dir / "worker-rank0.pt.trace.json",
                [
                    cpu_op_event("torch.distributed.all_gather", 10),
                    profiler_event("nccl:_all_gather_base", "user_annotation", 300.0, 10),
                    cuda_runtime_event("cudaLaunchKernel", 5.0, correlation=77, external_id=10),
                    correlated_kernel_event("ncclDevKernel_AllGather_RING_LL", 20.0, correlation=77),
                    profiler_event("scheduler.run_batch", "user_annotation", 1000.0),
                ],
            )

            profile = parse_profile_dir_by_rank(
                report_dir,
                rank_selector="0",
                progress_every=0,
                use_cache=False,
            )
            markdown, _ = build_markdown(profile, "0", model_name="Qwen3.6-35B-A3B-FP8-TP4-P32768D1024C1")

        self.assertEqual(profile.total_kernel_us, 20.0)
        self.assertEqual(profile.rank_stats[0].distributed[DistributedOpKind.DISTRIBUTED_ALL_GATHER].calls, 1)
        self.assertIn("torch.distributed.all_gather", markdown)
        kernel_section = markdown.split("## True GPU Kernel", 1)[1].split("## Profiler Event 热点", 1)[0]
        self.assertNotIn("scheduler.run_batch", kernel_section)
        self.assertIn("duplicate_comm_event_filtered_count", markdown)


if __name__ == "__main__":
    unittest.main()
