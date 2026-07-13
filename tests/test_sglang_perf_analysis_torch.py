from __future__ import annotations

import json
import importlib
import os
import re
import tempfile
import unittest
from pathlib import Path

from scripts.tools import sglang_perf_analysis_torch as analyzer
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
    select_trace_files,
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
    def _comm_formatter(self):
        try:
            module = importlib.import_module("scripts.tools.sglang_comm_report_formatter")
        except ModuleNotFoundError:
            module = None
        self.assertIsNotNone(module, "sglang_comm_report_formatter module must exist")
        return module

    def _metadata_fixture(self, root: Path, model_name: str = "TestModel-TP4-P128D16C1") -> tuple[Path, Path, Path, Path]:
        config_path = root / f"config.yaml.{model_name}"
        config_path.write_text(
            json.dumps(
                {
                    "model": {"name": model_name, "tensor_parallel_size": 4},
                    "benchmark": {
                        "scenarios": {
                            "optimized": [
                                {
                                    "name": "p128d16_c1",
                                    "input_len": 128,
                                    "output_len": 16,
                                    "concurrency": 1,
                                }
                            ]
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        trace_dir = root / "results" / model_name / "sglang-torch-raw" / "report-sglang"
        trace_dir.mkdir(parents=True)
        trace_path = trace_dir / "123.0-TP-0.trace.json"
        write_trace(trace_path, [kernel_event("kernel", 1.0)])
        output_dir = root / "reports" / model_name
        output_dir.mkdir(parents=True)
        metadata_path = output_dir / "run_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "model": {
                        "model_name": {"value": model_name, "source": str(config_path)},
                        "tp_size": {"value": 4, "source": str(config_path)},
                    },
                    "benchmark": {
                        "scenario_name": {"value": "p128d16_c1", "source": str(config_path)}
                    },
                    "trace": {
                        "trace_files": {
                            "value": [{"path": str(trace_path)}],
                            "source": str(trace_dir),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return config_path, trace_path, output_dir, metadata_path

    def test_validate_report_metadata_accepts_consistent_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, trace_path, output_dir, metadata_path = self._metadata_fixture(root)
            validator = getattr(analyzer, "validate_report_metadata", lambda **_kwargs: None)

            result = validator(
                config_path=config_path,
                trace_files=[trace_path],
                output_dir=output_dir,
                run_metadata_path=metadata_path,
                expected_model="TestModel-TP4-P128D16C1",
                expected_scenario="p128d16_c1",
                expected_tp_size=4,
            )

        self.assertIsInstance(result, dict)
        self.assertEqual(result["model_name"], "TestModel-TP4-P128D16C1")

    def test_validate_report_metadata_rejects_trace_model_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, trace_path, output_dir, metadata_path = self._metadata_fixture(root)
            wrong_trace = root / "results" / "OtherModel-TP8-P2048D128C1" / "sglang-torch-raw" / "report-sglang" / trace_path.name
            wrong_trace.parent.mkdir(parents=True)
            wrong_trace.write_text(trace_path.read_text(encoding="utf-8"), encoding="utf-8")
            validator = getattr(analyzer, "validate_report_metadata", lambda **_kwargs: None)

            with self.assertRaisesRegex(Exception, "metadata mismatch: report model does not match trace path"):
                validator(
                    config_path=config_path,
                    trace_files=[wrong_trace],
                    output_dir=output_dir,
                    run_metadata_path=metadata_path,
                    expected_model="TestModel-TP4-P128D16C1",
                    expected_scenario="p128d16_c1",
                    expected_tp_size=4,
                )

    def test_validate_report_metadata_requires_run_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, trace_path, output_dir, metadata_path = self._metadata_fixture(root)
            metadata_path.unlink()
            validator = getattr(analyzer, "validate_report_metadata", lambda **_kwargs: None)

            with self.assertRaisesRegex(Exception, "metadata mismatch: run_metadata.json is missing"):
                validator(
                    config_path=config_path,
                    trace_files=[trace_path],
                    output_dir=output_dir,
                    run_metadata_path=metadata_path,
                    expected_model="TestModel-TP4-P128D16C1",
                    expected_scenario="p128d16_c1",
                    expected_tp_size=4,
                )

    def test_validate_report_metadata_rejects_tp_size_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path, trace_path, output_dir, metadata_path = self._metadata_fixture(root)
            validator = getattr(analyzer, "validate_report_metadata", lambda **_kwargs: None)

            with self.assertRaisesRegex(Exception, "metadata mismatch: report TP size does not match config TP size"):
                validator(
                    config_path=config_path,
                    trace_files=[trace_path],
                    output_dir=output_dir,
                    run_metadata_path=metadata_path,
                    expected_model="TestModel-TP4-P128D16C1",
                    expected_scenario="p128d16_c1",
                    expected_tp_size=8,
                )

    def test_comm_formatter_classifies_custom_ar_separately_from_nccl(self) -> None:
        formatter = self._comm_formatter()
        custom = formatter.classify_kernel(
            kernel_name="all_reduce_one_shot_push_kernel<bf16, 4>",
            op_name="sglang::outplace_all_reduce",
            source_file="",
            mappings=[],
        )
        nccl = formatter.classify_kernel(
            kernel_name="ncclDevKernel_AllGather_RING_LL",
            op_name="record_param_comms",
            source_file="sglang/srt/distributed/device_communicators/pynccl.py [source_map:high]",
            mappings=[],
        )

        self.assertEqual(custom.provider, "SGLang CustomAllReduceV2")
        self.assertEqual(custom.op_kind, "Communication/SGLang Custom AllReduce")
        self.assertEqual(custom.communication_type, "custom_all_reduce")
        self.assertEqual(custom.current_judgment, "自研通信算子，不是 NCCL")
        self.assertEqual(nccl.provider, "NCCL")
        self.assertEqual(nccl.op_kind, "Communication/NCCL")
        self.assertEqual(nccl.communication_type, "nccl")
        self.assertEqual(nccl.source_type, "source_map_high")

    def test_comm_formatter_keeps_compute_fusion_non_communication(self) -> None:
        formatter = self._comm_formatter()
        norm = formatter.classify_kernel(
            kernel_name="kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnorm",
            op_name="unknown",
            source_file="sglang/srt/layers/layernorm.py [source_map:medium]",
            mappings=[],
        )
        gemm = formatter.classify_kernel(
            kernel_name="deep_gemm::sm90_fp8_gemm_1d2d_impl",
            op_name="sglang::deep_gemm_fp8_fp8_bf16_nt",
            source_file="",
            mappings=[],
        )
        moe = formatter.classify_kernel(
            kernel_name="fused_moe_kernel",
            op_name="sglang::outplace_fused_experts",
            source_file="",
            mappings=[],
            has_explicit_ep_communication=False,
        )

        self.assertEqual((norm.provider, norm.op_kind, norm.communication_type), ("FlashInfer", "Norm/Fused Norm", "none"))
        self.assertIn("不等价于通信融合", norm.current_judgment)
        self.assertEqual((gemm.provider, gemm.op_kind, gemm.communication_type), ("DeepGEMM", "GEMM/Linear", "none"))
        self.assertEqual((moe.provider, moe.op_kind, moe.communication_type), ("SGLang MoE/Triton", "MoE/Expert", "none"))

    def test_comm_formatter_marks_generic_fwd_kernel_for_source_check(self) -> None:
        formatter = self._comm_formatter()
        result = formatter.classify_kernel(
            kernel_name="_fwd_grouped_kernel_stage1",
            op_name="unknown",
            source_file="",
            mappings=[],
        )

        self.assertEqual(result.op_kind, "Unknown Major Kernel")
        self.assertEqual(result.communication_type, "unknown")
        self.assertTrue(result.need_source_check)
        self.assertEqual(result.confidence, "low")

    def test_comm_formatter_builds_mentor_focus_sections_with_top_ten_only(self) -> None:
        formatter = self._comm_formatter()
        rows = [
            {"kind": "distributed_all_reduce", "op_name": "sglang::outplace_all_reduce", "kernel_name": "all_reduce_one_shot_push_kernel", "source_file": "sglang/srt/distributed/parallel_state.py [source_map:high]", "calls": 100, "total_us": 100000.0},
            {"kind": "norm", "op_name": "unknown", "kernel_name": "flashinfernorm_fused_add_rmsnorm", "source_file": "", "calls": 100, "total_us": 90000.0},
            {"kind": "non_distributed", "op_name": "unknown", "kernel_name": "_fwd_grouped_kernel_stage1", "source_file": "", "calls": 10, "total_us": 80000.0},
            {"kind": "gemm", "op_name": "sglang::deep_gemm_fp8_fp8_bf16_nt", "kernel_name": "deep_gemm::sm90_fp8_gemm_1d2d_impl", "source_file": "", "calls": 20, "total_us": 70000.0},
            {"kind": "moe", "op_name": "sglang::outplace_fused_experts", "kernel_name": "fused_moe_kernel", "source_file": "", "calls": 20, "total_us": 60000.0},
            {"kind": "distributed_all_gather", "op_name": "record_param_comms", "kernel_name": "ncclDevKernel_AllGather_RING_LL", "source_file": "", "calls": 2, "total_us": 5000.0},
        ]
        rows.extend(
            {"kind": "non_distributed", "op_name": "aten::add", "kernel_name": f"at::native::kernel_{index}", "source_file": "", "calls": 1, "total_us": float(4000 - index)}
            for index in range(8)
        )

        markdown, tables = formatter.build_focus_report_sections(
            kernel_rows=rows,
            event_rows=[{"event_name": "gloo:broadcast", "calls": 1, "total_us": 1000.0, "source_file": ""}],
            total_gpu_us=sum(row["total_us"] for row in rows),
            mappings=[],
        )

        self.assertIn("## Top 10 Kernel 源码核查表", markdown)
        self.assertIn("## 通信算子拆解", markdown)
        self.assertIn("## 当前可确认结论", markdown)
        self.assertIn("## 仍需源码确认项", markdown)
        top_section = markdown.split("## Top 10 Kernel 源码核查表", 1)[1].split("## 通信算子拆解", 1)[0]
        self.assertEqual(sum(1 for line in top_section.splitlines() if re.match(r"\|\s*\d+\s*\|", line)), 10)
        self.assertIn("自研通信算子，不是 NCCL", markdown)
        self.assertIn("计算融合，不等价于通信融合", markdown)
        self.assertIn("FP8 GEMM 计算 kernel", markdown)
        self.assertNotIn("distributed_nccl_other", markdown)
        self.assertTrue(any(table["sheet_name"] == "Top10SourceAudit" for table in tables))

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

    def test_select_trace_files_uses_latest_root_trace_per_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report-sglang"
            report_dir.mkdir()
            old_rank0 = report_dir / "1783500627.1491842-TP-0.trace.json.gz"
            latest_rank0 = report_dir / "1783585090.6296785-TP-0.trace.json.gz"
            rank1 = report_dir / "1783585090.6296785-TP-1.trace.json.gz"
            for path in (old_rank0, latest_rank0, rank1):
                path.write_bytes(b"{}")
            archive = report_dir / "archive_before_latest_1783585090.6296785"
            archive.mkdir()
            (archive / "9999999999.0-TP-0.trace.json.gz").write_bytes(b"{}")

            self.assertEqual(select_trace_files(report_dir, "0"), [latest_rank0])
            self.assertEqual(select_trace_files(report_dir, "all"), [latest_rank0, rank1])

    def test_parser_deduplicates_only_identical_gpu_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report-sglang"
            report_dir.mkdir()
            duplicate = kernel_event("ncclDevKernel_AllGather_RING_LL", 20.0)
            duplicate.update({"pid": 3, "tid": 7, "ts": 10.0})
            duplicate["args"].update({"correlation": 77, "device": 0, "stream": 4})
            legitimate = json.loads(json.dumps(duplicate))
            legitimate["ts"] = 11.0
            write_trace(
                report_dir / "1783585090.0-TP-0.trace.json",
                [duplicate, json.loads(json.dumps(duplicate)), legitimate],
            )

            profile = parse_profile_dir_by_rank(
                report_dir,
                rank_selector="0",
                progress_every=0,
                use_cache=False,
            )

        stats = profile.rank_stats[0]
        self.assertEqual(stats.raw_gpu_kernel_events, 3)
        self.assertEqual(stats.raw_gpu_kernel_us, 60.0)
        self.assertEqual(stats.gpu_kernel_events, 2)
        self.assertEqual(stats.total_kernel_us, 40.0)
        self.assertEqual(stats.duplicate_gpu_kernel_events_filtered, 1)
        self.assertEqual(stats.duplicate_gpu_kernel_us_filtered, 20.0)
        self.assertEqual(stats.raw_comm_kernel_events, 3)
        self.assertEqual(stats.dedup_comm_kernel_events, 2)
        self.assertEqual(stats.duplicate_comm_event_filtered_count, 1)
        self.assertEqual(stats.duplicate_comm_event_filtered_us, 20.0)

    def test_kernel_mapping_preserves_provenance_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report-sglang"
            report_dir.mkdir()
            direct = kernel_event("RMSNormKernel", 10.0)
            direct["args"]["Source Location"] = "native/profiler.py:10"
            correlated_cpu = cpu_op_event("aten::mm", 22)
            correlated_cpu["args"]["Source Location"] = "model/layer.py:20"
            correlated = kernel_event("deep_gemm::sm90_fp8_gemm", 20.0, 22)
            mapped = kernel_event("fused_moe_kernel", 30.0)
            source_mapped = kernel_event("source_only_kernel", 40.0)
            write_trace(
                report_dir / "1783585090.0-TP-0.trace.json",
                [direct, correlated_cpu, correlated, mapped, source_mapped],
            )
            kernel_mappings = [
                {
                    "pattern": "RMSNormKernel",
                    "op_name": "flashinfer::rmsnorm",
                    "source_file": "mapping/should_not_replace.py",
                    "source_type": "kernel_name_mapping",
                    "provider": "FlashInfer",
                    "op_kind": "Norm/Fused Norm",
                    "communication_type": "none",
                    "confidence": "medium",
                },
                {
                    "pattern": "fused_moe_kernel",
                    "op_name": "sglang::outplace_fused_experts",
                    "source_file": "sglang/srt/layers/moe/",
                    "source_type": "kernel_name_mapping",
                    "provider": "SGLang MoE/Triton",
                    "op_kind": "MoE/Expert",
                    "communication_type": "none",
                    "confidence": "medium",
                },
            ]
            source_map = [
                {
                    "pattern": "source_only_kernel",
                    "match_field": "kernel_name",
                    "source_file_guess": "sglang/srt/unknown/source_candidate.py",
                    "confidence": "low",
                }
            ]

            profile = parse_profile_dir_by_rank(
                report_dir,
                rank_selector="0",
                progress_every=0,
                use_cache=False,
                source_map=source_map,
                kernel_mappings=kernel_mappings,
            )

        aggs = {key[2]: agg for key, agg in profile.rank_stats[0].kernel_aggs.items()}
        self.assertEqual(aggs["RMSNormKernel"].source_type, "profiler_stack")
        self.assertEqual(aggs["RMSNormKernel"].source_file, "native/profiler.py:10")
        self.assertEqual(aggs["deep_gemm::sm90_fp8_gemm"].source_type, "correlation")
        self.assertEqual(aggs["deep_gemm::sm90_fp8_gemm"].source_file, "model/layer.py:20")
        self.assertEqual(aggs["fused_moe_kernel"].source_type, "kernel_name_mapping")
        self.assertEqual(aggs["fused_moe_kernel"].op_name, "sglang::outplace_fused_experts")
        self.assertEqual(aggs["source_only_kernel"].source_type, "source_map_low")

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
        cuda_idx = markdown.index("## Mentor Style CUDA Kernel（按 op_name 聚合）")
        credibility_idx = markdown.index("# 数据可信度说明")

        self.assertLess(env_idx, op_idx)
        self.assertLess(op_idx, cuda_idx)
        self.assertLess(cuda_idx, credibility_idx)

        front_cuda_section = markdown.split("## Mentor Style CUDA Kernel（按 op_name 聚合）", 1)[1].split("# 数据可信度说明", 1)[0]
        self.assertIn("| source_file | op_name | kernel_name | 调用次数 | 总时间(ms) | 平均时间(us) | pct | pct_denom | overall_pct | source_type | provider | op_kind |", front_cuda_section)
        self.assertIn("flashinfer_attention_kernel", front_cuda_section)
        self.assertNotIn("scheduler.run_batch", front_cuda_section)
        self.assertIn("source_type", front_cuda_section)
        self.assertIn("overall_pct", front_cuda_section)
        self.assertIn("## True GPU Kernel 明细", markdown)

    def test_mentor_pct_uses_primary_allreduce_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "report-sglang"
            report_dir.mkdir()
            write_trace(
                report_dir / "1783585090.0-TP-0.trace.json",
                [
                    cpu_op_event("sglang::outplace_all_reduce", 1),
                    kernel_event("all_reduce_one_shot_push_kernel", 100.0, 1),
                    cpu_op_event("flashinfer::fused_add_rmsnorm", 2),
                    kernel_event("RMSNormKernel", 80.0, 2),
                    cpu_op_event("record_param_comms", 3),
                    kernel_event("ncclDevKernel_AllGather_RING_LL", 20.0, 3),
                ],
            )
            profile = parse_profile_dir_by_rank(
                report_dir,
                rank_selector="0",
                progress_every=0,
                use_cache=False,
            )
            markdown, _ = build_markdown(profile, "0", model_name="TestModel")

        mentor = markdown.split("## Mentor Style CUDA Kernel（按 op_name 聚合）", 1)[1].split("# 数据可信度说明", 1)[0]
        self.assertRegex(mentor, r"sglang::outplace_all_reduce.*50\.00%.*total_kernel.*50\.00%")
        self.assertRegex(mentor, r"flashinfer::fused_add_rmsnorm.*80\.00%.*kernel_excluding_primary_allreduce.*40\.00%")
        self.assertRegex(mentor, r"record_param_comms.*20\.00%.*kernel_excluding_primary_allreduce.*10\.00%")
        self.assertIn("只有主 all_reduce 使用 total denominator；NCCL record_param_comms 使用 residual denominator。", markdown)

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
