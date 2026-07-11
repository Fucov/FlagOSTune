#!/usr/bin/env python3
"""Format mentor-focused SGLang communication and fusion report sections."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - server profiling environment provides PyYAML
    yaml = None


@dataclass(frozen=True)
class KernelClassification:
    provider: str
    op_kind: str
    communication_type: str
    source_type: str
    confidence: str
    need_source_check: bool
    evidence: str
    current_judgment: str
    source_files: Tuple[str, ...] = ()


@dataclass(frozen=True)
class KernelSummary:
    kind: str
    op_name: str
    kernel_name: str
    source_file: str
    calls: int
    total_us: float
    classification: KernelClassification

    @property
    def avg_us(self) -> float:
        return self.total_us / self.calls if self.calls else 0.0


def load_kernel_mappings(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file() or yaml is None:
        return []
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, dict):
        payload = payload.get("mappings", [])
    return [item for item in (payload or []) if isinstance(item, dict)]


def infer_source_type(source_file: str, *, used_mapping: Optional[Dict[str, Any]] = None) -> str:
    text = str(source_file or "")
    if "[correlation]" in text:
        return "correlation"
    match = re.search(r"\[source_map:(high|medium|low)\]", text, flags=re.IGNORECASE)
    if match:
        return f"source_map_{match.group(1).lower()}"
    if text:
        return "profiler_stack"
    if used_mapping is not None:
        confidence = str(used_mapping.get("confidence", "low")).lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        return f"source_map_{confidence}"
    return "unknown"


def _match_mapping(
    mappings: Sequence[Dict[str, Any]],
    *,
    kernel_name: str,
    op_name: str,
    source_file: str,
) -> Optional[Dict[str, Any]]:
    all_text = " ".join((kernel_name, op_name, source_file))
    haystacks = {
        "kernel_name": kernel_name,
        "op_name": op_name,
        "operator_name": op_name,
        "event_name": all_text,
        "event_or_kernel_name": all_text,
        "source_file": source_file,
        "all": all_text,
    }
    for item in mappings:
        pattern = str(item.get("pattern") or "")
        if not pattern:
            continue
        field = str(item.get("match_field") or "all")
        target = haystacks.get(field, all_text)
        try:
            matched = re.search(pattern, target, flags=re.IGNORECASE) is not None
        except re.error:
            matched = pattern.lower() in target.lower()
        if matched:
            return item
    return None


def _normalized_provider(value: str) -> str:
    text = value.lower()
    if "customallreducev2" in text:
        return "SGLang CustomAllReduceV2"
    if "customallreduce" in text or "custom allreduce" in text:
        return "SGLang CustomAllreduceV1"
    if "nccl" in text:
        return "NCCL"
    if "gloo" in text or "torch.distributed" in text or "pytorch distributed" in text:
        return "torch.distributed/Gloo"
    if "flashinfer" in text:
        return "FlashInfer"
    if "deepgemm" in text or "deep_gemm" in text:
        return "DeepGEMM"
    if "moe" in text:
        return "SGLang MoE/Triton"
    if "triton" in text or "inductor" in text:
        return "TorchInductor/Triton"
    if "aten" in text or "pytorch" in text:
        return "PyTorch Aten"
    return "Unknown"


def _normalized_op_kind(value: str) -> str:
    text = value.lower()
    if "custom" in text and "all" in text and "reduce" in text:
        return "Communication/SGLang Custom AllReduce"
    if "nccl" in text:
        return "Communication/NCCL"
    if "control" in text or "gloo" in text or "cpu" in text:
        return "Communication/Control Plane"
    if "communication+compute" in text or "allreduce residual" in text or "fused possible" in text:
        return "Communication/Fused Possible"
    if "norm" in text:
        return "Norm/Fused Norm"
    if "gemm" in text or "linear" in text and "attention" not in text:
        return "GEMM/Linear"
    if "moe" in text or "expert" in text:
        return "MoE/Expert"
    if "recurrent" in text or "mamba" in text or "convolution" in text:
        return "Linear Attention / Mamba"
    if "attention" in text:
        return "Attention"
    if "quant" in text:
        return "Quantization"
    if "kv cache" in text or "kvcache" in text:
        return "KV Cache"
    if "unknown" in text:
        return "Unknown Major Kernel"
    if "compute" in text or "gpu kernel" in text:
        return "Other GPU Kernel"
    return "Unknown Major Kernel"


def _normalized_communication_type(value: str) -> str:
    text = value.lower()
    if "custom_all_reduce" in text:
        return "custom_all_reduce"
    if text == "nccl" or "communication/nccl" in text:
        return "nccl"
    if "gloo" in text or "control" in text or "torch_distributed" in text:
        return "gloo_or_control_plane"
    if "flashinfer" in text and ("fusion" in text or "comm" in text):
        return "flashinfer_comm_fusion"
    if "moe" in text and ("possible" in text or "dispatch" in text or "combine" in text):
        return "moe_ep_possible"
    if text in {"none", "none_observed"}:
        return "none"
    return "unknown"


def _result(
    *,
    provider: str,
    op_kind: str,
    communication_type: str,
    source_file: str,
    confidence: str,
    need_source_check: bool,
    evidence: str,
    current_judgment: str,
    source_files: Iterable[str] = (),
    mapping: Optional[Dict[str, Any]] = None,
) -> KernelClassification:
    return KernelClassification(
        provider=provider,
        op_kind=op_kind,
        communication_type=communication_type,
        source_type=infer_source_type(source_file, used_mapping=mapping),
        confidence=confidence if confidence in {"high", "medium", "low"} else "low",
        need_source_check=need_source_check,
        evidence=evidence,
        current_judgment=current_judgment,
        source_files=tuple(str(item) for item in source_files),
    )


def classify_kernel(
    *,
    kernel_name: str,
    op_name: str = "",
    source_file: str = "",
    mappings: Sequence[Dict[str, Any]] = (),
    has_explicit_ep_communication: bool = False,
) -> KernelClassification:
    text = " ".join((kernel_name, op_name, source_file)).lower()

    if any(
        token in text
        for token in (
            "all_reduce_one_shot_push_kernel",
            "all_reduce_one_shot_pull_kernel",
            "all_reduce_one_shot_kernel",
            "all_reduce_two_shot_kernel",
            "two_shot_pull",
        )
    ):
        mapping = _match_mapping(mappings, kernel_name=kernel_name, op_name=op_name, source_file=source_file)
        return _result(
            provider="SGLang CustomAllReduceV2",
            op_kind="Communication/SGLang Custom AllReduce",
            communication_type="custom_all_reduce",
            source_file=source_file,
            confidence="high",
            need_source_check=False,
            evidence="kernel_name + source_map" if mapping else "kernel_name",
            current_judgment="自研通信算子，不是 NCCL",
            source_files=(mapping or {}).get("source_files", ()),
            mapping=mapping,
        )
    if "_c_custom_ar" in text or "cross_device_reduce" in text:
        return _result(
            provider="SGLang CustomAllreduceV1",
            op_kind="Communication/SGLang Custom AllReduce",
            communication_type="custom_all_reduce",
            source_file=source_file,
            confidence="high",
            need_source_check=False,
            evidence="kernel_name + op_name",
            current_judgment="legacy 自研通信算子，不是 NCCL",
        )
    if "nccldevkernel_" in text:
        return _result(
            provider="NCCL",
            op_kind="Communication/NCCL",
            communication_type="nccl",
            source_file=source_file,
            confidence="high",
            need_source_check=False,
            evidence="kernel_name",
            current_judgment="NCCL 通用通信算子",
        )
    if "gloo:" in text or ("torch.distributed.broadcast" in text and "nccl" not in text):
        return _result(
            provider="torch.distributed/Gloo",
            op_kind="Communication/Control Plane",
            communication_type="gloo_or_control_plane",
            source_file=source_file,
            confidence="medium",
            need_source_check="gloo:" not in text,
            evidence="profiler event",
            current_judgment="CPU/control-plane 通信，不计入 GPU 通信时间",
        )
    if "flashinfer_comm_fusion" in text or "flashinfer.comm" in text or "flashinfer_allreduce_residual_rmsnorm" in text:
        return _result(
            provider="FlashInfer",
            op_kind="Communication/Fused Possible",
            communication_type="flashinfer_comm_fusion",
            source_file=source_file,
            confidence="medium",
            need_source_check=True,
            evidence="event/source marker",
            current_judgment="可能的通信融合，需要运行配置和精确 fusion event 共同确认",
        )
    if "flashinfernorm" in text or "fused_add_rmsnorm" in text or "rmsnormkernel" in text:
        return _result(
            provider="FlashInfer",
            op_kind="Norm/Fused Norm",
            communication_type="none",
            source_file=source_file,
            confidence="high",
            need_source_check=False,
            evidence="kernel_name",
            current_judgment="计算融合，不等价于通信融合",
        )
    if "deep_gemm::sm90_fp8_gemm" in text or "sglang::deep_gemm_fp8_fp8_bf16_nt" in text:
        return _result(
            provider="DeepGEMM",
            op_kind="GEMM/Linear",
            communication_type="none",
            source_file=source_file,
            confidence="high",
            need_source_check=False,
            evidence="kernel_name + op_name",
            current_judgment="FP8 GEMM 计算 kernel，未观察到本身融合通信",
        )
    if any(token in text for token in ("fused_moe_kernel", "topkgatingsoftmax", "moe_align", "moe_sum_reduce")):
        comm_type = "moe_ep_possible" if has_explicit_ep_communication else "none"
        judgment = (
            "MoE 计算邻接显式 EP 通信，融合关系仍需确认"
            if has_explicit_ep_communication
            else "本地 MoE routing/expert 计算；无 all_to_all/reduce_scatter 证据时不是确定通信融合"
        )
        return _result(
            provider="SGLang MoE/Triton",
            op_kind="MoE/Expert",
            communication_type=comm_type,
            source_file=source_file,
            confidence="high" if comm_type == "none" else "medium",
            need_source_check=comm_type != "none",
            evidence="kernel_name",
            current_judgment=judgment,
        )
    if "_fwd_grouped_kernel_stage1" in text or re.search(r"(?:^|\W)_fwd_kernel(?:\W|$)", text):
        return _result(
            provider="TorchInductor/Triton",
            op_kind="Unknown Major Kernel",
            communication_type="unknown",
            source_file=source_file,
            confidence="low",
            need_source_check=True,
            evidence="generic kernel name",
            current_judgment="大耗时 Triton kernel，需 correlation/debug stack 确认精确来源",
        )

    mapping = _match_mapping(mappings, kernel_name=kernel_name, op_name=op_name, source_file=source_file)
    if mapping is not None:
        provider = _normalized_provider(str(mapping.get("provider") or "Unknown"))
        op_kind = _normalized_op_kind(str(mapping.get("op_kind") or "Unknown Major Kernel"))
        communication_type = _normalized_communication_type(str(mapping.get("communication_type") or "unknown"))
        confidence = str(mapping.get("confidence") or "low").lower()
        return _result(
            provider=provider,
            op_kind=op_kind,
            communication_type=communication_type,
            source_file=source_file,
            confidence=confidence,
            need_source_check=bool(mapping.get("needs_source_check", mapping.get("need_source_check", True))),
            evidence=str(mapping.get("evidence") or "source_map"),
            current_judgment="按 kernel mapping 分类，结论强度由 confidence/source_type 限定",
            source_files=mapping.get("source_files", ()),
            mapping=mapping,
        )

    if "per_token_group_quant" in text or "quant" in text:
        provider = "TorchInductor/Triton" if "triton" in text else "Unknown"
        op_kind = "Quantization"
    elif "store_kvcache" in text or "kvcache" in text:
        provider, op_kind = "Unknown", "KV Cache"
    elif "fused_recurrent" in text or "causal_conv1d" in text or "gated_delta" in text:
        provider, op_kind = "TorchInductor/Triton", "Linear Attention / Mamba"
    elif "attention" in text or "flashattn" in text:
        provider, op_kind = "Unknown", "Attention"
    elif "aten::" in text or "at::native" in text:
        provider, op_kind = "PyTorch Aten", "Other GPU Kernel"
    elif "triton" in text:
        provider, op_kind = "TorchInductor/Triton", "Other GPU Kernel"
    else:
        provider, op_kind = "Unknown", "Unknown Major Kernel"
    return _result(
        provider=provider,
        op_kind=op_kind,
        communication_type="none" if op_kind != "Unknown Major Kernel" else "unknown",
        source_file=source_file,
        confidence="low",
        need_source_check=op_kind == "Unknown Major Kernel",
        evidence="fallback kernel-name heuristic",
        current_judgment="需要更多 profiler stack/correlation 证据" if op_kind == "Unknown Major Kernel" else "普通 GPU 计算 kernel",
    )


def _md_escape(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", "<br>")


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend(
        "| " + " | ".join(_md_escape(cell) for cell in row) + " |" for row in rows
    )
    return "\n".join(lines)


def _fmt_ms(total_us: float) -> str:
    return f"{total_us / 1000.0:.3f}"


def _fmt_us(total_us: float) -> str:
    return f"{total_us:.3f}"


def _fmt_pct(total_us: float, total_gpu_us: float) -> str:
    return f"{(total_us / total_gpu_us * 100.0) if total_gpu_us else 0.0:.2f}%"


def _candidate_source(row: Dict[str, Any], classification: KernelClassification) -> str:
    source_file = str(row.get("source_file") or "")
    if source_file:
        return re.sub(r"\s*\[(?:source_map:[a-z]+|correlation)\]\s*$", "", source_file)
    return "<br>".join(classification.source_files) or "unknown"


def _explicit_ep_communication(
    kernel_rows: Sequence[Dict[str, Any]], event_rows: Sequence[Dict[str, Any]]
) -> bool:
    texts = [
        " ".join(
            str(row.get(key) or "")
            for key in ("kernel_name", "op_name", "kind", "event_name")
        ).lower()
        for row in (*kernel_rows, *event_rows)
    ]
    return any(
        any(token in text for token in ("all_to_all", "alltoall", "reduce_scatter", "reducescatter", "deepep"))
        for text in texts
    )


def _summaries(
    kernel_rows: Sequence[Dict[str, Any]],
    mappings: Sequence[Dict[str, Any]],
    *,
    has_explicit_ep_communication: bool,
) -> List[KernelSummary]:
    summaries: List[KernelSummary] = []
    for row in kernel_rows:
        classification = classify_kernel(
            kernel_name=str(row.get("kernel_name") or "unknown"),
            op_name=str(row.get("op_name") or "unknown"),
            source_file=str(row.get("source_file") or ""),
            mappings=mappings,
            has_explicit_ep_communication=has_explicit_ep_communication,
        )
        summaries.append(
            KernelSummary(
                kind=str(row.get("kind") or "unknown"),
                op_name=str(row.get("op_name") or "unknown"),
                kernel_name=str(row.get("kernel_name") or "unknown"),
                source_file=_candidate_source(row, classification),
                calls=int(row.get("calls") or 0),
                total_us=float(row.get("total_us") or 0.0),
                classification=classification,
            )
        )
    return sorted(summaries, key=lambda item: item.total_us, reverse=True)


def _communication_label(summary: KernelSummary) -> str:
    comm_type = summary.classification.communication_type
    kernel = summary.kernel_name.lower()
    if comm_type == "custom_all_reduce":
        return "SGLang Custom AllReduce"
    if comm_type == "nccl":
        if "allgather" in kernel:
            return "NCCL AllGather"
        if "allreduce" in kernel:
            return "NCCL AllReduce"
        if "reducescatter" in kernel:
            return "NCCL ReduceScatter"
        if "alltoall" in kernel:
            return "NCCL AllToAll"
        return "NCCL Other Collective"
    if comm_type == "flashinfer_comm_fusion":
        return "FlashInfer Communication Fusion"
    if comm_type == "moe_ep_possible":
        return "MoE EP Communication Possible"
    return "Unknown Communication"


def build_focus_report_sections(
    *,
    kernel_rows: Sequence[Dict[str, Any]],
    event_rows: Sequence[Dict[str, Any]],
    total_gpu_us: float,
    mappings: Sequence[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Build the compact mentor-facing Top/communication/source-audit sections."""
    has_ep_comm = _explicit_ep_communication(kernel_rows, event_rows)
    summaries = _summaries(
        kernel_rows,
        mappings,
        has_explicit_ep_communication=has_ep_comm,
    )
    lines: List[str] = []
    tables: List[Dict[str, Any]] = []

    top_headers = [
        "rank",
        "kernel",
        "op_name",
        "source_file",
        "source_type",
        "provider",
        "op_kind",
        "communication_type",
        "调用次数",
        "总时间(ms)",
        "平均时间(us)",
        "overall_pct",
        "当前判断",
        "需要核查源码",
    ]
    top_rows: List[List[Any]] = []
    for rank, summary in enumerate(summaries[:10], 1):
        classification = summary.classification
        top_rows.append(
            [
                rank,
                summary.kernel_name,
                summary.op_name,
                summary.source_file,
                classification.source_type,
                classification.provider,
                classification.op_kind,
                classification.communication_type,
                summary.calls,
                _fmt_ms(summary.total_us),
                _fmt_us(summary.avg_us),
                _fmt_pct(summary.total_us, total_gpu_us),
                classification.current_judgment,
                str(classification.need_source_check).lower(),
            ]
        )
    lines.extend(("## Top 10 Kernel 源码核查表", "", _md_table(top_headers, top_rows), ""))
    tables.append({"sheet_name": "Top10SourceAudit", "headers": top_headers, "rows": top_rows})

    comm_headers = [
        "通信类型",
        "provider",
        "kernel/event",
        "op_name",
        "source_file",
        "source_type",
        "调用次数",
        "总时间(ms)",
        "平均时间(us)",
        "overall_pct",
        "证据",
        "是否计入 GPU 通信",
    ]
    comm_rows: List[List[Any]] = []
    for summary in summaries:
        classification = summary.classification
        if classification.communication_type not in {
            "custom_all_reduce",
            "nccl",
            "flashinfer_comm_fusion",
            "moe_ep_possible",
        }:
            continue
        comm_rows.append(
            [
                _communication_label(summary),
                classification.provider,
                summary.kernel_name,
                summary.op_name,
                summary.source_file,
                classification.source_type,
                summary.calls,
                _fmt_ms(summary.total_us),
                _fmt_us(summary.avg_us),
                _fmt_pct(summary.total_us, total_gpu_us),
                classification.evidence,
                "yes" if classification.communication_type in {"custom_all_reduce", "nccl"} else "possible",
            ]
        )
    for event in event_rows:
        event_name = str(event.get("event_name") or event.get("name") or "")
        source_file = str(event.get("source_file") or "")
        classification = classify_kernel(
            kernel_name=event_name,
            op_name=event_name,
            source_file=source_file,
            mappings=mappings,
        )
        if classification.communication_type != "gloo_or_control_plane":
            continue
        calls = int(event.get("calls") or 0)
        total_us = float(event.get("total_us") or 0.0)
        comm_rows.append(
            [
                "Control Plane Broadcast",
                classification.provider,
                event_name,
                event_name,
                source_file or "unknown",
                classification.source_type,
                calls,
                _fmt_ms(total_us),
                _fmt_us(total_us / calls if calls else 0.0),
                _fmt_pct(total_us, total_gpu_us),
                classification.evidence,
                "no",
            ]
        )
    lines.extend(("## 通信算子拆解", "", _md_table(comm_headers, comm_rows), ""))
    tables.append({"sheet_name": "CommunicationBreakdown", "headers": comm_headers, "rows": comm_rows})

    custom = [item for item in summaries if item.classification.communication_type == "custom_all_reduce"]
    nccl = [item for item in summaries if item.classification.communication_type == "nccl"]
    norm = [item for item in summaries if item.classification.op_kind == "Norm/Fused Norm"]
    deep_gemm = [item for item in summaries if item.classification.provider == "DeepGEMM"]
    moe = [item for item in summaries if item.classification.op_kind == "MoE/Expert"]
    custom_us = sum(item.total_us for item in custom)
    nccl_us = sum(item.total_us for item in nccl)
    norm_us = sum(item.total_us for item in norm)
    deep_gemm_us = sum(item.total_us for item in deep_gemm)
    moe_names = " / ".join(dict.fromkeys(item.kernel_name for item in moe[:4])) or "未观察到"
    if custom_us >= nccl_us and custom_us > 0:
        primary_comm = "SGLang custom all-reduce"
        primary_evidence = (
            f"custom all-reduce 合计 {_fmt_ms(custom_us)} ms，占 overall {_fmt_pct(custom_us, total_gpu_us)}；"
            f"其中 Top kernel 为 {custom[0].kernel_name}"
        )
    elif nccl_us > 0:
        primary_comm = "NCCL"
        primary_evidence = f"NCCL 合计 {_fmt_ms(nccl_us)} ms，占 overall {_fmt_pct(nccl_us, total_gpu_us)}"
    else:
        primary_comm = "未观察到明确 GPU collective"
        primary_evidence = "custom all-reduce 与 NCCL kernel 时间均为 0"
    lines.extend(
        (
            "## 当前可确认结论",
            "",
            f"1. 当前 rank 的主要通信是 {primary_comm}。证据：{primary_evidence}。",
            "",
            f"2. NCCL 通用通信单独统计，合计 {_fmt_ms(nccl_us)} ms，占 overall {_fmt_pct(nccl_us, total_gpu_us)}，不与 SGLang custom all-reduce 混类。",
            "",
            f"3. FlashInfer fused_add_rmsnorm / RMSNorm 合计 {_fmt_ms(norm_us)} ms，占 overall {_fmt_pct(norm_us, total_gpu_us)}；它属于 Norm/Fused Norm 计算融合，不是通信融合。是否启用 FlashInfer communication fusion 需另查运行配置与精确 fusion event。",
            "",
            f"4. DeepGEMM 合计 {_fmt_ms(deep_gemm_us)} ms，占 overall {_fmt_pct(deep_gemm_us, total_gpu_us)}；它是 FP8 GEMM 计算 kernel，当前未观察到本身融合通信。",
            "",
            f"5. MoE 相关 Top kernel 包括 {moe_names}。当前{'已观察到' if has_ep_comm else '未观察到'}显式 all_to_all / reduce_scatter / DeepEP，因此{'只能标记为 EP communication possible，不能仅凭 MoE kernel 断言融合' if has_ep_comm else '不能断言 MoE EP 通信融合是主瓶颈'}。",
            "",
        )
    )

    pending_headers = ["kernel", "原因", "建议查看文件"]
    pending_rows: List[List[Any]] = []
    for summary in summaries[:10]:
        if summary.classification.need_source_check:
            pending_rows.append(
                [
                    summary.kernel_name,
                    summary.classification.current_judgment,
                    "<br>".join(summary.classification.source_files) or "根据 debug/correlation 查",
                ]
            )
    pending_rows.extend(
        [
            [
                "FlashInfer comm fusion",
                "当前普通 FlashInfer norm 不等价于 communication fusion",
                "sglang/srt/layers/flashinfer_comm_fusion.py<br>sglang/srt/layers/communicator.py",
            ],
            [
                "MoE EP communication",
                "需用 all_to_all/reduce_scatter/dispatch/combine/DeepEP 事件与配置确认",
                "sglang/srt/layers/moe/<br>sglang/srt/distributed/communication_op.py",
            ],
        ]
    )
    lines.extend(("## 仍需源码确认项", "", _md_table(pending_headers, pending_rows), ""))
    lines.append("说明：source_map 是候选源码路径，不是 profiler 原生 stack。")
    lines.append("")
    tables.append({"sheet_name": "NeedsSourceCheck", "headers": pending_headers, "rows": pending_rows})

    return "\n".join(lines), tables
