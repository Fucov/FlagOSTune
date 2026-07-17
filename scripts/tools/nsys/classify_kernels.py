"""Ordered, evidence-preserving kernel classification rules."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from .models import ClassifiedKernel, KernelSummary
from .utils import write_csv


@dataclass(frozen=True)
class KernelClassification:
    category: str
    rule: str
    confidence: str
    runtime_communication: bool = False

    def __getitem__(self, index: int) -> str:
        return (self.category, self.rule, self.confidence)[index]


RUNTIME_COMMUNICATION_CATEGORIES = {
    "NCCL AllReduce",
    "NCCL AllGather",
    "NCCL ReduceScatter",
    "NCCL AllToAll",
    "Custom AllReduce",
    "Custom AllGather",
    "P2P Send/Recv",
}

COMMUNICATION_EVIDENCE = re.compile(
    r"(all[_]?reduce|all[_]?gather|reduce[_]?scatter|all[_]?to[_]?all|"
    r"nccl|p2p|send|recv|collective)",
    re.IGNORECASE,
)
COMPUTE_EVIDENCE = re.compile(
    r"(rms.?norm|layer.?norm|gemm|matmul|mma|attention|softmax|moe|expert|"
    r"silu|gelu|relu|activation|quant|dequant|top.?k)",
    re.IGNORECASE,
)
EXPLICIT_FUSION_EVIDENCE = re.compile(r"(fused|fusion|fuse[_:]?)", re.IGNORECASE)


def base_family(name: str) -> str:
    value = re.sub(r"<.*>", "<...>", name)
    value = re.sub(r"0x[0-9a-fA-F]+", "0x...", value)
    value = re.sub(r"(?<![A-Za-z0-9])\d{3,}(?![A-Za-z0-9])", "N", value)
    return re.sub(r"\s+", " ", value).strip()


def classify_kernel(
    name: str, *, nvtx: Optional[str] = None, module: Optional[str] = None
) -> KernelClassification:
    lowered = name.lower()
    evidence = " ".join(value for value in (name, nvtx or "", module or "") if value).lower()

    memory_pattern = r"(transpose|permute|copy|memcpy|memset|(?:^|[_:])gather(?:[_:]|$)|(?:^|[_:])scatter(?:[_:]|$)|(?:^|[_:])cat(?:[_:]|$)|layout)"
    if re.search(memory_pattern, lowered):
        return KernelClassification(
            "Memory/Layout Transform", f"priority-1 regex:{memory_pattern}", "HIGH"
        )

    if re.search(r"(nccl.*(?:comminit|initrank)|ncclcomminitrank|comm_init)", lowered):
        return KernelClassification(
            "Communication Init", "priority-2 NCCL communicator initialization", "HIGH"
        )

    communication_rules = (
        ("NCCL AllReduce", r"nccl.*all[_]?reduce"),
        ("NCCL AllGather", r"nccl.*all[_]?gather"),
        ("NCCL ReduceScatter", r"nccl.*reduce[_]?scatter"),
        ("NCCL AllToAll", r"nccl.*all[_]?to[_]?all"),
        ("Custom AllReduce", r"(all_reduce|allreduce).*(two_shot|one_shot|push|pull)|(?:two_shot|one_shot).*(all_reduce|allreduce)|(?:all_reduce_)?(?:two_shot|one_shot)_(?:push|pull|kernel)"),
        ("Custom AllGather", r"(custom.*all[_]?gather|all[_]?gather.*(?:custom|push|pull))"),
        ("P2P Send/Recv", r"(nccl.*(?:send|recv)|p2p.*(?:send|recv)|(?:send|recv).*p2p)"),
    )
    for category, pattern in communication_rules:
        if re.search(pattern, lowered):
            return KernelClassification(category, f"priority-2 regex:{pattern}", "HIGH", True)

    gemm_pattern = r"(deep_gemm|sm90_fp8_gemm|grouped_gemm|gemm|matmul|cutlass|cublas|mma)"
    if re.search(gemm_pattern, lowered):
        if re.search(r"(moe|expert|fused_experts)", evidence):
            category = "MoE GEMM"
            confidence = "HIGH" if nvtx or module else "MEDIUM"
        elif re.search(r"(attention|flash_attn|mla|qkv)", evidence):
            category = "Attention GEMM"
            confidence = "HIGH" if nvtx or module else "MEDIUM"
        elif re.search(r"(dense|mlp|linear)", evidence):
            category = "Dense GEMM"
            confidence = "HIGH" if nvtx or module else "MEDIUM"
        else:
            category = "GEMM (unattributed)"
            confidence = "MEDIUM"
        return KernelClassification(category, f"priority-3 regex:{gemm_pattern}", confidence)

    quant_pattern = r"(scaled_quant|per_token_quant|cast_fp8|fp8_quant|int8_quant|dequant|(?:^|[_:])quant(?:[_:]|$))"
    if re.search(quant_pattern, lowered):
        return KernelClassification("Quant/Dequant", f"priority-4 regex:{quant_pattern}", "HIGH")

    known_rules = (
        ("Normalization", r"(rms.?norm|layer.?norm|group.?norm|batch.?norm|softmax)"),
        ("Attention", r"(flash.?attn|attention|mla|paged.?attention|decode.?attention)"),
        ("MoE Routing", r"(moe|expert).*(route|router|topk|dispatch)"),
        ("MoE Combine", r"(moe|expert).*(combine|unpermute)"),
        ("MoE", r"(moe|fused_expert|expert)"),
        ("Elementwise", r"(silu|gelu|relu|activation|add|mul|div|sub|where|sigmoid)"),
        ("KV Cache", r"(kv.?cache|cache.*(?:store|load)|page.*cache)"),
        ("Sampling", r"(sampling|multinomial|top.?k|top.?p|argmax)"),
        ("Mamba/SSM", r"(mamba|selective.?scan|\bssm\b|state.?space)"),
    )
    for category, pattern in known_rules:
        if re.search(pattern, lowered):
            return KernelClassification(category, f"priority-5 regex:{pattern}", "MEDIUM")
    if lowered.startswith(("void ", "triton_", "cuda")):
        return KernelClassification(
            "Other", "known kernel prefix without a specialized rule", "LOW"
        )
    return KernelClassification("Unknown", "no classification rule matched", "LOW")


def judge_comm_compute_fusion(
    name: str, classification: KernelClassification
) -> tuple[str, str, str]:
    """Conservatively judge whether one kernel fuses communication and compute."""
    has_communication = bool(COMMUNICATION_EVIDENCE.search(name))
    has_compute = bool(COMPUTE_EVIDENCE.search(name))
    has_explicit_fusion = bool(EXPLICIT_FUSION_EVIDENCE.search(name))
    if has_communication and has_compute and has_explicit_fusion:
        return (
            "YES",
            "communication-compute",
            "single kernel name contains explicit fusion, communication, and compute evidence",
        )
    if has_communication and has_compute:
        return (
            "UNKNOWN",
            "possible-communication-compute",
            "communication and compute tokens coexist, but explicit fused-kernel evidence is absent",
        )
    if classification.runtime_communication:
        return (
            "NO",
            "standalone-communication",
            "kernel matches a standalone runtime collective rule",
        )
    if classification.category == "Communication Init":
        return (
            "NO",
            "communication-initialization",
            "kernel is communicator initialization, not inference compute fusion",
        )
    if classification.category == "Unknown":
        return (
            "UNKNOWN",
            "unresolved",
            "kernel classification is unknown; source or profiler-stack evidence is required",
        )
    return (
        "NO",
        "standalone-compute",
        "recognized compute or memory kernel has no communication evidence",
    )


def classify_kernels(rows: Iterable[KernelSummary]) -> List[ClassifiedKernel]:
    materialized = list(rows)
    denominator = sum(row.total_ns for row in materialized)
    output = []
    for row in materialized:
        classification = classify_kernel(row.name)
        fusion_verdict, fusion_type, fusion_evidence = judge_comm_compute_fusion(
            row.name, classification
        )
        output.append(
            ClassifiedKernel(
                name=row.name,
                base_family=base_family(row.name),
                category=classification.category,
                classification_rule=classification.rule,
                classification_confidence=classification.confidence,
                total_ns=row.total_ns,
                instances=row.instances,
                time_percentage=(row.total_ns / denominator * 100.0) if denominator else 0.0,
                fusion_verdict=fusion_verdict,
                fusion_type=fusion_type,
                fusion_evidence=fusion_evidence,
            )
        )
    return sorted(output, key=lambda row: row.total_ns, reverse=True)


def build_operator_hotspots(rows: Iterable[KernelSummary]) -> List[ClassifiedKernel]:
    """Aggregate all measured prefill and decode kernels by normalized family."""
    grouped = {}
    for row in rows:
        family = base_family(row.name)
        values = grouped.setdefault(family, {"total_ns": 0.0, "instances": 0})
        values["total_ns"] += row.total_ns
        values["instances"] += row.instances
    denominator = sum(values["total_ns"] for values in grouped.values())
    output = []
    for family, values in grouped.items():
        classification = classify_kernel(family)
        verdict, fusion_type, evidence = judge_comm_compute_fusion(
            family, classification
        )
        output.append(
            ClassifiedKernel(
                name=family,
                base_family=family,
                category=(
                    "Fused Communication-Compute"
                    if verdict == "YES"
                    else classification.category
                ),
                classification_rule=classification.rule,
                classification_confidence=classification.confidence,
                total_ns=values["total_ns"],
                instances=values["instances"],
                time_percentage=(
                    values["total_ns"] / denominator * 100.0 if denominator else 0.0
                ),
                fusion_verdict=verdict,
                fusion_type=fusion_type,
                fusion_evidence=evidence,
            )
        )
    return sorted(output, key=lambda row: row.total_ns, reverse=True)


def write_classification(rows: Iterable[ClassifiedKernel], output_dir: Path) -> None:
    materialized = list(rows)
    fields = (
        "name", "base_family", "category", "classification_rule",
        "classification_confidence", "total_ns", "instances", "time_percentage",
        "fusion_verdict", "fusion_type", "fusion_evidence",
    )
    write_csv(output_dir / "kernel_classification.csv", fields, [row.__dict__ for row in materialized])
    unknown = [row.__dict__ for row in materialized if row.category == "Unknown"]
    write_csv(output_dir / "unknown_kernels.csv", fields, unknown)


def write_operator_hotspots(
    rows: Iterable[ClassifiedKernel], output_dir: Path
) -> None:
    fields = (
        "base_family", "category", "total_ns", "instances", "time_percentage",
        "fusion_verdict", "fusion_type", "fusion_evidence",
        "classification_confidence", "classification_rule",
    )
    write_csv(
        output_dir / "operator_hotspots.csv",
        fields,
        [row.__dict__ for row in rows],
    )
