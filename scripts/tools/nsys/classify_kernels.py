"""Ordered, evidence-preserving kernel classification rules."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

from .models import ClassifiedKernel, KernelSummary
from .utils import write_csv


RULES = (
    ("NCCL Communication", r"(nccl|allreduce|all_reduce|allgather|all_gather|reduce.?scatter|alltoall|all_to_all|broadcast|barrier|send|recv)", "distributed collective"),
    ("Attention/MLA", r"(flash.?attn|attention|mla|paged.?attention|decode.?attention)", "attention/MLA kernel"),
    ("MoE Routing/Permute", r"(moe.*(route|router|permute)|topk.*route|dispatch)", "MoE routing or permutation"),
    ("MoE Combine/Unpermute", r"(moe.*(combine|unpermute)|unpermute|combine.*expert)", "MoE combine or unpermute"),
    ("MoE Expert GEMM", r"(moe|expert).*(gemm|matmul)|(gemm|matmul).*(moe|expert)", "MoE expert matrix multiply"),
    ("Mamba/SSM", r"(mamba|selective.?scan|\bssm\b|state.?space)", "Mamba/SSM"),
    ("Quant/Dequant", r"(quant|dequant|fp8|int8|scaled_mm)", "quantization/dequantization"),
    ("KV Cache", r"(kv.?cache|cache.*(store|load|copy)|page.*cache)", "KV cache"),
    ("Norm/Activation", r"(rms.?norm|layer.?norm|softmax|silu|gelu|activation)", "normalization or activation"),
    ("Sampling", r"(sampling|multinomial|top.?k|top.?p|argmax)", "token sampling"),
    ("Dense GEMM", r"(gemm|matmul|cutlass|cublas|mma)", "dense matrix multiply"),
    ("Memory/Copy", r"(memcpy|memset|copy|transpose|scatter|gather)", "memory movement"),
)


def base_family(name: str) -> str:
    value = re.sub(r"<.*>", "<...>", name)
    value = re.sub(r"0x[0-9a-fA-F]+", "0x...", value)
    value = re.sub(r"\b\d{3,}\b", "N", value)
    return re.sub(r"\s+", " ", value).strip()


def classify_kernel(name: str) -> Tuple[str, str]:
    lowered = name.lower()
    for category, pattern, evidence in RULES:
        if re.search(pattern, lowered, re.IGNORECASE):
            return category, f"regex:{pattern} ({evidence})"
    if lowered.startswith(("void ", "triton_", "cuda")):
        return "Other", "known kernel prefix without a specialized rule"
    return "Unknown", "no classification rule matched"


def classify_kernels(rows: Iterable[KernelSummary]) -> List[ClassifiedKernel]:
    materialized = list(rows)
    denominator = sum(row.total_ns for row in materialized)
    output = []
    for row in materialized:
        category, rule = classify_kernel(row.name)
        output.append(
            ClassifiedKernel(
                name=row.name,
                base_family=base_family(row.name),
                category=category,
                rule=rule,
                total_ns=row.total_ns,
                instances=row.instances,
                time_percentage=(row.total_ns / denominator * 100.0) if denominator else 0.0,
            )
        )
    return sorted(output, key=lambda row: row.total_ns, reverse=True)


def write_classification(rows: Iterable[ClassifiedKernel], output_dir: Path) -> None:
    materialized = list(rows)
    fields = (
        "name", "base_family", "category", "rule", "total_ns", "instances", "time_percentage"
    )
    write_csv(output_dir / "kernel_classification.csv", fields, [row.__dict__ for row in materialized])
    unknown = [row.__dict__ for row in materialized if row.category == "Unknown"][:30]
    write_csv(output_dir / "unknown_kernels.csv", fields, unknown)
