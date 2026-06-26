#!/usr/bin/env bash
#
# patch-vllm-fused-marlin-moe.sh - 将 vLLM DeepSeekV4 MXFP4 Marlin MoE 路径替换为 FlagGems 实现
#
# 补丁点:
#   1. vllm/model_executor/layers/fused_moe/oracle/mxfp4.py
#      MARLIN/BATCHED_MARLIN 分支保留 raw MXFP4 uint8 packed weights，不做 vLLM Marlin repack。
#   2. vllm/model_executor/layers/fused_moe/fused_marlin_moe.py
#      MarlinExpertsBase.moe_problem_size 按 raw MXFP4 layout 计算 N。
#      MarlinExperts.apply 调用 flag_gems.fused.fused_marlin_moe.fused_marlin_moe。
#
# 用法:
#   ./patch-vllm-fused-marlin-moe.sh --apply
#   ./patch-vllm-fused-marlin-moe.sh --restore
#   ./patch-vllm-fused-marlin-moe.sh --status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/patch-vllm-common.sh"
PYTHON_EXECUTABLE="${Python_EXECUTABLE:-python3}"

ACTION=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

BASE_START_MARKER="# >>> FLAGTUNE FUSED_MARLIN_MOE PROBLEM_SIZE PATCH START >>>"
BASE_END_MARKER="# <<< FLAGTUNE FUSED_MARLIN_MOE PROBLEM_SIZE PATCH END <<<"
APPLY_START_MARKER="# >>> FLAGTUNE FUSED_MARLIN_MOE APPLY PATCH START >>>"
APPLY_END_MARKER="# <<< FLAGTUNE FUSED_MARLIN_MOE APPLY PATCH END <<<"
MXFP4_START_MARKER="# >>> FLAGTUNE FUSED_MARLIN_MOE RAW MXFP4 PATCH START >>>"
MXFP4_END_MARKER="# <<< FLAGTUNE FUSED_MARLIN_MOE RAW MXFP4 PATCH END <<<"
BAK_SUFFIX=".bak.fused_marlin_moe"

show_help() {
    echo "用法: $0 --apply|--restore|--status"
    echo ""
    echo "  --apply    应用补丁"
    echo "  --restore  还原补丁"
    echo "  --status   检查补丁状态"
}

set_action() {
    local next_action="$1"
    if [[ -n "$ACTION" ]]; then
        log_error "只能指定一个动作: --apply, --restore, --status"
        exit 1
    fi
    ACTION="$next_action"
}

if [[ $# -eq 0 ]]; then
    show_help
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --apply)
            set_action "apply"
            shift
            ;;
        --restore)
            set_action "restore"
            shift
            ;;
        --status)
            set_action "status"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$ACTION" ]]; then
    log_error "必须指定一个动作: --apply, --restore, --status"
    show_help
    exit 1
fi

detect_vllm_path() {
    "$PYTHON_EXECUTABLE" -c "
import vllm, os
print(os.path.dirname(vllm.__file__))
" 2>/dev/null || { log_error "无法定位 vllm"; flagtune_emit_result "TARGET_MISMATCH"; exit 1; }
}

if ! VLLM_DIR=$(detect_vllm_path); then
    log_error "无法定位 vllm"
    flagtune_emit_result "TARGET_MISMATCH"
    exit 1
fi

FUSED_FILE="${VLLM_DIR}/model_executor/layers/fused_moe/fused_marlin_moe.py"
MXFP4_FILE="${VLLM_DIR}/model_executor/layers/fused_moe/oracle/mxfp4.py"

if [[ ! -f "$FUSED_FILE" || ! -f "$MXFP4_FILE" ]]; then
    log_error "找不到目标文件"
    log_error "FUSED_FILE=$FUSED_FILE"
    log_error "MXFP4_FILE=$MXFP4_FILE"
    flagtune_emit_result "TARGET_MISMATCH"
    exit 1
fi

target_matches() {
    grep -qE "^class MarlinExpertsBase\\(" "$FUSED_FILE" 2>/dev/null && \
    grep -qE "^class MarlinExperts\\(" "$FUSED_FILE" 2>/dev/null && \
    grep -qE "^def convert_weight_to_mxfp4_moe_kernel_format\\(" "$MXFP4_FILE" 2>/dev/null
}

is_patched() {
    flagtune_has_marker_pair "$FUSED_FILE" "$BASE_START_MARKER" "$BASE_END_MARKER" && \
    flagtune_has_marker_pair "$FUSED_FILE" "$APPLY_START_MARKER" "$APPLY_END_MARKER" && \
    flagtune_has_marker_pair "$MXFP4_FILE" "$MXFP4_START_MARKER" "$MXFP4_END_MARKER"
}

patch_correct() {
    is_patched && \
    flagtune_has_all "$FUSED_FILE" \
        "_flag_gems_fused_marlin_moe" \
        "group_size=32" \
        "FlagGems fused_marlin_moe does not support expert_map" \
        "N = w1.size(1) // 2" && \
    flagtune_has_all "$MXFP4_FILE" \
        "FlagGems fused_marlin_moe expects raw MXFP4 packed weights" \
        "w13_weight.data.contiguous()" \
        "w2_weight.data.contiguous()" \
        "w13_weight_scale.data.contiguous()" \
        "w2_weight_scale.data.contiguous()"
}

patch_state() {
    if ! target_matches; then
        echo "target_mismatch"
    elif patch_correct; then
        if flagtune_backup_exists "$FUSED_FILE" "$BAK_SUFFIX" && \
           flagtune_backup_exists "$MXFP4_FILE" "$BAK_SUFFIX"; then
            echo "patched_correct"
        else
            echo "patched_correct_backup_missing"
        fi
    elif flagtune_has_any_marker "$FUSED_FILE" "$BASE_START_MARKER" "$BASE_END_MARKER" || \
         flagtune_has_any_marker "$FUSED_FILE" "$APPLY_START_MARKER" "$APPLY_END_MARKER" || \
         flagtune_has_any_marker "$MXFP4_FILE" "$MXFP4_START_MARKER" "$MXFP4_END_MARKER" || \
         grep -qF "_flag_gems_fused_marlin_moe" "$FUSED_FILE" 2>/dev/null; then
        echo "patched_invalid"
    else
        echo "clean"
    fi
}

check_status() {
    log_info "vLLM 路径: $VLLM_DIR"
    log_info "目标文件: $FUSED_FILE"
    log_info "MXFP4 文件: $MXFP4_FILE"
    local state
    state="$(patch_state)"
    case "$state" in
        patched_correct)
            log_info "fused-marlin-moe 状态: ${GREEN}已打补丁${NC}"
            flagtune_emit_result "ALREADY_PATCHED"
            ;;
        patched_correct_backup_missing)
            log_warn "fused-marlin-moe 状态: 已打补丁，但备份缺失"
            flagtune_emit_result "ALREADY_PATCHED_BACKUP_MISSING"
            ;;
        patched_invalid)
            log_error "fused-marlin-moe 状态: 补丁不完整或不正确"
            flagtune_emit_result "PATCH_INVALID"
            ;;
        target_mismatch)
            log_error "fused-marlin-moe 状态: 目标文件结构不匹配"
            flagtune_emit_result "TARGET_MISMATCH"
            ;;
        clean)
            log_info "fused-marlin-moe 状态: ${YELLOW}未打补丁${NC}"
            flagtune_emit_result "ALREADY_RESTORED"
            ;;
    esac
    if [[ -f "${FUSED_FILE}${BAK_SUFFIX}" ]]; then
        log_info "备份: ${FUSED_FILE}${BAK_SUFFIX}"
    fi
    if [[ -f "${MXFP4_FILE}${BAK_SUFFIX}" ]]; then
        log_info "备份: ${MXFP4_FILE}${BAK_SUFFIX}"
    fi
    true
}

backup_targets() {
    flagtune_backup_file "$FUSED_FILE" "$BAK_SUFFIX"
    flagtune_backup_file "$MXFP4_FILE" "$BAK_SUFFIX"
}

restore_targets() {
    local restored=0
    if flagtune_backup_exists "$FUSED_FILE" "$BAK_SUFFIX"; then
        cp "${FUSED_FILE}${BAK_SUFFIX}" "$FUSED_FILE"
        rm -f "${FUSED_FILE}${BAK_SUFFIX}"
        log_info "已还原: $FUSED_FILE"
        restored=1
    else
        log_warn "无 fused_marlin_moe 备份: ${FUSED_FILE}${BAK_SUFFIX}"
    fi

    if flagtune_backup_exists "$MXFP4_FILE" "$BAK_SUFFIX"; then
        cp "${MXFP4_FILE}${BAK_SUFFIX}" "$MXFP4_FILE"
        rm -f "${MXFP4_FILE}${BAK_SUFFIX}"
        log_info "已还原: $MXFP4_FILE"
        restored=1
    else
        log_warn "无 MXFP4 备份: ${MXFP4_FILE}${BAK_SUFFIX}"
    fi

    if [[ "$restored" -eq 1 ]]; then
        "$PYTHON_EXECUTABLE" -m py_compile "$FUSED_FILE" "$MXFP4_FILE"
        flagtune_emit_result "RESTORED"
    else
        flagtune_emit_result "ALREADY_RESTORED"
    fi
}

apply_targets() {
    case "$(patch_state)" in
        patched_correct)
            log_warn "已经有正确 fused-marlin-moe 补丁，跳过"
            flagtune_emit_result "ALREADY_PATCHED"
            return 0
            ;;
        patched_correct_backup_missing)
            log_warn "已经有正确 fused-marlin-moe 补丁，但备份缺失"
            flagtune_emit_result "ALREADY_PATCHED_BACKUP_MISSING"
            return 0
            ;;
        patched_invalid)
            log_error "检测到不完整或非预期 fused-marlin-moe 补丁，请先还原 vLLM 文件"
            flagtune_emit_result "PATCH_INVALID"
            return 1
            ;;
        target_mismatch)
            log_error "vLLM 文件结构与脚本预期不匹配"
            flagtune_emit_result "TARGET_MISMATCH"
            return 1
            ;;
    esac

    backup_targets

    export FUSED_FILE MXFP4_FILE
    export BASE_START_MARKER BASE_END_MARKER APPLY_START_MARKER APPLY_END_MARKER
    export MXFP4_START_MARKER MXFP4_END_MARKER

    "$PYTHON_EXECUTABLE" <<'PYEOF'
import ast
import os
import tempfile
from pathlib import Path

fused_file = Path(os.environ["FUSED_FILE"])
mxfp4_file = Path(os.environ["MXFP4_FILE"])

base_start = os.environ["BASE_START_MARKER"]
base_end = os.environ["BASE_END_MARKER"]
apply_start = os.environ["APPLY_START_MARKER"]
apply_end = os.environ["APPLY_END_MARKER"]
mxfp4_start = os.environ["MXFP4_START_MARKER"]
mxfp4_end = os.environ["MXFP4_END_MARKER"]


def atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def find_method(tree: ast.AST, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise RuntimeError(f"Cannot find {class_name}.{method_name}")


def replace_node_lines(text: str, node: ast.AST, replacement: str) -> str:
    lines = text.splitlines()
    start = node.lineno - 1
    end = node.end_lineno
    new_lines = replacement.rstrip("\n").splitlines()
    lines[start:end] = new_lines
    return "\n".join(lines) + "\n"


problem_size_replacement = f'''    {base_start}
    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        assert w1.dim() == 3 and w2.dim() == 3

        E = w1.size(0)
        K = a1.size(-1)
        if self.quant_config.use_mxfp4_w4a16 or self.quant_config.use_nvfp4_w4a16:
            # FlagGems consumes raw MXFP4 layout:
            #   w1: (E, 2N, K // 2), w2: (E, K, N // 2)
            N = w1.size(1) // 2
        else:
            N = marlin_moe_intermediate_size(w1, w2)

        if a1.dim() == 2:
            assert topk_ids.size(0) == a1.size(0), f"{{topk_ids.size(0)}} != {{a1.size(0)}}"
            M = a1.size(0)
        else:
            assert a1.dim() == 3
            assert a1.size(0) == E, f"{{a1.size(0)}} == {{E}}"
            M = a1.size(1)

        assert topk_ids.dim() == 2
        topk = topk_ids.size(1)

        return E, M, N, K, topk
    {base_end}'''


apply_replacement = f'''    {apply_start}
    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert self.w1_scale is not None
        assert self.w2_scale is not None

        ctx = self._lora_context
        if ctx is not None:
            raise NotImplementedError("FlagTune fused_marlin_moe patch does not support LoRA path")
        if expert_map is not None:
            raise NotImplementedError(
                "FlagGems fused_marlin_moe does not support expert_map in the end-to-end patch"
            )

        from vllm.scalar_type import scalar_types as _flagtune_vllm_scalar_types
        from flag_gems.fused.fused_marlin_moe import (
            fused_marlin_moe as _flag_gems_fused_marlin_moe,
        )

        _flagtune_quant_type_map = {{
            _flagtune_vllm_scalar_types.uint4b8.id: 0,
            _flagtune_vllm_scalar_types.uint8b128.id: 1,
            _flagtune_vllm_scalar_types.float4_e2m1f.id: 6,
        }}
        _flagtune_quant_type_id = _flagtune_quant_type_map.get(
            self.quant_type_id, self.quant_type_id
        )

        import os as _flagtune_os
        if _flagtune_os.getenv("FLAGTUNE_DEBUG_FUSED_MARLIN_MOE") == "1":
            import torch as _flagtune_torch
            _flagtune_cap = (
                _flagtune_torch.cuda.get_device_capability()
                if _flagtune_torch.cuda.is_available()
                else None
            )
            print(
                "[FLAGTUNE][fused_marlin_moe] "
                f"cuda_available={{_flagtune_torch.cuda.is_available()}} "
                f"cuda_cap={{_flagtune_cap}} "
                f"hidden_dtype={{hidden_states.dtype}} "
                f"w1_dtype={{w1.dtype}} w2_dtype={{w2.dtype}} "
                f"w1_shape={{tuple(w1.shape)}} w2_shape={{tuple(w2.shape)}} "
                f"w1_scale_dtype={{self.w1_scale.dtype}} "
                f"w2_scale_dtype={{self.w2_scale.dtype}} "
                f"w1_scale_shape={{tuple(self.w1_scale.shape)}} "
                f"w2_scale_shape={{tuple(self.w2_scale.shape)}} "
                f"quant_type_id={{_flagtune_quant_type_id}} "
                f"bias1={{self.w1_bias is not None}} bias2={{self.w2_bias is not None}} "
                f"expert_map={{expert_map is not None}} "
                f"global_num_experts={{global_num_experts}}",
                flush=True,
            )

        _flag_gems_fused_marlin_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            bias1=self.w1_bias,
            bias2=self.w2_bias,
            w1_scale=self.w1_scale,
            w2_scale=self.w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_type_id=_flagtune_quant_type_id,
            apply_router_weight_on_input=apply_router_weight_on_input,
            # FlagGems MXFP4 fast path only uses global_num_experts in its
            # precondition guard. vLLM may pass a global count that does not
            # match the local/raw weight expert dimension, so let FlagGems
            # infer it from w1 instead.
            global_num_experts=-1,
            activation=activation,
            expert_map=None,
            output=output,
            group_size=32,
        )
        return
    {apply_end}'''


text = fused_file.read_text(encoding="utf-8")
tree = ast.parse(text)
for class_name, method_name, replacement in [
    ("MarlinExperts", "apply", apply_replacement),
    ("MarlinExpertsBase", "moe_problem_size", problem_size_replacement),
]:
    tree = ast.parse(text)
    node = find_method(tree, class_name, method_name)
    text = replace_node_lines(text, node, replacement)
atomic_write(fused_file, text)


def patch_raw_mxfp4_function(text: str, func_name: str) -> str:
    tree = ast.parse(text)
    func = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            func = node
            break
    if func is None:
        raise RuntimeError(f"Cannot find {func_name}")

    lines = text.splitlines()
    body_start = func.body[0].lineno - 1
    insert_at = None
    for idx in range(body_start, func.end_lineno):
        if "if mxfp4_backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4" in lines[idx]:
            depth = len(lines[idx]) - len(lines[idx].lstrip())
            seen_deepgemm_return = False
            for j in range(idx + 1, func.end_lineno):
                stripped = lines[j].strip()
                indent = len(lines[j]) - len(lines[j].lstrip())
                if stripped.startswith("return ("):
                    seen_deepgemm_return = True
                    continue
                if (
                    seen_deepgemm_return
                    and indent == depth
                    and (
                        stripped.startswith("if ")
                        or stripped.startswith("elif ")
                        or stripped.startswith("num_experts =")
                    )
                ):
                    insert_at = j
                    break
            break
    if insert_at is None:
        raise RuntimeError(f"Cannot find insertion point in {func_name}")

    block = f'''    {mxfp4_start}
    if mxfp4_backend in (
        Mxfp4MoeBackend.MARLIN,
        Mxfp4MoeBackend.BATCHED_MARLIN,
    ):
        # FlagGems fused_marlin_moe expects raw MXFP4 packed weights/scales.
        # vLLM Marlin repack would change the layout before MarlinExperts.apply().
        return (
            w13_weight.data.contiguous(),
            w2_weight.data.contiguous(),
            w13_weight_scale.data.contiguous(),
            w2_weight_scale.data.contiguous(),
            w13_bias,
            w2_bias,
        )
    {mxfp4_end}'''.splitlines()
    lines[insert_at:insert_at] = block + [""]
    return "\n".join(lines) + "\n"


text = mxfp4_file.read_text(encoding="utf-8")
for func_name in [
    "convert_gpt_oss_weight_to_mxfp4_moe_kernel_format",
    "convert_weight_to_mxfp4_moe_kernel_format",
]:
    text = patch_raw_mxfp4_function(text, func_name)
atomic_write(mxfp4_file, text)
PYEOF

    "$PYTHON_EXECUTABLE" -m py_compile "$FUSED_FILE" "$MXFP4_FILE"
    log_info "fused-marlin-moe 补丁已应用"
    flagtune_emit_result "APPLIED"
}

case "$ACTION" in
    status)
        check_status
        ;;
    apply)
        apply_targets
        ;;
    restore)
        restore_targets
        ;;
esac
