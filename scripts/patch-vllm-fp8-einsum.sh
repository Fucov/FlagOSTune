#!/usr/bin/env bash
#
# patch-vllm-fp8-einsum.sh - 将 vLLM deepseek_v4_fp8_einsum 替换为 FlagGems 实现
#
# 补丁点:
#   vllm/model_executor/layers/deepseek_v4_attention.py
#     deepseek_v4_fp8_einsum -> flag_gems.runtime.backend._nvidia.hopper.ops.fp8_einsum.fp8_einsum
#
# 行为:
#   用 FlagGems fp8_einsum 实现替换 vLLM DeepSeek V4 attention 中的 einsum 包装函数。
#
# 用法:
#   ./patch-vllm-fp8-einsum.sh --apply
#   ./patch-vllm-fp8-einsum.sh --restore
#   ./patch-vllm-fp8-einsum.sh --status
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/patch-vllm-common.sh"
PYTHON_EXECUTABLE="${Python_EXECUTABLE:-python3}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

PATCH_MARKER="# >>> FLAGGEMS FP8_EINSUM PATCH >>>"
PATCH_END="# <<< FLAGGEMS FP8_EINSUM PATCH <<<"
BAK_SUFFIX=".einsum_bak"

ACTION=""

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
        --apply)   set_action "apply"; shift ;;
        --restore) set_action "restore"; shift ;;
        --status)  set_action "status"; shift ;;
        -h|--help) show_help; exit 0 ;;
        *) log_error "未知参数: $1"; exit 1 ;;
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

TARGET="${VLLM_DIR}/model_executor/layers/deepseek_v4_attention.py"

if [[ ! -f "$TARGET" ]]; then
    log_error "目标文件不存在: $TARGET"
    flagtune_emit_result "TARGET_MISMATCH"
    exit 1
fi

target_matches() {
    [[ -f "$TARGET" ]] && grep -qE "^def deepseek_v4_fp8_einsum\(" "$TARGET" 2>/dev/null
}

is_patched() { flagtune_has_marker_pair "$TARGET" "$PATCH_MARKER" "$PATCH_END"; }

patch_correct() {
    is_patched && \
        flagtune_has_all "$TARGET" \
            "def deepseek_v4_fp8_einsum(" \
            "from flag_gems.runtime.backend._nvidia.hopper.ops.fp8_einsum import (" \
            "fp8_einsum as _fg_fp8_einsum" \
            "out.copy_(result)"
}

patch_state() {
    if ! target_matches; then
        echo "target_mismatch"
    elif patch_correct; then
        if flagtune_backup_exists "$TARGET" "$BAK_SUFFIX"; then
            echo "patched_correct"
        else
            echo "patched_correct_backup_missing"
        fi
    elif flagtune_has_any_marker "$TARGET" "$PATCH_MARKER" "$PATCH_END" || \
         grep -qF "_fg_fp8_einsum" "$TARGET" 2>/dev/null; then
        echo "patched_invalid"
    else
        echo "clean"
    fi
}

backup() {
    if [[ ! -f "${TARGET}${BAK_SUFFIX}" ]]; then
        cp "$TARGET" "${TARGET}${BAK_SUFFIX}"
        log_info "备份: ${TARGET}${BAK_SUFFIX}"
    fi
}

restore() {
    if [[ -f "${TARGET}${BAK_SUFFIX}" ]]; then
        cp "${TARGET}${BAK_SUFFIX}" "$TARGET"
        rm -f "${TARGET}${BAK_SUFFIX}"
        log_info "已还原: $TARGET"
    else
        log_warn "无备份: $TARGET"
    fi
}

check_status() {
    log_info "vLLM 路径: $VLLM_DIR"
    log_info "目标文件: $TARGET"
    if is_patched; then
        log_info "状态: ${GREEN}已打补丁${NC}"
    else
        log_info "状态: ${YELLOW}未打补丁${NC}"
    fi
    if [[ -f "${TARGET}${BAK_SUFFIX}" ]]; then
        log_info "备份: 存在 (${TARGET}${BAK_SUFFIX})"
    else
        log_info "备份: 无"
    fi
}

apply_patch() {
    case "$(patch_state)" in
        patched_correct)
            log_warn "已经有正确 fp8_einsum 补丁，跳过"
            flagtune_emit_result "ALREADY_PATCHED"
            return 0
            ;;
        patched_correct_backup_missing)
            log_warn "已经有正确 fp8_einsum 补丁，但备份缺失"
            flagtune_emit_result "ALREADY_PATCHED_BACKUP_MISSING"
            return 0
            ;;
        patched_invalid)
            log_error "fp8_einsum 补丁不完整或不正确"
            flagtune_emit_result "PATCH_INVALID"
            return 1
            ;;
        target_mismatch)
            log_error "deepseek_v4_attention.py 和 fp8_einsum 补丁脚本预期不匹配"
            flagtune_emit_result "TARGET_MISMATCH"
            return 1
            ;;
    esac

    backup

    "$PYTHON_EXECUTABLE" << 'PYEOF'
import sys
import re

target = sys.argv[1] if len(sys.argv) > 1 else ""
if not target:
    # Read from env
    import os
    target = os.environ.get("PATCH_TARGET", "")

import os
target = os.environ["PATCH_TARGET"]

with open(target, "r") as f:
    content = f.read()

# --- Patch 1: Replace deepseek_v4_fp8_einsum function body ---

old_func = '''\
def deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))'''

new_func = '''\
# >>> FLAGGEMS FP8_EINSUM PATCH >>>
def deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    from flag_gems.runtime.backend._nvidia.hopper.ops.fp8_einsum import (
        fp8_einsum as _fg_fp8_einsum,
    )

    block_size = [recipe[1], recipe[2]]
    result = _fg_fp8_einsum(
        equation,
        a,
        a_scale,
        b,
        b_scale,
        block_size=block_size,
        output_dtype=out.dtype,
    )
    out.copy_(result)
# <<< FLAGGEMS FP8_EINSUM PATCH <<<'''

if old_func not in content:
    print("ERROR: deepseek_v4_fp8_einsum 函数体格式不匹配，无法打补丁", file=sys.stderr)
    print("请检查 vLLM 版本是否兼容", file=sys.stderr)
    sys.exit(1)

new_content = content.replace(old_func, new_func, 1)

with open(target, "w") as f:
    f.write(new_content)

print("OK: deepseek_v4_fp8_einsum 已替换为 FlagGems 实现")
PYEOF
    flagtune_emit_result "APPLIED"
}

# 主逻辑
if [[ "$ACTION" == "status" ]]; then
    check_status
    case "$(patch_state)" in
        patched_correct) flagtune_emit_result "ALREADY_PATCHED" ;;
        patched_correct_backup_missing) flagtune_emit_result "ALREADY_PATCHED_BACKUP_MISSING" ;;
        patched_invalid) flagtune_emit_result "PATCH_INVALID"; exit 1 ;;
        target_mismatch) flagtune_emit_result "TARGET_MISMATCH"; exit 1 ;;
        *) flagtune_emit_result "ALREADY_RESTORED" ;;
    esac
    exit 0
fi

if [[ "$ACTION" == "restore" ]]; then
    case "$(patch_state)" in
        clean)
            log_info "未检测到 fp8_einsum 补丁，无需还原"
            flagtune_emit_result "ALREADY_RESTORED"
            ;;
        patched_correct)
            restore
            flagtune_emit_result "RESTORED"
            ;;
        patched_correct_backup_missing)
            log_error "fp8_einsum 补丁已存在，但备份丢失: ${TARGET}${BAK_SUFFIX}"
            flagtune_emit_result "BACKUP_MISSING"
            exit 1
            ;;
        patched_invalid)
            log_error "fp8_einsum 补丁不完整或不正确，拒绝还原"
            flagtune_emit_result "PATCH_INVALID"
            exit 1
            ;;
        target_mismatch)
            log_error "deepseek_v4_attention.py 和 fp8_einsum 补丁脚本预期不匹配"
            flagtune_emit_result "TARGET_MISMATCH"
            exit 1
            ;;
    esac
    exit 0
fi

log_info "vLLM 路径: $VLLM_DIR"
log_info "补丁目标: $TARGET"
log_info "替换 deepseek_v4_fp8_einsum -> FlagGems fp8_einsum"

export PATCH_TARGET="$TARGET"
apply_patch

log_info "补丁完成！还原命令: $0 --restore"
