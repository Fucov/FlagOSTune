#!/usr/bin/env bash
#
# patch-vllm-cp-gather-indexer.sh - 将 vLLM 的 cp_gather_indexer_k_quant_cache
#   替换为 FlagGems Triton 实现
#
# 补丁点:
#   vllm/_custom_ops.py
#     cp_gather_indexer_k_quant_cache -> flag_gems.fused.cp_gather_indexer_k_quant_cache
#
# 行为:
#   用 FlagGems fused CP gather/indexer K quant cache 实现替换 vLLM custom op 包装函数。
#
# 用法:
#   ./patch-vllm-cp-gather-indexer.sh --apply
#   ./patch-vllm-cp-gather-indexer.sh --restore
#   ./patch-vllm-cp-gather-indexer.sh --status
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

PATCH_MARKER="# >>> FLAGGEMS CP_GATHER_INDEXER PATCH >>>"
PATCH_END="# <<< FLAGGEMS CP_GATHER_INDEXER PATCH <<<"
BAK_SUFFIX=".cp_gather_bak"

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

CUSTOM_OPS="${VLLM_DIR}/_custom_ops.py"
TARGETS=("$CUSTOM_OPS")
OTHER_SHARED_MARKERS=(
    "# >>> FLAGTUNE ROUTER_GEMM PATCH START >>>"
    "# >>> FLAGGEMS INDEXER_K_QUANT PATCH >>>"
)

target_matches() {
    [[ -f "$CUSTOM_OPS" ]] && \
        grep -qE "def cp_gather_indexer_k_quant_cache\(" "$CUSTOM_OPS" 2>/dev/null
}

is_patched() { flagtune_has_marker_pair "$1" "$PATCH_MARKER" "$PATCH_END"; }

patch_correct() {
    local f="$1"
    is_patched "$f" && \
        flagtune_has_all "$f" \
            "def cp_gather_indexer_k_quant_cache(" \
            "from flag_gems.fused import cp_gather_indexer_k_quant_cache as _gems_cp_gather" \
            "_gems_cp_gather("
}

patch_state() {
    local f="$1"
    if ! target_matches; then
        echo "target_mismatch"
    elif patch_correct "$f"; then
        if flagtune_backup_exists "$f" "$BAK_SUFFIX"; then
            echo "patched_correct"
        else
            echo "patched_correct_backup_missing"
        fi
    elif flagtune_has_any_marker "$f" "$PATCH_MARKER" "$PATCH_END" || \
         grep -qF "_gems_cp_gather" "$f" 2>/dev/null; then
        echo "patched_invalid"
    else
        echo "clean"
    fi
}

backup() {
    local f="$1"
    if [[ ! -f "${f}${BAK_SUFFIX}" ]]; then
        cp "$f" "${f}${BAK_SUFFIX}"
        log_info "备份: ${f}${BAK_SUFFIX}"
    fi
}

restore() {
    local f="$1"
    if [[ -f "${f}${BAK_SUFFIX}" ]]; then
        cp "${f}${BAK_SUFFIX}" "$f"
        rm -f "${f}${BAK_SUFFIX}"
        log_info "已还原: $f"
    else
        log_warn "无备份: $f"
    fi
}

check_status() {
    echo "=== cp_gather_indexer_k_quant_cache 补丁状态 ==="
    for f in "${TARGETS[@]}"; do
        local name
        name=$(basename "$f")
        if is_patched "$f"; then
            echo -e "  $name: ${GREEN}已补丁${NC}"
        else
            echo -e "  $name: ${YELLOW}未补丁${NC}"
        fi
        if [[ -f "${f}${BAK_SUFFIX}" ]]; then
            echo -e "    备份: ${GREEN}存在${NC}"
        fi
    done
}

patch_cp_gather_indexer() {
    local f="$CUSTOM_OPS"
    case "$(patch_state "$f")" in
        patched_correct)
            log_warn "_custom_ops.py 已有正确 cp_gather_indexer 补丁，跳过"
            flagtune_emit_result "ALREADY_PATCHED"
            return 0
            ;;
        patched_correct_backup_missing)
            log_warn "_custom_ops.py 已有正确 cp_gather_indexer 补丁，但备份缺失"
            flagtune_emit_result "ALREADY_PATCHED_BACKUP_MISSING"
            return 0
            ;;
        patched_invalid)
            log_error "_custom_ops.py 存在不完整或非预期 cp_gather_indexer 补丁"
            flagtune_emit_result "PATCH_INVALID"
            return 1
            ;;
        target_mismatch)
            log_error "_custom_ops.py 和 cp_gather_indexer 补丁脚本预期不匹配"
            flagtune_emit_result "TARGET_MISMATCH"
            return 1
            ;;
    esac
    backup "$f"

    "$PYTHON_EXECUTABLE" - "$f" "$PATCH_MARKER" "$PATCH_END" << 'PYEOF'
import re, sys

filepath = sys.argv[1]
marker = sys.argv[2]
end_marker = sys.argv[3]

with open(filepath, "r") as f:
    content = f.read()

old = """def cp_gather_indexer_k_quant_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
    )"""

new = f"""{marker}
def cp_gather_indexer_k_quant_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    from flag_gems.fused import cp_gather_indexer_k_quant_cache as _gems_cp_gather

    _gems_cp_gather(
        k_cache=kv_cache,
        k_fp8=dst_k,
        k_fp8_scale=dst_scale,
        block_table=block_table,
        cu_seqlen=cu_seq_lens,
    )
{end_marker}"""

if old not in content:
    print("WARN: cp_gather_indexer_k_quant_cache 函数格式不匹配，尝试正则匹配...",
          file=sys.stderr)
    pattern = r'def cp_gather_indexer_k_quant_cache\(\s*\n\s+kv_cache:.*?\n\s+dst_k:.*?\n\s+dst_scale:.*?\n\s+block_table:.*?\n\s+cu_seq_lens:.*?\n\).*?:\n\s+torch\.ops\._C_cache_ops\.cp_gather_indexer_k_quant_cache\(\s*\n\s+kv_cache,\s*dst_k,\s*dst_scale,\s*block_table,\s*cu_seq_lens\s*\n\s+\)'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        old = match.group(0)
        new_content = content.replace(old, new, 1)
    else:
        print("ERROR: 无法匹配 cp_gather_indexer_k_quant_cache", file=sys.stderr)
        sys.exit(1)
else:
    new_content = content.replace(old, new, 1)

with open(filepath, "w") as f:
    f.write(new_content)

print("OK: cp_gather_indexer_k_quant_cache 已替换为 FlagGems Triton 实现")
PYEOF
    flagtune_emit_result "APPLIED"
}

# 主逻辑
if [[ "$ACTION" == "status" ]]; then
    check_status
    case "$(patch_state "$CUSTOM_OPS")" in
        patched_correct) flagtune_emit_result "ALREADY_PATCHED" ;;
        patched_correct_backup_missing) flagtune_emit_result "ALREADY_PATCHED_BACKUP_MISSING" ;;
        patched_invalid) flagtune_emit_result "PATCH_INVALID"; exit 1 ;;
        target_mismatch) flagtune_emit_result "TARGET_MISMATCH"; exit 1 ;;
        *) flagtune_emit_result "ALREADY_RESTORED" ;;
    esac
    exit 0
fi

if [[ "$ACTION" == "restore" ]]; then
    state="$(patch_state "$CUSTOM_OPS")"
    case "$state" in
        clean)
            log_info "未检测到 cp_gather_indexer 补丁，无需还原"
            flagtune_emit_result "ALREADY_RESTORED"
            ;;
        patched_correct)
            if flagtune_restore_function_from_backup \
                "$CUSTOM_OPS" "$BAK_SUFFIX" "$PATCH_MARKER" "$PATCH_END" \
                "cp_gather_indexer_k_quant_cache"; then
                flagtune_emit_result "RESTORED"
            else
                flagtune_emit_result "PATCH_INVALID"
                exit 1
            fi
            ;;
        patched_correct_backup_missing)
            log_error "cp_gather_indexer 补丁已存在，但备份丢失: ${CUSTOM_OPS}${BAK_SUFFIX}"
            flagtune_emit_result "BACKUP_MISSING"
            exit 1
            ;;
        patched_invalid)
            log_error "cp_gather_indexer 补丁不完整或不正确，拒绝还原"
            flagtune_emit_result "PATCH_INVALID"
            exit 1
            ;;
        target_mismatch)
            log_error "_custom_ops.py 和 cp_gather_indexer 补丁脚本预期不匹配"
            flagtune_emit_result "TARGET_MISMATCH"
            exit 1
            ;;
    esac
    exit 0
fi

log_info "vLLM 路径: $VLLM_DIR"
log_info "补丁目标:"
log_info "  _custom_ops.py - cp_gather_indexer_k_quant_cache → FlagGems Triton"

patch_cp_gather_indexer

log_info "补丁完成！还原命令: $0 --restore"
