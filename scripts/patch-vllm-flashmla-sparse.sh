#!/usr/bin/env bash
#
# patch-vllm-flashmla-sparse.sh - 将 vLLM 的 flash_mla_sparse_fwd 替换为 FlagGems Triton 实现
#
# 补丁点:
#   vllm/v1/attention/ops/flashmla.py
#     flash_mla_sparse_fwd -> flag_gems.fused.flash_mla_sparse_fwd wrapper
#
# 行为:
#   在 vLLM FlashMLA op 导入处注入兼容包装层，保留 flash_mla_with_kvcache 等其它导入。
#
# 注意: flash_mla_with_kvcache 由 patch-vllm-flashmla.sh 管理，本脚本不涉及。
#
# 用法:
#   ./patch-vllm-flashmla-sparse.sh --apply
#   ./patch-vllm-flashmla-sparse.sh --restore
#   ./patch-vllm-flashmla-sparse.sh --status
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

PATCH_MARKER="# >>> FLAGGEMS FLASHMLA_SPARSE PATCH >>>"
PATCH_END="# <<< FLAGGEMS FLASHMLA_SPARSE PATCH <<<"
COMPAT_MARKER="_gems_flash_mla_sparse_fwd"
BAK_SUFFIX=".flashmla_sparse_bak"

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

FLASHMLA_OPS="${VLLM_DIR}/v1/attention/ops/flashmla.py"
TARGETS=("$FLASHMLA_OPS")

target_matches() {
    [[ -f "$FLASHMLA_OPS" ]] && {
        grep -qF "flash_mla_sparse_fwd" "$FLASHMLA_OPS" 2>/dev/null || \
        grep -qF "$COMPAT_MARKER" "$FLASHMLA_OPS" 2>/dev/null
    }
}

has_patch_marker() { grep -qF "$PATCH_MARKER" "$1" 2>/dev/null; }
is_patched() { flagtune_has_marker_pair "$1" "$PATCH_MARKER" "$PATCH_END"; }

patch_correct() {
    local f="$1"
    is_patched "$f" && \
        flagtune_has_all "$f" \
            "from flag_gems.fused import (" \
            "flash_mla_sparse_fwd as ${COMPAT_MARKER}" \
            "def flash_mla_sparse_fwd(" \
            "out.copy_(output)" \
            "return output, max_logits, lse"
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
         grep -qF "$COMPAT_MARKER" "$f" 2>/dev/null; then
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
    echo "=== FlashMLA Sparse 补丁状态 ==="
    for f in "${TARGETS[@]}"; do
        local name
        name=$(basename "$f")
        if patch_correct "$f"; then
            echo -e "  $name: ${GREEN}已补丁${NC}"
        elif has_patch_marker "$f"; then
            echo -e "  $name: ${YELLOW}旧版补丁，需升级${NC}"
        else
            echo -e "  $name: ${YELLOW}未补丁${NC}"
        fi
        if [[ -f "${f}${BAK_SUFFIX}" ]]; then
            echo -e "    备份: ${GREEN}存在${NC}"
        fi
    done
}

patch_flashmla_ops() {
    local f="$FLASHMLA_OPS"
    case "$(patch_state "$f")" in
        patched_correct)
            log_warn "flashmla.py 已有正确 sparse 兼容补丁，跳过"
            flagtune_emit_result "ALREADY_PATCHED"
            return 0
            ;;
        patched_correct_backup_missing)
            log_warn "flashmla.py 已有正确 sparse 兼容补丁，但备份缺失"
            flagtune_emit_result "ALREADY_PATCHED_BACKUP_MISSING"
            return 0
            ;;
        patched_invalid)
            log_error "flashmla.py 存在不完整或非预期 sparse 补丁"
            flagtune_emit_result "PATCH_INVALID"
            return 1
            ;;
        target_mismatch)
            log_error "flashmla.py 和 flashmla-sparse 补丁脚本预期不匹配"
            flagtune_emit_result "TARGET_MISMATCH"
            return 1
            ;;
    esac
    backup "$f"

    "$PYTHON_EXECUTABLE" - "$f" "$PATCH_MARKER" "$PATCH_END" "$COMPAT_MARKER" << 'PYEOF'
import sys

filepath = sys.argv[1]
marker = sys.argv[2]
end_marker = sys.argv[3]
compat_marker = sys.argv[4]

with open(filepath, "r") as f:
    content = f.read()

# 目标: 仅将 flash_mla_sparse_fwd 的导入重定向到 FlagGems
# 保留 flash_mla_with_kvcache 等其他导入不变

# 精确匹配导入块
old = """if _is_flashmla_available()[0]:
    from vllm.third_party.flashmla.flash_mla_interface import (  # noqa: F401
        FlashMLASchedMeta,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )"""

def make_wrapper(indent=""):
    return f"""{indent}from flag_gems.fused import (
{indent}    flash_mla_sparse_fwd as {compat_marker},
{indent})

{indent}def flash_mla_sparse_fwd(
{indent}    q,
{indent}    kv,
{indent}    indices,
{indent}    sm_scale,
{indent}    d_v=512,
{indent}    attn_sink=None,
{indent}    topk_length=None,
{indent}    out=None,
{indent}):
{indent}    output, max_logits, lse = {compat_marker}(
{indent}        q,
{indent}        kv,
{indent}        indices,
{indent}        sm_scale,
{indent}        d_v,
{indent}        attn_sink,
{indent}        topk_length,
{indent}    )
{indent}    if out is not None:
{indent}        out.copy_(output)
{indent}        output = out
{indent}    return output, max_logits, lse"""


# 从原始模块导入除 flash_mla_sparse_fwd 外的所有内容，并通过包装层
# 兼容 vLLM 0.21.0 新增的 out 参数。
new = f"""{marker}
if _is_flashmla_available()[0]:
    from vllm.third_party.flashmla.flash_mla_interface import (  # noqa: F401
        FlashMLASchedMeta,
        flash_attn_varlen_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )
{make_wrapper("    ")}
{end_marker}"""

if marker in content:
    start = content.index(marker)
    end = content.find(end_marker, start)
    if end == -1:
        print("ERROR: 找到补丁起始标记，但缺少结束标记", file=sys.stderr)
        sys.exit(1)
    end += len(end_marker)
    new_content = content[:start] + new + content[end:]
    action = "已将旧版补丁升级为兼容包装层"
elif old in content:
    new_content = content.replace(old, new, 1)
    action = "已替换为 FlagGems 兼容包装层"
else:
    print("WARN: flashmla.py 导入块格式不匹配，尝试宽松匹配...", file=sys.stderr)
    print("ERROR: 无法安全定位 flash_mla_sparse_fwd 导入块，拒绝追加覆盖", file=sys.stderr)
    sys.exit(1)

with open(filepath, "w") as f:
    f.write(new_content)

print(f"OK: flash_mla_sparse_fwd {action}")
PYEOF
    flagtune_emit_result "APPLIED"
}

# 主逻辑
if [[ "$ACTION" == "status" ]]; then
    check_status
    case "$(patch_state "$FLASHMLA_OPS")" in
        patched_correct) flagtune_emit_result "ALREADY_PATCHED" ;;
        patched_correct_backup_missing) flagtune_emit_result "ALREADY_PATCHED_BACKUP_MISSING" ;;
        patched_invalid) flagtune_emit_result "PATCH_INVALID"; exit 1 ;;
        target_mismatch) flagtune_emit_result "TARGET_MISMATCH"; exit 1 ;;
        *) flagtune_emit_result "ALREADY_RESTORED" ;;
    esac
    exit 0
fi

if [[ "$ACTION" == "restore" ]]; then
    state="$(patch_state "$FLASHMLA_OPS")"
    case "$state" in
        clean)
            log_info "未检测到 flashmla-sparse 补丁，无需还原"
            flagtune_emit_result "ALREADY_RESTORED"
            ;;
        patched_correct)
            restore "$FLASHMLA_OPS"
            flagtune_emit_result "RESTORED"
            ;;
        patched_correct_backup_missing)
            log_error "flashmla-sparse 补丁已存在，但备份丢失: ${FLASHMLA_OPS}${BAK_SUFFIX}"
            flagtune_emit_result "BACKUP_MISSING"
            exit 1
            ;;
        patched_invalid)
            log_error "flashmla-sparse 补丁不完整或不正确，拒绝还原"
            flagtune_emit_result "PATCH_INVALID"
            exit 1
            ;;
        target_mismatch)
            log_error "flashmla.py 和 flashmla-sparse 补丁脚本预期不匹配"
            flagtune_emit_result "TARGET_MISMATCH"
            exit 1
            ;;
    esac
    exit 0
fi

log_info "vLLM 路径: $VLLM_DIR"
log_info "补丁目标:"
log_info "  flashmla.py - flash_mla_sparse_fwd → FlagGems Triton 兼容包装层"

patch_flashmla_ops

log_info "补丁完成！还原命令: $0 --restore"
