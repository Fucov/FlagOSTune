#!/usr/bin/env bash
#
# patch-vllm-flashmla.sh - 将 vLLM 的 flash_mla_with_kvcache 替换为 FlagGems Triton 实现
#
# 补丁点:
#   vllm/third_party/flashmla/flash_mla_interface.py
#     FlashMLASchedMeta/get_mla_metadata/flash_mla_with_kvcache -> flag_gems.fused.flash_mla_with_kvcache
#
# 行为:
#   将 FlashMLA metadata 和 kvcache forward 入口替换为 FlagGems 实现。
#   保留文件中的其他函数（flash_mla_sparse_fwd、flash_attn_varlen_* 等）。
#
# 用法:
#   ./patch-vllm-flashmla.sh --apply
#   ./patch-vllm-flashmla.sh --restore
#   ./patch-vllm-flashmla.sh --status
#

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/patch-vllm-common.sh"
PYTHON_EXECUTABLE="${Python_EXECUTABLE:-python3}"

PATCH_MARKER="# >>> FLAGGEMS FLASHMLA PATCH >>>"
PATCH_END="# <<< FLAGGEMS FLASHMLA PATCH <<<"

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

# 动态检测当前环境中 vllm 的安装路径
detect_target() {
    "$PYTHON_EXECUTABLE" -c "
import vllm, os
print(os.path.join(os.path.dirname(vllm.__file__), 'third_party', 'flashmla', 'flash_mla_interface.py'))
" 2>/dev/null || { log_error "无法定位 vllm（当前环境未安装 vllm）"; flagtune_emit_result "TARGET_MISMATCH"; exit 1; }
}

if ! TARGET="$(detect_target)"; then
    log_error "无法定位 vllm（当前环境未安装 vllm）"
    flagtune_emit_result "TARGET_MISMATCH"
    exit 1
fi
BACKUP="${TARGET}.flashmlabak"

if [[ ! -f "$TARGET" ]]; then
    log_error "目标文件不存在: $TARGET"
    flagtune_emit_result "TARGET_MISMATCH"
    exit 1
fi

target_matches() {
    [[ -f "$TARGET" ]] && {
        grep -qE "^def get_mla_metadata\(" "$TARGET" 2>/dev/null || \
        grep -qF "from flag_gems.fused.flash_mla_with_kvcache import get_mla_metadata" "$TARGET" 2>/dev/null
    } && {
        grep -qE "^def flash_mla_with_kvcache\(" "$TARGET" 2>/dev/null || \
        grep -qF "from flag_gems.fused.flash_mla_with_kvcache import flash_mla_with_kvcache" "$TARGET" 2>/dev/null
    }
}

is_patched() { flagtune_has_marker_pair "$TARGET" "$PATCH_MARKER" "$PATCH_END"; }

patch_correct() {
    is_patched && \
        flagtune_has_all "$TARGET" \
            "from flag_gems.fused.flash_mla_with_kvcache import FlashMLASchedMeta" \
            "from flag_gems.fused.flash_mla_with_kvcache import get_mla_metadata" \
            "from flag_gems.fused.flash_mla_with_kvcache import flash_mla_with_kvcache" && \
        ! grep -qF "/workspace/FlagGems-dev/src" "$TARGET" 2>/dev/null
}

patch_state() {
    if ! target_matches; then
        echo "target_mismatch"
    elif patch_correct; then
        if [[ -f "$BACKUP" ]]; then
            echo "patched_correct"
        else
            echo "patched_correct_backup_missing"
        fi
    elif flagtune_has_any_marker "$TARGET" "$PATCH_MARKER" "$PATCH_END" || \
         grep -qF "flag_gems.fused.flash_mla_with_kvcache" "$TARGET" 2>/dev/null; then
        echo "patched_invalid"
    else
        echo "clean"
    fi
}

check_status() {
    if is_patched; then
        log_info "状态: 已打补丁"
        log_info "  目标: $TARGET"
        if [[ -f "$BACKUP" ]]; then
            log_info "  备份: $BACKUP"
        fi
    else
        log_info "状态: 未打补丁（原始状态）"
    fi
}

do_restore() {
    if [[ -f "$BACKUP" ]]; then
        cp "$BACKUP" "$TARGET"
        rm -f "$BACKUP"
        log_info "已还原: $TARGET"
    else
        log_warn "无备份: $TARGET"
    fi
}

do_patch() {
    case "$(patch_state)" in
        patched_correct)
            log_warn "已经有正确 flashmla 补丁，跳过"
            flagtune_emit_result "ALREADY_PATCHED"
            return 0
            ;;
        patched_correct_backup_missing)
            log_warn "已经有正确 flashmla 补丁，但备份缺失"
            flagtune_emit_result "ALREADY_PATCHED_BACKUP_MISSING"
            return 0
            ;;
        patched_invalid)
            log_error "flashmla 补丁不完整或不正确"
            flagtune_emit_result "PATCH_INVALID"
            return 1
            ;;
        target_mismatch)
            log_error "flash_mla_interface.py 和 flashmla 补丁脚本预期不匹配"
            flagtune_emit_result "TARGET_MISMATCH"
            return 1
            ;;
    esac

    # 备份
    if [[ ! -f "$BACKUP" ]]; then
        cp "$TARGET" "$BACKUP"
        log_info "备份: $BACKUP"
    fi

    # 生成补丁后的文件
    "$PYTHON_EXECUTABLE" << PYEOF
import sys
import re

target = "$TARGET"
marker = "$PATCH_MARKER"
end_marker = "$PATCH_END"

with open(target, "r") as f:
    content = f.read()

# Find the end of get_mla_metadata function (before flash_mla_with_kvcache)
get_mla_start = content.find("\ndef get_mla_metadata(")
if get_mla_start == -1:
    print("ERROR: 找不到 get_mla_metadata 函数定义", file=sys.stderr)
    sys.exit(1)

# Find the start of flash_mla_with_kvcache function
flash_mla_start = content.find("\ndef flash_mla_with_kvcache(")
if flash_mla_start == -1:
    print("ERROR: 找不到 flash_mla_with_kvcache 函数定义", file=sys.stderr)
    sys.exit(1)

# Find the start of the next top-level function after flash_mla_with_kvcache
# Look for the next "def " at the beginning of a line
next_func_match = re.search(r'\ndef [a-zA-Z_]', content[flash_mla_start + 1:])
if next_func_match:
    next_func_start = flash_mla_start + 1 + next_func_match.start()
else:
    # No more functions, take everything till the end
    next_func_start = len(content)

# Keep: imports + FlashMLASchedMeta + everything before get_mla_metadata
header = content[:get_mla_start]

# Keep: everything after flash_mla_with_kvcache (flash_mla_sparse_fwd, flash_attn_varlen_*, etc.)
tail = content[next_func_start:]

# Build the patched file
patched = header + f"""
{marker}
# get_mla_metadata, FlashMLASchedMeta, flash_mla_with_kvcache 已替换为 FlagGems Triton 实现
from flag_gems.fused.flash_mla_with_kvcache import FlashMLASchedMeta  # noqa: F811
from flag_gems.fused.flash_mla_with_kvcache import get_mla_metadata  # noqa: F811
from flag_gems.fused.flash_mla_with_kvcache import flash_mla_with_kvcache  # noqa: F811
{end_marker}
""" + tail

with open(target, "w") as f:
    f.write(patched)

print("OK: flash_mla_with_kvcache 已替换为 FlagGems Triton 实现")
PYEOF

    if [[ $? -ne 0 ]]; then
        log_error "补丁失败"
        do_restore
        exit 1
    fi

    log_info "补丁完成！"
    log_info "  flash_mla_with_kvcache → flag_gems.fused.flash_mla_with_kvcache"
    log_info "  还原命令: $0 --restore"
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
            log_info "未检测到 flashmla 补丁，无需还原"
            flagtune_emit_result "ALREADY_RESTORED"
            ;;
        patched_correct)
            do_restore
            flagtune_emit_result "RESTORED"
            ;;
        patched_correct_backup_missing)
            log_error "flashmla 补丁已存在，但备份丢失: $BACKUP"
            flagtune_emit_result "BACKUP_MISSING"
            exit 1
            ;;
        patched_invalid)
            log_error "flashmla 补丁不完整或不正确，拒绝还原"
            flagtune_emit_result "PATCH_INVALID"
            exit 1
            ;;
        target_mismatch)
            log_error "flash_mla_interface.py 和 flashmla 补丁脚本预期不匹配"
            flagtune_emit_result "TARGET_MISMATCH"
            exit 1
            ;;
    esac
    exit 0
fi

do_patch
