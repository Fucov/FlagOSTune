#!/usr/bin/env bash
#
# patch-vllm-mm.sh - 将 vLLM 中所有 aten::mm 调用替换为 FlagGems Triton 实现
#
# 补丁点:
#   vllm/v1/worker/gpu_model_runner.py
#     module import region -> flag_gems.only_enable(include=["mm", "mm_out"])
#   vllm/worker/gpu_model_runner.py
#     fallback path when vLLM v1 path is absent
#
# 行为:
#   通过 flag_gems.only_enable 注册 PyTorch dispatch，使进程内 aten::mm 走 FlagGems。
#   - 此补丁是进程级全局生效的，覆盖所有 torch.mm / F.linear → aten::mm 路径
#   - 仅在 VLLM_BATCH_INVARIANT 未启用时有效 (batch_invariant 模式绕过 aten::mm)
#
# 用法:
#   ./patch-vllm-mm.sh --apply
#   ./patch-vllm-mm.sh --restore
#   ./patch-vllm-mm.sh --status
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

PATCH_MARKER="# >>> FLAGGEMS MM PATCH >>>"
PATCH_END="# <<< FLAGGEMS MM PATCH <<<"
BAK_SUFFIX=".mm_bak"

ACTION=""

show_help() {
    echo "用法: $0 --apply|--restore|--status"
    echo ""
    echo "  --apply    应用补丁"
    echo "  --restore  还原原始文件"
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
        -h|--help)
            show_help
            exit 0
            ;;
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

# 补丁目标: gpu_model_runner.py (vLLM v1 主入口)
GPU_RUNNER="${VLLM_DIR}/v1/worker/gpu_model_runner.py"
# 备选: 如果 v1 不存在，尝试旧路径
if [[ ! -f "$GPU_RUNNER" ]]; then
    GPU_RUNNER="${VLLM_DIR}/worker/gpu_model_runner.py"
fi

if [[ ! -f "$GPU_RUNNER" ]]; then
    log_error "找不到 gpu_model_runner.py"
    flagtune_emit_result "TARGET_MISMATCH"
    exit 1
fi

TARGETS=("$GPU_RUNNER")

target_matches() {
    [[ -f "$GPU_RUNNER" ]] && grep -qE '^(import |from |class |def )' "$GPU_RUNNER" 2>/dev/null
}

is_patched() { flagtune_has_marker_pair "$1" "$PATCH_MARKER" "$PATCH_END"; }

patch_correct() {
    local f="$1"
    is_patched "$f" && \
        flagtune_has_all "$f" \
            "import flag_gems as _fg_mm" \
            "_fg_mm.only_enable(include=[\"mm\", \"mm_out\"])" \
            "del _fg_mm"
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
    elif flagtune_has_any_marker "$f" "$PATCH_MARKER" "$PATCH_END" || grep -qF "_fg_mm.only_enable" "$f" 2>/dev/null; then
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
    echo "=== mm (aten dispatch) 补丁状态 ==="
    for f in "${TARGETS[@]}"; do
        local name
        name=$(basename "$f")
        if is_patched "$f"; then
            echo -e "  $name: ${GREEN}已补丁${NC}"
            echo -e "    模式: 无条件启用"
        else
            echo -e "  $name: ${YELLOW}未补丁${NC}"
        fi
        if [[ -f "${f}${BAK_SUFFIX}" ]]; then
            echo -e "    备份: ${GREEN}存在${NC}"
        fi
    done
}

patch_mm() {
    local f="$GPU_RUNNER"
    case "$(patch_state "$f")" in
        patched_correct)
            log_warn "gpu_model_runner.py 已有正确 mm 补丁，跳过"
            flagtune_emit_result "ALREADY_PATCHED"
            return 0
            ;;
        patched_correct_backup_missing)
            log_warn "gpu_model_runner.py 已有正确 mm 补丁，但备份缺失"
            flagtune_emit_result "ALREADY_PATCHED_BACKUP_MISSING"
            return 0
            ;;
        patched_invalid)
            log_error "gpu_model_runner.py 存在不完整或非预期 mm 补丁"
            flagtune_emit_result "PATCH_INVALID"
            return 1
            ;;
        target_mismatch)
            log_error "gpu_model_runner.py 和 mm 补丁脚本预期不匹配"
            flagtune_emit_result "TARGET_MISMATCH"
            return 1
            ;;
    esac
    backup "$f"

    "$PYTHON_EXECUTABLE" - "$f" "$PATCH_MARKER" "$PATCH_END" << 'PYEOF'
import sys

filepath = sys.argv[1]
marker = sys.argv[2]
end_marker = sys.argv[3]

with open(filepath, "r") as f:
    content = f.read()

# 构造补丁代码
patch_code = f"""{marker}
# FlagGems mm: override aten::mm with Triton kernel (unconditional)
import flag_gems as _fg_mm
_fg_mm.only_enable(include=["mm", "mm_out"])
del _fg_mm
{end_marker}
"""

# 寻找插入点: 在文件头部所有 import 块之后插入
# 策略: 正确处理多行括号 import (from x import (\n  ...\n))
lines = content.split('\n')
insert_idx = 0
paren_depth = 0  # 追踪括号嵌套层级

for i, line in enumerate(lines):
    stripped = line.strip()

    # 在括号内部 (多行 import 的续行)，跳过直到括号闭合
    if paren_depth > 0:
        paren_depth += stripped.count('(') - stripped.count(')')
        if paren_depth <= 0:
            paren_depth = 0
            insert_idx = i + 1
        continue

    if not stripped:
        continue
    if stripped.startswith('#'):
        continue
    if stripped.startswith('import ') or stripped.startswith('from '):
        # 检查是否是多行 import (有未闭合的左括号)
        paren_depth += stripped.count('(') - stripped.count(')')
        insert_idx = i + 1
        continue
    # 跳过 if TYPE_CHECKING: 块 (常见于 vllm 头部)
    if stripped == 'if TYPE_CHECKING:':
        # 跳过整个 TYPE_CHECKING 块
        indent_next = True
        for j in range(i + 1, len(lines)):
            s = lines[j].strip()
            if not s or s.startswith('#'):
                continue
            if lines[j].startswith('    ') or lines[j].startswith('\t'):
                insert_idx = j + 1
            else:
                break
        continue
    # 遇到第一个非 import/comment/空行，就在这之前插入
    insert_idx = i
    break

# 在 insert_idx 位置插入 (加一个空行分隔)
lines.insert(insert_idx, '\n' + patch_code)
new_content = '\n'.join(lines)

with open(filepath, "w") as f:
    f.write(new_content)

print("OK: aten::mm 已替换为 FlagGems Triton 实现 (模式: 无条件)")
PYEOF
    flagtune_emit_result "APPLIED"
}

# 主逻辑
if [[ "$ACTION" == "status" ]]; then
    check_status
    case "$(patch_state "$GPU_RUNNER")" in
        patched_correct) flagtune_emit_result "ALREADY_PATCHED" ;;
        patched_correct_backup_missing) flagtune_emit_result "ALREADY_PATCHED_BACKUP_MISSING" ;;
        patched_invalid) flagtune_emit_result "PATCH_INVALID"; exit 1 ;;
        target_mismatch) flagtune_emit_result "TARGET_MISMATCH"; exit 1 ;;
        *) flagtune_emit_result "ALREADY_RESTORED" ;;
    esac
    exit 0
fi

if [[ "$ACTION" == "restore" ]]; then
    state="$(patch_state "$GPU_RUNNER")"
    case "$state" in
        clean)
            log_info "未检测到 mm 补丁，无需还原"
            flagtune_emit_result "ALREADY_RESTORED"
            ;;
        patched_correct)
            restore "$GPU_RUNNER"
            flagtune_emit_result "RESTORED"
            ;;
        patched_correct_backup_missing)
            log_error "mm 补丁已存在，但备份丢失: ${GPU_RUNNER}${BAK_SUFFIX}"
            flagtune_emit_result "BACKUP_MISSING"
            exit 1
            ;;
        patched_invalid)
            log_error "mm 补丁不完整或不正确，拒绝还原"
            flagtune_emit_result "PATCH_INVALID"
            exit 1
            ;;
        target_mismatch)
            log_error "gpu_model_runner.py 和 mm 补丁脚本预期不匹配"
            flagtune_emit_result "TARGET_MISMATCH"
            exit 1
            ;;
    esac
    exit 0
fi

log_info "vLLM 路径: $VLLM_DIR"
log_info "补丁目标:"
log_info "  gpu_model_runner.py - aten::mm → FlagGems Triton"
log_info "  模式: 无条件启用"

patch_mm

log_info "补丁完成！"
log_info "还原命令: $0 --restore"
