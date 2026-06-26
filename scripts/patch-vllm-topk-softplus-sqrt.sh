#!/usr/bin/env bash
#
# patch-vllm-topk-softplus-sqrt.sh - Replace vLLM topk_hash_softplus_sqrt
# with the FlagGems Triton implementation.
#
# Patch point:
#   vllm/_custom_ops.py
#     topk_hash_softplus_sqrt -> flag_gems.fused.topk_softplus_sqrt
#
# Usage:
#   ./patch-vllm-topk-softplus-sqrt.sh --apply
#   ./patch-vllm-topk-softplus-sqrt.sh --restore
#   ./patch-vllm-topk-softplus-sqrt.sh --status
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

PATCH_MARKER="# >>> FLAGTUNE TOPK_SOFTPLUS_SQRT PATCH START >>>"
PATCH_END="# <<< FLAGTUNE TOPK_SOFTPLUS_SQRT PATCH END <<<"
BAK_SUFFIX=".topk_softplus_sqrt_bak"

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
    case "$1" in
        --apply) set_action "apply"; shift ;;
        --restore) set_action "restore"; shift ;;
        --status) set_action "status"; shift ;;
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

TARGET="${VLLM_DIR}/_custom_ops.py"
TARGETS=("$TARGET")

target_matches() {
    [[ -f "$TARGET" ]] && \
        grep -qE "^def topk_hash_softplus_sqrt\(" "$TARGET" 2>/dev/null
}

is_patched() {
    flagtune_has_marker_pair "$1" "$PATCH_MARKER" "$PATCH_END"
}

patch_correct() {
    local f="$1"
    is_patched "$f" && \
        flagtune_has_all "$f" \
            "def topk_hash_softplus_sqrt(" \
            "from flag_gems.fused import topk_softplus_sqrt as _flagtune_topk_softplus_sqrt" \
            "_flagtune_topk_softplus_sqrt("
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
         grep -qF "_flagtune_topk_softplus_sqrt" "$f" 2>/dev/null; then
        echo "patched_invalid"
    else
        echo "clean"
    fi
}

check_status() {
    echo "=== topk_hash_softplus_sqrt 补丁状态 ==="
    for f in "${TARGETS[@]}"; do
        local name
        name="${f#$VLLM_DIR/}"
        case "$(patch_state "$f")" in
            patched_correct)
                echo -e "  $name: ${GREEN}已补丁${NC}"
                ;;
            patched_correct_backup_missing)
                echo -e "  $name: ${YELLOW}已补丁，备份缺失${NC}"
                ;;
            clean)
                echo -e "  $name: ${YELLOW}未补丁${NC}"
                ;;
            *)
                echo -e "  $name: ${RED}异常${NC}"
                ;;
        esac
        if [[ -f "${f}${BAK_SUFFIX}" ]]; then
            echo -e "    备份: ${GREEN}存在${NC}"
        fi
    done
}

patch_topk_softplus_sqrt() {
    local f="$TARGET"
    case "$(patch_state "$f")" in
        patched_correct)
            log_warn "_custom_ops.py 已有正确 topk_softplus_sqrt 补丁，跳过"
            flagtune_emit_result "ALREADY_PATCHED"
            return 0
            ;;
        patched_correct_backup_missing)
            log_warn "_custom_ops.py 已有正确 topk_softplus_sqrt 补丁，但备份缺失"
            flagtune_emit_result "ALREADY_PATCHED_BACKUP_MISSING"
            return 0
            ;;
        patched_invalid)
            log_error "_custom_ops.py 存在不完整或非预期 topk_softplus_sqrt 补丁"
            flagtune_emit_result "PATCH_INVALID"
            return 1
            ;;
        target_mismatch)
            log_error "_custom_ops.py 和 topk_softplus_sqrt 补丁脚本预期不匹配"
            flagtune_emit_result "TARGET_MISMATCH"
            return 1
            ;;
    esac

    flagtune_backup_file "$f" "$BAK_SUFFIX"

    "$PYTHON_EXECUTABLE" - "$f" "$PATCH_MARKER" "$PATCH_END" << 'PYEOF'
import ast
import os
import sys
import tempfile

filepath, marker, end_marker = sys.argv[1:]

with open(filepath, "r") as f:
    content = f.read()

tree = ast.parse(content, filename=filepath)
nodes = [
    node for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name == "topk_hash_softplus_sqrt"
]
if len(nodes) != 1:
    print(f"ERROR: 找到 {len(nodes)} 个 topk_hash_softplus_sqrt，期望 1 个", file=sys.stderr)
    sys.exit(1)

node = nodes[0]
lines = content.splitlines(keepends=True)
start = node.lineno - 1
end = node.end_lineno

new_func = f'''{marker}
def topk_hash_softplus_sqrt(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
) -> None:
    from flag_gems.fused import topk_softplus_sqrt as _flagtune_topk_softplus_sqrt

    _flagtune_topk_softplus_sqrt(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
        routed_scaling_factor,
        correction_bias=e_score_correction_bias,
        input_ids=input_tokens,
        tid2eid=hash_indices_table,
    )
{end_marker}
'''

new_content = "".join(lines[:start]) + new_func + "".join(lines[end:])

try:
    ast.parse(new_content, filename=filepath)
except SyntaxError as exc:
    print(f"ERROR: 补丁后语法校验失败: {exc}", file=sys.stderr)
    sys.exit(1)

tmp_path = None
try:
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(filepath) + ".",
        suffix=".tmp",
        dir=os.path.dirname(filepath),
    )
    with os.fdopen(fd, "w") as f:
        f.write(new_content)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp_path, os.stat(filepath).st_mode)
    os.replace(tmp_path, filepath)
except Exception:
    if tmp_path is not None:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
    raise

print("OK: topk_hash_softplus_sqrt 已替换为 FlagGems Triton 实现")
PYEOF
    flagtune_emit_result "APPLIED"
}

if [[ "$ACTION" == "status" ]]; then
    check_status
    case "$(patch_state "$TARGET")" in
        patched_correct) flagtune_emit_result "ALREADY_PATCHED" ;;
        patched_correct_backup_missing) flagtune_emit_result "ALREADY_PATCHED_BACKUP_MISSING" ;;
        patched_invalid) flagtune_emit_result "PATCH_INVALID"; exit 1 ;;
        target_mismatch) flagtune_emit_result "TARGET_MISMATCH"; exit 1 ;;
        *) flagtune_emit_result "ALREADY_RESTORED" ;;
    esac
    exit 0
fi

if [[ "$ACTION" == "restore" ]]; then
    state="$(patch_state "$TARGET")"
    case "$state" in
        clean)
            log_info "未检测到 topk_softplus_sqrt 补丁，无需还原"
            flagtune_emit_result "ALREADY_RESTORED"
            ;;
        patched_correct)
            if flagtune_restore_function_from_backup \
                "$TARGET" "$BAK_SUFFIX" "$PATCH_MARKER" "$PATCH_END" \
                "topk_hash_softplus_sqrt"; then
                flagtune_emit_result "RESTORED"
            else
                flagtune_emit_result "PATCH_INVALID"
                exit 1
            fi
            ;;
        patched_correct_backup_missing)
            log_error "topk_softplus_sqrt 补丁已存在，但备份丢失: ${TARGET}${BAK_SUFFIX}"
            flagtune_emit_result "BACKUP_MISSING"
            exit 1
            ;;
        patched_invalid)
            log_error "topk_softplus_sqrt 补丁不完整或不正确，拒绝还原"
            flagtune_emit_result "PATCH_INVALID"
            exit 1
            ;;
        target_mismatch)
            log_error "_custom_ops.py 和 topk_softplus_sqrt 补丁脚本预期不匹配"
            flagtune_emit_result "TARGET_MISMATCH"
            exit 1
            ;;
    esac
    exit 0
fi

log_info "vLLM 路径: $VLLM_DIR"
log_info "补丁目标:"
log_info "  _custom_ops.py - topk_hash_softplus_sqrt → FlagGems Triton"

patch_topk_softplus_sqrt

log_info "补丁完成！还原命令: $0 --restore"
