#!/usr/bin/env bash
#
# patch-vllm-router-gemm.sh - 将 vLLM 的 router_gemm_bf16_fp32 替换为 FlagGems Triton 实现
#
# 补丁点:
#   vllm/_custom_ops.py
#     router_gemm_bf16_fp32 -> flag_gems.router_gemm
#
# 行为:
#   用 FlagGems router_gemm 实现替换 vLLM MoE router GEMM custom op 包装函数。
#
# 用法:
#   ./patch-vllm-router-gemm.sh --apply
#   ./patch-vllm-router-gemm.sh --restore
#   ./patch-vllm-router-gemm.sh --status
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/patch-vllm-common.sh"
PYTHON_EXECUTABLE="${Python_EXECUTABLE:-python3}"

ACTION=""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 补丁标记
PATCH_START_MARKER="# >>> FLAGTUNE ROUTER_GEMM PATCH START >>>"
PATCH_END_MARKER="# <<< FLAGTUNE ROUTER_GEMM PATCH END <<<"

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

# 解析参数
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

# 定位目标文件
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
BAK_SUFFIX=".bak.router_gemm"
OTHER_SHARED_MARKERS=(
    "# >>> FLAGGEMS INDEXER_K_QUANT PATCH >>>"
    "# >>> FLAGGEMS CP_GATHER_INDEXER PATCH >>>"
)

if [[ ! -f "$CUSTOM_OPS" ]]; then
    log_error "找不到目标文件: $CUSTOM_OPS"
    flagtune_emit_result "TARGET_MISMATCH"
    exit 1
fi

# 备份
backup_file() {
    local file="$1"
    local backup="${file}${BAK_SUFFIX}"
    if [[ ! -f "$backup" ]]; then
        cp "$file" "$backup"
        log_info "已备份: $backup"
    else
        log_info "备份已存在: $backup"
    fi
}

# 还原
restore_file() {
    local file="$1"
    local backup="${file}${BAK_SUFFIX}"
    if [[ -f "$backup" ]]; then
        cp "$backup" "$file"
        log_info "已还原: $file"
        rm -f "$backup"
        log_info "已删除备份: $backup"
    else
        log_warn "无备份: $file"
    fi
}

# 检查是否已打补丁
is_patched() {
    flagtune_has_marker_pair "$1" "$PATCH_START_MARKER" "$PATCH_END_MARKER"
}

target_matches() {
    [[ -f "$CUSTOM_OPS" ]] && \
        grep -qE "^def router_gemm_bf16_fp32\(input: torch.Tensor, weight: torch.Tensor\)" "$CUSTOM_OPS" 2>/dev/null
}

patch_correct() {
    local file="$1"
    is_patched "$file" && \
        flagtune_has_all "$file" \
            "def router_gemm_bf16_fp32(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:" \
            "import flag_gems" \
            "return flag_gems.router_gemm(input, weight)"
}

patch_state() {
    local file="$1"
    if ! target_matches; then
        echo "target_mismatch"
    elif patch_correct "$file"; then
        if flagtune_backup_exists "$file" "$BAK_SUFFIX"; then
            echo "patched_correct"
        else
            echo "patched_correct_backup_missing"
        fi
    elif flagtune_has_any_marker "$file" "$PATCH_START_MARKER" "$PATCH_END_MARKER" || \
         grep -qF "flag_gems.router_gemm" "$file" 2>/dev/null; then
        echo "patched_invalid"
    else
        echo "clean"
    fi
}

# 检查补丁状态
check_status() {
    echo "=== Router GEMM 补丁状态 ==="
    local name
    name=$(basename "$CUSTOM_OPS")
    if is_patched "$CUSTOM_OPS"; then
        echo -e "  $name: ${GREEN}已补丁${NC}"
    else
        echo -e "  $name: ${YELLOW}未补丁${NC}"
    fi
    if [[ -f "${CUSTOM_OPS}${BAK_SUFFIX}" ]]; then
        echo -e "    备份: ${GREEN}存在${NC}"
    fi
}

# 应用补丁
apply_patch() {
    local file="$1"

    case "$(patch_state "$file")" in
        patched_correct)
            log_warn "文件已有正确 router_gemm 补丁，跳过"
            flagtune_emit_result "ALREADY_PATCHED"
            return 0
            ;;
        patched_correct_backup_missing)
            log_warn "文件已有正确 router_gemm 补丁，但备份缺失"
            flagtune_emit_result "ALREADY_PATCHED_BACKUP_MISSING"
            return 0
            ;;
        patched_invalid)
            log_error "文件存在不完整或非预期 router_gemm 补丁"
            flagtune_emit_result "PATCH_INVALID"
            return 1
            ;;
        target_mismatch)
            log_error "_custom_ops.py 和 router_gemm 补丁脚本预期不匹配"
            flagtune_emit_result "TARGET_MISMATCH"
            return 1
            ;;
    esac

    backup_file "$file"

    # 找到 router_gemm_bf16_fp32 函数定义的行号
    local func_line
    func_line=$(grep -n "^def router_gemm_bf16_fp32(input: torch.Tensor, weight: torch.Tensor)" "$file" | head -1 | cut -d: -f1)

    if [[ -z "$func_line" ]]; then
        log_error "找不到 router_gemm_bf16_fp32 函数定义"
        exit 1
    fi

    log_info "找到 router_gemm_bf16_fp32 定义在第 ${func_line} 行"

    # 构造补丁内容：在函数体内替换实现
    # 原始代码:
    #   def router_gemm_bf16_fp32(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    #       """bf16 x bf16 -> fp32 GEMM via cuBLAS. weight shape: (N, K)."""
    #       return torch.ops._moe_C.router_gemm_bf16_fp32(input, weight)
    #
    # 替换为调用 FlagGems 的 triton 实现

    local patch_content
    patch_content=$(cat <<'PATCH_EOF'
def router_gemm_bf16_fp32(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """bf16 x bf16 -> fp32 GEMM via FlagGems Triton kernel. weight shape: (N, K)."""
    import flag_gems
    return flag_gems.router_gemm(input, weight)
PATCH_EOF
)

    # 找到函数结束位置（下一个顶层 def/class/if 或空行后的非缩进行）
    local total_lines
    total_lines=$(wc -l < "$file")

    local func_end_line=$((func_line + 1))
    while [[ $func_end_line -le $total_lines ]]; do
        local line_content
        line_content=$(sed -n "${func_end_line}p" "$file")
        # 函数体内的行要么是空行，要么以空格/tab开头
        if [[ $func_end_line -gt $((func_line + 1)) ]] && [[ -n "$line_content" ]] && [[ ! "$line_content" =~ ^[[:space:]] ]]; then
            break
        fi
        func_end_line=$((func_end_line + 1))
    done
    func_end_line=$((func_end_line - 1))

    log_info "函数体范围: 第 ${func_line} - ${func_end_line} 行"

    # 用 sed 替换：删除原函数，插入新实现
    local tmp_file="${file}.tmp"

    {
        # 输出函数定义之前的内容
        head -n $((func_line - 1)) "$file"
        # 插入补丁
        echo "$PATCH_START_MARKER"
        echo "$patch_content"
        echo "$PATCH_END_MARKER"
        # 输出函数之后的内容
        tail -n +$((func_end_line + 1)) "$file"
    } > "$tmp_file"

    mv "$tmp_file" "$file"
    log_info "补丁已应用到: $file"
    flagtune_emit_result "APPLIED"
}

# 移除补丁（还原函数）
remove_patch() {
    local file="$1"
    local backup="${file}${BAK_SUFFIX}"

    if [[ -f "$backup" ]]; then
        # 从备份中提取原始函数，替换补丁区域
        cp "$backup" "$file"
        log_info "已从备份还原"
    else
        log_error "无法移除补丁：找不到备份文件"
        exit 1
    fi
}

# 主函数
main() {
    log_info "目标文件: $CUSTOM_OPS"
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
                log_info "未检测到 router_gemm 补丁，无需还原"
                flagtune_emit_result "ALREADY_RESTORED"
                ;;
            patched_correct)
                if flagtune_restore_function_from_backup \
                    "$CUSTOM_OPS" "$BAK_SUFFIX" "$PATCH_START_MARKER" "$PATCH_END_MARKER" \
                    "router_gemm_bf16_fp32"; then
                    flagtune_emit_result "RESTORED"
                else
                    flagtune_emit_result "PATCH_INVALID"
                    exit 1
                fi
                ;;
            patched_correct_backup_missing)
                log_error "router_gemm 补丁已存在，但备份丢失: ${CUSTOM_OPS}${BAK_SUFFIX}"
                flagtune_emit_result "BACKUP_MISSING"
                exit 1
                ;;
            patched_invalid)
                log_error "router_gemm 补丁不完整或不正确，拒绝还原"
                flagtune_emit_result "PATCH_INVALID"
                exit 1
                ;;
            target_mismatch)
                log_error "_custom_ops.py 和 router_gemm 补丁脚本预期不匹配"
                flagtune_emit_result "TARGET_MISMATCH"
                exit 1
                ;;
        esac
        exit 0
    fi

    apply_patch "$CUSTOM_OPS"

    log_info "完成!"
}

main "$@"
