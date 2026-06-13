#!/usr/bin/env bash
#
# patch-vllm-all.sh - 顶层脚本，一键管理所有 FlagGems → vLLM 补丁
#
# 用法:
#   ./patch-vllm-all.sh --apply        # 应用所有补丁
#   ./patch-vllm-all.sh --restore      # 还原所有补丁
#   ./patch-vllm-all.sh --status       # 查看所有补丁状态
#   ./patch-vllm-all.sh --apply  --only fp8,mm       # 仅应用指定补丁
#   ./patch-vllm-all.sh --restore --only fp8,mm      # 仅还原指定补丁
#

set -uo pipefail
# 注意: 不使用 set -e，因为子脚本失败时我们需要继续处理后续补丁

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# ============================================================
# 补丁列表 (有序)
# 格式: "短名称:脚本文件名:描述"
# ============================================================
PATCHES=(
    "fp8:patch-vllm-fp8.sh:FP8 quantization (cutlass_scaled_mm)"
    "router-gemm:patch-vllm-router-gemm.sh:Router GEMM (gate linear)"
    "flashmla-sparse:patch-vllm-flashmla-sparse.sh:FlashMLA sparse (flash_mla_sparse_fwd)"
    "flashmla:patch-vllm-flashmla.sh:FlashMLA (flash_mla_with_kvcache)"
    "fp8-einsum:patch-vllm-fp8-einsum.sh:FP8 Einsum (fp8_einsum_2x)"
    "indexer-k-quant:patch-vllm-indexer-k-quant.sh:Indexer K quant (indexer_k_quant_and_cache)"
    "cp-gather-indexer:patch-vllm-cp-gather-indexer.sh:CP gather indexer (cp_gather_indexer_k_quant_cache)"
    "per-token-group-fp8:patch-vllm-per-token-group-fp8.sh:Per-token group FP8 quantization"
    "mm:patch-vllm-mm.sh:MM (aten::mm dispatch override)"
)

# ============================================================
# 参数解析
# ============================================================
ACTION=""
ONLY_LIST=""

show_help() {
    echo "用法: $0 --apply|--restore|--status [--only name1,name2,...]"
    echo ""
    echo "动作:"
    echo "  --apply    应用所有补丁"
    echo "  --restore  还原所有补丁"
    echo "  --status   查看所有补丁状态"
    echo ""
    echo "选项:"
    echo "  --only name1,name2,...  仅操作指定补丁 (逗号分隔)"
    echo ""
    echo "可用补丁名称:"
    for entry in "${PATCHES[@]}"; do
        local_name="${entry%%:*}"
        local_rest="${entry#*:}"
        local_desc="${local_rest#*:}"
        printf "  %-22s %s\n" "$local_name" "$local_desc"
    done
}

set_action() {
    local next_action="$1"
    if [[ -n "$ACTION" ]]; then
        echo -e "${RED}[ERROR]${NC} 只能指定一个动作: --apply, --restore, --status"
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
        --status)    set_action "status"; shift ;;
        --apply)     set_action "apply"; shift ;;
        --restore)   set_action "restore"; shift ;;
        --only)      ONLY_LIST="$2"; shift 2 ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}[ERROR]${NC} 未知参数: $1"
            echo "使用 -h 查看帮助"
            exit 1
            ;;
    esac
done

if [[ -z "$ACTION" ]]; then
    echo -e "${RED}[ERROR]${NC} 必须指定一个动作: --apply, --restore, --status"
    show_help
    exit 1
fi

# ============================================================
# 辅助函数
# ============================================================

# 检查某个补丁名是否在 --only 列表中
should_process() {
    local name="$1"
    if [[ -z "$ONLY_LIST" ]]; then
        return 0  # 无过滤，全部处理
    fi
    # 按逗号分割检查
    IFS=',' read -ra parts <<< "$ONLY_LIST"
    for part in "${parts[@]}"; do
        # 去除空格
        part="${part// /}"
        if [[ "$part" == "$name" ]]; then
            return 0
        fi
    done
    return 1
}

extract_result() {
    local output="$1"
    echo "$output" | awk -F= '/^FLAGTUNE_RESULT=/{result=$2} END{print result}'
}

print_child_error() {
    local output="$1"
    echo "$output" | grep -v '^FLAGTUNE_RESULT=' | grep -E '\[ERROR\]|ERROR:|WARN:|备份丢失|拒绝还原|目标文件|格式不匹配|无法|找不到' | head -3 | sed 's/^/    /'
}

# ============================================================
# 主逻辑
# ============================================================

echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║       FlagGems → vLLM 补丁管理器                           ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

case "$ACTION" in
    status)
        echo -e "${BLUE}[状态]${NC} 检查所有补丁..."
        echo ""
        for entry in "${PATCHES[@]}"; do
            name="${entry%%:*}"
            rest="${entry#*:}"
            script="${rest%%:*}"
            desc="${rest#*:}"

            if ! should_process "$name"; then
                continue
            fi

            script_path="${SCRIPT_DIR}/${script}"
            if [[ ! -f "$script_path" ]]; then
                echo -e "  ${YELLOW}⚠${NC}  ${BOLD}${name}${NC} - 脚本不存在: ${script}"
                continue
            fi

            output=$(bash "$script_path" --status 2>&1)
            rc=$?
            result=$(extract_result "$output")
            case "$result" in
                ALREADY_PATCHED)
                    echo -e "  ${GREEN}✓${NC}  ${BOLD}${name}${NC} - ${desc}"
                    ;;
                ALREADY_PATCHED_BACKUP_MISSING)
                    echo -e "  ${YELLOW}⚠${NC}  ${BOLD}${name}${NC} - ${desc} (已补丁，备份缺失)"
                    ;;
                ALREADY_RESTORED)
                    echo -e "  ${YELLOW}○${NC}  ${BOLD}${name}${NC} - ${desc}"
                    ;;
                PATCH_INVALID|BACKUP_MISSING|TARGET_MISMATCH)
                    echo -e "  ${RED}✗${NC}  ${BOLD}${name}${NC} - ${desc} (${result})"
                    print_child_error "$output"
                    ;;
                "")
                    echo -e "  ${RED}✗${NC}  ${BOLD}${name}${NC} - ${desc} (无结果协议, rc=${rc})"
                    print_child_error "$output"
                    ;;
                *)
                    echo -e "  ${RED}✗${NC}  ${BOLD}${name}${NC} - ${desc} (${result})"
                    ;;
            esac
        done
        echo ""
        echo -e "图例: ${GREEN}✓${NC} 已补丁  ${YELLOW}○${NC} 未补丁"
        ;;

    apply)
        echo -e "${BLUE}[应用]${NC} 开始应用补丁..."
        echo ""
        success=0
        skipped=0
        warned=0
        failed=0

        for entry in "${PATCHES[@]}"; do
            name="${entry%%:*}"
            rest="${entry#*:}"
            script="${rest%%:*}"
            desc="${rest#*:}"

            if ! should_process "$name"; then
                continue
            fi

            script_path="${SCRIPT_DIR}/${script}"
            if [[ ! -f "$script_path" ]]; then
                echo -e "  ${YELLOW}⚠${NC}  ${name} - 脚本不存在，跳过"
                ((skipped++)) || true
                continue
            fi

            # 构建参数
            args=("--apply")

            echo -ne "  ${BOLD}${name}${NC} ... "

            output=$(bash "$script_path" "${args[@]}" 2>&1)
            rc=$?
            result=$(extract_result "$output")

            if [[ $rc -eq 0 && -n "$result" ]]; then
                case "$result" in
                    APPLIED)
                        echo -e "${GREEN}补丁成功${NC}"
                        ((success++)) || true
                        ;;
                    ALREADY_PATCHED)
                    echo -e "${YELLOW}已存在，跳过${NC}"
                    ((skipped++)) || true
                        ;;
                    ALREADY_PATCHED_BACKUP_MISSING)
                        echo -e "${YELLOW}已存在，备份缺失${NC}"
                        ((skipped++)) || true
                        ((warned++)) || true
                        ;;
                    PATCH_INVALID|TARGET_MISMATCH|BACKUP_MISSING)
                        echo -e "${RED}异常 (${result})${NC}"
                        print_child_error "$output"
                        ((failed++)) || true
                        ;;
                    *)
                        echo -e "${RED}未知结果 (${result})${NC}"
                        print_child_error "$output"
                        ((failed++)) || true
                        ;;
                esac
            else
                echo -e "${RED}失败${NC}"
                if [[ -z "$result" ]]; then
                    echo "    子脚本未输出 FLAGTUNE_RESULT"
                fi
                print_child_error "$output"
                ((failed++)) || true
            fi
        done

        echo ""
        echo -e "结果: ${GREEN}${success} 成功${NC}, ${YELLOW}${skipped} 跳过${NC}, ${YELLOW}${warned} 警告${NC}, ${RED}${failed} 失败${NC}"
        ;;

    restore)
        echo -e "${BLUE}[还原]${NC} 开始还原补丁..."
        echo ""
        success=0
        noop=0
        failed=0

        # 还原时逆序执行 (先应用的后还原)
        for ((i=${#PATCHES[@]}-1; i>=0; i--)); do
            entry="${PATCHES[$i]}"
            name="${entry%%:*}"
            rest="${entry#*:}"
            script="${rest%%:*}"
            desc="${rest#*:}"

            if ! should_process "$name"; then
                continue
            fi

            script_path="${SCRIPT_DIR}/${script}"
            if [[ ! -f "$script_path" ]]; then
                continue
            fi

            echo -ne "  ${BOLD}${name}${NC} ... "

            output=$(bash "$script_path" --restore 2>&1)
            rc=$?
            result=$(extract_result "$output")

            if [[ $rc -eq 0 && -n "$result" ]]; then
                case "$result" in
                    RESTORED)
                    echo -e "${GREEN}已还原${NC}"
                    ((success++)) || true
                        ;;
                    ALREADY_RESTORED)
                        echo -e "${YELLOW}无需还原${NC}"
                        ((noop++)) || true
                        ;;
                    BACKUP_MISSING|PATCH_INVALID|TARGET_MISMATCH)
                        echo -e "${RED}异常 (${result})${NC}"
                        print_child_error "$output"
                        ((failed++)) || true
                        ;;
                    *)
                        echo -e "${RED}未知结果 (${result})${NC}"
                        print_child_error "$output"
                        ((failed++)) || true
                        ;;
                esac
            else
                echo -e "${RED}失败${NC}"
                if [[ -z "$result" ]]; then
                    echo "    子脚本未输出 FLAGTUNE_RESULT"
                fi
                print_child_error "$output"
                ((failed++)) || true
            fi
        done

        echo ""
        echo -e "结果: ${GREEN}${success} 还原${NC}, ${YELLOW}${noop} 无需还原${NC}, ${RED}${failed} 失败${NC}"
        ;;
esac
