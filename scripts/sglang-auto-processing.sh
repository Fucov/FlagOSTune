#!/usr/bin/env bash
#
# sglang-auto-processing.sh - SGLang profiler processing entrypoint
#
# 用法:
#   ./scripts/sglang-auto-processing.sh --model DeepSeek-V4-Flash --workflow torch
#   ./scripts/sglang-auto-processing.sh --model DeepSeek-V4-Flash --workflow torch --rank all
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

MODEL_CONFIG=""
WORKFLOW="torch"
RANK="0"
WORKERS=""
PROGRESS_EVERY=""
MAX_EVENTS=""
NO_XLSX="false"

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)
                MODEL_CONFIG="$2"
                shift 2
                ;;
            --workflow)
                WORKFLOW="$2"
                shift 2
                ;;
            --rank)
                RANK="$2"
                shift 2
                ;;
            --workers)
                WORKERS="$2"
                shift 2
                ;;
            --progress-every)
                PROGRESS_EVERY="$2"
                shift 2
                ;;
            --max-events)
                MAX_EVENTS="$2"
                shift 2
                ;;
            --no-xlsx)
                NO_XLSX="true"
                shift
                ;;
            -h|--help)
                head -8 "$0" | tail -5
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                exit 1
                ;;
        esac
    done
}

validate_args() {
    if [[ -z "$MODEL_CONFIG" ]]; then
        log_error "必须指定 --model，取值为 config.yaml.<模型名> 的后缀"
        exit 1
    fi
    if [[ "$WORKFLOW" != "torch" ]]; then
        log_error "SGLang 当前仅支持 --workflow torch"
        exit 1
    fi
    if [[ ! "$RANK" =~ ^[0-9]+$ && "$RANK" != "all" ]]; then
        log_error "--rank 仅支持数字或 all，当前值: $RANK"
        exit 1
    fi
}

main() {
    parse_args "$@"
    validate_args
    local args=("--model" "$MODEL_CONFIG" "--workflow" "$WORKFLOW" "--rank" "$RANK")
    if [[ -n "$WORKERS" ]]; then
        args+=("--workers" "$WORKERS")
    fi
    if [[ -n "$PROGRESS_EVERY" ]]; then
        args+=("--progress-every" "$PROGRESS_EVERY")
    fi
    if [[ -n "$MAX_EVENTS" ]]; then
        args+=("--max-events" "$MAX_EVENTS")
    fi
    if [[ "$NO_XLSX" == "true" ]]; then
        args+=("--no-xlsx")
    fi
    log_info "SGLang processing workflow"
    log_step "分析 SGLang Torch profiler 数据"
    "${SCRIPT_DIR}/sglang-run-processing.sh" "${args[@]}"
    log_info "完成"
}

main "$@"
