#!/usr/bin/env bash
#
# sglang-auto-workflow.sh - SGLang profiling one-shot entrypoint
#
# 用法:
#   ./scripts/sglang-auto-workflow.sh --model DeepSeek-V4-Flash --torch
#   ./scripts/sglang-auto-workflow.sh --model Qwen3.5-397B-A17B --torch --scenario optimized
#   ./scripts/sglang-auto-workflow.sh --model DeepSeek-V4-Flash --torch --scenario optimized --runs 2
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

MODEL_CONFIG=""
DEVICE=0
SCENARIO="optimized"
RUNS_OVERRIDE=""
TORCH_PROFILE=false
DRY_RUN=false
TORCH_WITH_STACK="false"
TORCH_RECORD_SHAPES="false"
TORCH_PROFILE_MEMORY="false"
TORCH_WITH_MODULES="false"
TORCH_PROFILER_LIGHT="true"

usage() {
    head -12 "$0" | tail -9
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)
                MODEL_CONFIG="$2"
                shift 2
                ;;
            --device)
                DEVICE="$2"
                shift 2
                ;;
            --scenario|--scnario)
                SCENARIO="$2"
                shift 2
                ;;
            --runs)
                RUNS_OVERRIDE="$2"
                shift 2
                ;;
            --torch)
                TORCH_PROFILE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --torch-with-stack)
                TORCH_WITH_STACK="$2"
                shift 2
                ;;
            --torch-record-shapes)
                TORCH_RECORD_SHAPES="$2"
                shift 2
                ;;
            --torch-profile-memory)
                TORCH_PROFILE_MEMORY="$2"
                shift 2
                ;;
            --torch-with-modules)
                TORCH_WITH_MODULES="$2"
                shift 2
                ;;
            --torch-profiler-light)
                TORCH_PROFILER_LIGHT="$2"
                shift 2
                ;;
            -h|--help)
                usage
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
    case "$SCENARIO" in
        optimized|full|shape) ;;
        *)
            log_error "--scenario 仅支持 optimized|full|shape，当前值: $SCENARIO"
            exit 1
            ;;
    esac
    if [[ -n "$RUNS_OVERRIDE" && ! "$RUNS_OVERRIDE" =~ ^[1-9][0-9]*$ ]]; then
        log_error "--runs 必须是大于 0 的整数，当前值: $RUNS_OVERRIDE"
        exit 1
    fi
    if [[ "$TORCH_PROFILE" != "true" ]]; then
        log_error "SGLang 当前入口只支持 --torch profiling"
        exit 1
    fi
}

main() {
    parse_args "$@"
    validate_args

    local args=()
    args+=("--model" "$MODEL_CONFIG")
    args+=("--device" "$DEVICE")
    args+=("--scenario" "$SCENARIO")
    args+=("--torch")
    if [[ -n "$RUNS_OVERRIDE" ]]; then
        args+=("--runs" "$RUNS_OVERRIDE")
    fi
    if [[ "$DRY_RUN" == "true" ]]; then
        args+=("--dry-run")
    fi
    args+=("--torch-with-stack" "$TORCH_WITH_STACK")
    args+=("--torch-record-shapes" "$TORCH_RECORD_SHAPES")
    args+=("--torch-profile-memory" "$TORCH_PROFILE_MEMORY")
    args+=("--torch-with-modules" "$TORCH_WITH_MODULES")
    args+=("--torch-profiler-light" "$TORCH_PROFILER_LIGHT")

    log_info "SGLang profiling workflow"
    log_info "模型配置: config.yaml.${MODEL_CONFIG}"
    log_info "场景: ${SCENARIO}"
    log_step "启动 SGLang Torch profiler"
    "${SCRIPT_DIR}/sglang-run-workflow.sh" "${args[@]}"
    log_info "完成"
}

main "$@"
