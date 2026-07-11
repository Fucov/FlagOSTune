#!/usr/bin/env bash
#
# sglang-run-processing.sh - SGLang profiler analyzer launcher
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOL_CONFIG="${SCRIPT_DIR}/tools/sglang_tool_config.yaml"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"
Python_EXECUTABLE="${Python_EXECUTABLE:-$(which python3 2>/dev/null || echo python3)}"

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
USE_CACHE=""
FORCE_REPARSE=""
TOP_K=""
TOP_KERNELS_PER_OP=""
SOURCE_MAP=""

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
            --use-cache)
                USE_CACHE="$2"
                shift 2
                ;;
            --force-reparse)
                FORCE_REPARSE="$2"
                shift 2
                ;;
            --top-k)
                TOP_K="$2"
                shift 2
                ;;
            --top-kernels-per-op)
                TOP_KERNELS_PER_OP="$2"
                shift 2
                ;;
            --source-map)
                SOURCE_MAP="$2"
                shift 2
                ;;
            -h|--help)
                head -4 "$0" | tail -2
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                exit 1
                ;;
        esac
    done
}

check_dependencies() {
    command -v yq >/dev/null 2>&1 || { log_error "缺少 yq"; exit 1; }
    command -v "$Python_EXECUTABLE" >/dev/null 2>&1 || { log_error "缺少 Python: $Python_EXECUTABLE"; exit 1; }
}

resolve_config_file() {
    if [[ -n "$MODEL_CONFIG" ]]; then
        CONFIG_FILE="${PROJECT_ROOT}/config.yaml.${MODEL_CONFIG}"
        if [[ ! -f "$CONFIG_FILE" ]]; then
            log_error "配置文件不存在: $CONFIG_FILE"
            exit 1
        fi
    elif [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "默认配置文件不存在: $CONFIG_FILE"
        exit 1
    fi
    log_info "使用模型配置: $CONFIG_FILE"
}

validate_args() {
    if [[ "$WORKFLOW" != "torch" ]]; then
        log_error "SGLang 当前仅支持 --workflow torch"
        exit 1
    fi
    if [[ ! "$RANK" =~ ^[0-9]+$ && "$RANK" != "all" ]]; then
        log_error "--rank 仅支持数字或 all，当前值: $RANK"
        exit 1
    fi
}

update_tool_config() {
    log_step "更新 SGLang 报告配置..."
    mkdir -p "$(dirname "$TOOL_CONFIG")"
    if [[ ! -f "$TOOL_CONFIG" ]]; then
        printf "{}\n" >"$TOOL_CONFIG"
    fi

    local model_name paths_results paths_reports paths_use_model_name path_prefix report_prefix
    model_name=$(yq '.model.name // "default"' "$CONFIG_FILE")
    if [[ -n "$MODEL_CONFIG" && "$MODEL_CONFIG" != "$model_name" ]]; then
        log_error "metadata mismatch: requested model does not match config model"
        exit 1
    fi
    paths_results=$(yq '.paths.results // "results"' "$CONFIG_FILE")
    paths_reports=$(yq '.paths.reports // "reports"' "$CONFIG_FILE")
    paths_use_model_name=$(yq '.paths.use_model_name // true' "$CONFIG_FILE")
    if [[ "$paths_use_model_name" == "true" ]]; then
        path_prefix="${paths_results}/${model_name}"
        report_prefix="${paths_reports}/${model_name}"
    else
        path_prefix="${paths_results}"
        report_prefix="${paths_reports}"
    fi

    yq -i ".paths.results = \"${paths_results}\"" "$TOOL_CONFIG"
    yq -i ".paths.reports = \"${paths_reports}\"" "$TOOL_CONFIG"
    yq -i ".paths.use_model_name = ${paths_use_model_name}" "$TOOL_CONFIG"
    yq -i ".paths.model_name = \"${model_name}\"" "$TOOL_CONFIG"
    yq -i ".paths.torch_output_dir = \"${path_prefix}/sglang-torch-raw\"" "$TOOL_CONFIG"
    yq -i ".paths.reports_dir = \"${report_prefix}\"" "$TOOL_CONFIG"
}

run_analyzer() {
    local torch_output_dir reports_dir model_name scenario_name tp_size run_metadata_path
    torch_output_dir=$(yq '.paths.torch_output_dir' "$TOOL_CONFIG")
    reports_dir=$(yq '.paths.reports_dir' "$TOOL_CONFIG")
    model_name=$(yq '.model.name // ""' "$CONFIG_FILE")
    scenario_name=$(yq '.benchmark.scenarios.optimized[0].name // ""' "$CONFIG_FILE")
    tp_size=$(yq '.model.tensor_parallel_size // 0' "$CONFIG_FILE")
    run_metadata_path="${reports_dir}/run_metadata.json"

    if [[ -z "$model_name" || -z "$scenario_name" || ! "$tp_size" =~ ^[0-9]+$ || "$tp_size" == "0" ]]; then
        log_error "metadata mismatch: config model/scenario/tp_size is incomplete"
        exit 1
    fi

    local args=(
        "--torch_path" "$torch_output_dir"
        "--output_path" "$reports_dir"
        "--rank" "$RANK"
        "--config-path" "$CONFIG_FILE"
        "--expected-model" "$model_name"
        "--expected-scenario" "$scenario_name"
        "--expected-tp-size" "$tp_size"
        "--run-metadata" "$run_metadata_path"
    )
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
    if [[ -n "$USE_CACHE" ]]; then
        args+=("--use-cache" "$USE_CACHE")
    fi
    if [[ -n "$FORCE_REPARSE" ]]; then
        args+=("--force-reparse" "$FORCE_REPARSE")
    fi
    if [[ -n "$TOP_K" ]]; then
        args+=("--top-k" "$TOP_K")
    fi
    if [[ -n "$TOP_KERNELS_PER_OP" ]]; then
        args+=("--top-kernels-per-op" "$TOP_KERNELS_PER_OP")
    fi
    if [[ -n "$SOURCE_MAP" ]]; then
        args+=("--source-map" "$SOURCE_MAP")
    fi

    cd "$PROJECT_ROOT"
    PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}" "$Python_EXECUTABLE" "${SCRIPT_DIR}/tools/sglang_perf_analysis_torch.py" "${args[@]}"
}

main() {
    parse_args "$@"
    validate_args
    check_dependencies
    resolve_config_file
    update_tool_config
    run_analyzer
}

main "$@"
