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
    local torch_output_dir reports_dir
    torch_output_dir=$(yq '.paths.torch_output_dir' "$TOOL_CONFIG")
    reports_dir=$(yq '.paths.reports_dir' "$TOOL_CONFIG")

    local args=("--torch_path" "$torch_output_dir" "--output_path" "$reports_dir" "--rank" "$RANK")
    if [[ -n "$WORKERS" ]]; then
        args+=("--workers" "$WORKERS")
    fi

    cd "$PROJECT_ROOT"
    "$Python_EXECUTABLE" "${SCRIPT_DIR}/tools/sglang_perf_analysis_torch.py" "${args[@]}"
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
