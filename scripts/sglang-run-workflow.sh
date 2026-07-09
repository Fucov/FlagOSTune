#!/usr/bin/env bash
#
# sglang-run-workflow.sh - SGLang profiling config resolver and runner
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TOOL_CONFIG="${SCRIPT_DIR}/tools/sglang_tool_config.yaml"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"
Python_EXECUTABLE="${Python_EXECUTABLE:-$(which python3 2>/dev/null || echo python3)}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

MODEL_CONFIG=""
DEVICE=0
SCENARIO_TYPE="optimized"
TORCH_PROFILE=false
RUNS_OVERRIDE=""
DRY_RUN=false
TORCH_WITH_STACK="false"
TORCH_RECORD_SHAPES="false"
TORCH_PROFILE_MEMORY="false"
TORCH_WITH_MODULES="false"
TORCH_PROFILER_LIGHT="true"

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
                SCENARIO_TYPE="$2"
                shift 2
                ;;
            --torch)
                TORCH_PROFILE=true
                shift
                ;;
            --runs)
                RUNS_OVERRIDE="$2"
                shift 2
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
    local missing=()
    command -v yq >/dev/null 2>&1 || missing+=("yq")
    command -v "$Python_EXECUTABLE" >/dev/null 2>&1 || missing+=("$Python_EXECUTABLE")
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "缺少必需依赖: ${missing[*]}"
        exit 1
    fi
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
    if [[ "$TORCH_PROFILE" != "true" ]]; then
        log_error "SGLang 当前只支持 --torch profiling"
        exit 1
    fi
    case "$SCENARIO_TYPE" in
        optimized|full|shape) ;;
        *)
            log_error "--scenario 仅支持 optimized|full|shape，当前值: $SCENARIO_TYPE"
            exit 1
            ;;
    esac
    if [[ -n "$RUNS_OVERRIDE" && ! "$RUNS_OVERRIDE" =~ ^[1-9][0-9]*$ ]]; then
        log_error "--runs 必须是大于 0 的整数，当前值: $RUNS_OVERRIDE"
        exit 1
    fi
}

update_tool_config() {
    log_step "更新 SGLang 工具配置..."
    mkdir -p "$(dirname "$TOOL_CONFIG")"
    printf "{}\n" >"$TOOL_CONFIG"

    local model_path model_name tokenizer_path tensor_parallel
    local paths_results paths_reports paths_use_model_name
    local path_prefix report_prefix log_dir torch_output_dir reports_dir
    local benchmark_host benchmark_num_runs

    model_path=$(yq '.model.path' "$CONFIG_FILE")
    model_name=$(yq '.model.name' "$CONFIG_FILE")
    tokenizer_path=$(yq '.model.tokenizer_path // ""' "$CONFIG_FILE")
    tensor_parallel=$(yq '.model.tensor_parallel_size // .serve.tensor_parallel_size // 1' "$CONFIG_FILE")

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

    log_dir="${path_prefix}/sglang-bench_${SCENARIO_TYPE}_torch_profile_log/sglang_bench_logs"
    torch_output_dir="${path_prefix}/sglang-torch-raw"
    reports_dir="${report_prefix}"

    benchmark_host=$(yq '.benchmark.host // "127.0.0.1"' "$CONFIG_FILE")
    benchmark_num_runs=$(yq '.benchmark.num_runs // 2' "$CONFIG_FILE")

    if [[ -n "$RUNS_OVERRIDE" ]]; then
        benchmark_num_runs="$RUNS_OVERRIDE"
    elif (( benchmark_num_runs < 2 )); then
        benchmark_num_runs=2
        log_warn "SGLang torch profiling 至少运行 2 轮，已将 benchmark.num_runs 调整为 2"
    fi

    yq -i ".current_run.device = \"${DEVICE}\"" "$TOOL_CONFIG"
    yq -i ".current_run.scenario_type = \"${SCENARIO_TYPE}\"" "$TOOL_CONFIG"
    yq -i ".current_run.torch_profile = true" "$TOOL_CONFIG"
    yq -i ".current_run.torch_profiler_light = ${TORCH_PROFILER_LIGHT}" "$TOOL_CONFIG"
    yq -i ".current_run.torch_with_stack = ${TORCH_WITH_STACK}" "$TOOL_CONFIG"
    yq -i ".current_run.torch_record_shapes = ${TORCH_RECORD_SHAPES}" "$TOOL_CONFIG"
    yq -i ".current_run.torch_profile_memory = ${TORCH_PROFILE_MEMORY}" "$TOOL_CONFIG"
    yq -i ".current_run.torch_with_modules = ${TORCH_WITH_MODULES}" "$TOOL_CONFIG"

    yq -i ".model.path = \"${model_path}\"" "$TOOL_CONFIG"
    yq -i ".model.name = \"${model_name}\"" "$TOOL_CONFIG"
    yq -i ".model.tokenizer_path = \"${tokenizer_path}\"" "$TOOL_CONFIG"
    yq -i ".model.tensor_parallel_size = ${tensor_parallel}" "$TOOL_CONFIG"

    yq -i ".serve = $(yq '.serve // {}' "$CONFIG_FILE" -o=json)" "$TOOL_CONFIG"
    yq -i ".sglang = $(yq '.sglang // {}' "$CONFIG_FILE" -o=json)" "$TOOL_CONFIG"

    # 关键修复：完整复制 benchmark，避免 dataset_name / dataset_path 丢失
    yq -i ".benchmark = $(yq '.benchmark // {}' "$CONFIG_FILE" -o=json)" "$TOOL_CONFIG"

    # 再覆盖 workflow 运行时控制字段
    yq -i ".benchmark.host = \"${benchmark_host}\"" "$TOOL_CONFIG"
    yq -i ".benchmark.num_runs = ${benchmark_num_runs}" "$TOOL_CONFIG"
    yq -i ".benchmark.scenarios = $(yq '.benchmark.scenarios' "$CONFIG_FILE" -o=json)" "$TOOL_CONFIG"

    yq -i ".paths.results = \"${paths_results}\"" "$TOOL_CONFIG"
    yq -i ".paths.reports = \"${paths_reports}\"" "$TOOL_CONFIG"
    yq -i ".paths.use_model_name = ${paths_use_model_name}" "$TOOL_CONFIG"
    yq -i ".paths.log_dir = \"${log_dir}\"" "$TOOL_CONFIG"
    yq -i ".paths.torch_output_dir = \"${torch_output_dir}\"" "$TOOL_CONFIG"
    yq -i ".paths.reports_dir = \"${reports_dir}\"" "$TOOL_CONFIG"
    yq -i ".paths.model_name = \"${model_name}\"" "$TOOL_CONFIG"

    log_info "日志目录: ${log_dir}"
    log_info "Profiler 原始目录: ${torch_output_dir}/report-sglang"
    log_info "报告目录: ${reports_dir}"
}

collect_metadata() {
    local phase="$1"
    local model_name path_prefix report_prefix trace_dir metadata_dir smi_file
    model_name=$(yq '.paths.model_name' "$TOOL_CONFIG")
    path_prefix=$(dirname "$(yq '.paths.torch_output_dir' "$TOOL_CONFIG")")
    report_prefix=$(yq '.paths.reports_dir' "$TOOL_CONFIG")
    trace_dir="$(yq '.paths.torch_output_dir' "$TOOL_CONFIG")/report-sglang"
    metadata_dir="${path_prefix}/sglang-run-metadata"
    mkdir -p "${PROJECT_ROOT}/${metadata_dir}" "${PROJECT_ROOT}/${report_prefix}"
    smi_file="${PROJECT_ROOT}/${metadata_dir}/nvidia_smi_${phase}.txt"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi >"$smi_file" 2>&1 || true
    fi
    if [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi
    SGLANG_TORCH_PROFILER_LIGHT=$([[ "$TORCH_PROFILER_LIGHT" == "true" ]] && echo 1 || echo 0) \
    SGLANG_TORCH_PROFILER_WITH_STACK=$([[ "$TORCH_WITH_STACK" == "true" ]] && echo 1 || echo 0) \
    SGLANG_TORCH_PROFILER_RECORD_SHAPES=$([[ "$TORCH_RECORD_SHAPES" == "true" ]] && echo 1 || echo 0) \
    SGLANG_TORCH_PROFILER_PROFILE_MEMORY=$([[ "$TORCH_PROFILE_MEMORY" == "true" ]] && echo 1 || echo 0) \
    SGLANG_TORCH_PROFILER_WITH_MODULES=$([[ "$TORCH_WITH_MODULES" == "true" ]] && echo 1 || echo 0) \
    "$Python_EXECUTABLE" "${SCRIPT_DIR}/tools/sglang_collect_metadata.py" \
        --model "$model_name" \
        --config "$CONFIG_FILE" \
        --output-dir "${PROJECT_ROOT}/${metadata_dir}" \
        --report-dir "${PROJECT_ROOT}/${report_prefix}" \
        --trace-dir "${PROJECT_ROOT}/${trace_dir}" \
        --rank "0" \
        --phase "$phase" \
        --workflow-command "$0 $*" \
        --nvidia-smi-file "$smi_file" || log_warn "metadata 采集失败: ${phase}"
}

run_python_runner() {
    local args=()
    if [[ "$DRY_RUN" == "true" ]]; then
        args+=("--dry-run")
    fi
    cd "$PROJECT_ROOT"
    Python_EXECUTABLE="$Python_EXECUTABLE" "$Python_EXECUTABLE" "${SCRIPT_DIR}/tools/sglang_profile_runner.py" "${args[@]}"
}

patch_sglang_compat() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warn "dry-run 模式跳过 SGLang Qwen3 vision_config 兼容补丁"
        return 0
    fi

    local apply_patch
    apply_patch=$(yq '.sglang.apply_qwen3_vision_config_patch // false' "$TOOL_CONFIG")
    if [[ "$apply_patch" != "true" ]]; then
        log_info "未启用 SGLang Qwen3 vision_config 兼容补丁"
        return 0
    fi

    log_step "应用 SGLang Qwen3 vision_config 兼容补丁..."
    "$Python_EXECUTABLE" "${SCRIPT_DIR}/tools/patch_sglang_qwen3_vision_config.py" --apply
}

main() {
    parse_args "$@"
    validate_args
    check_dependencies
    resolve_config_file
    update_tool_config
    collect_metadata "before"
    patch_sglang_compat
    run_python_runner
    collect_metadata "after_profile"
}

main "$@"
