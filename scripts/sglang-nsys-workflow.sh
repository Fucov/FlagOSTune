#!/usr/bin/env bash
# Independent SGLang Nsight Systems profiling workflow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${PROJECT_ROOT}/config.yaml"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"

MODEL_CONFIG=""
SCENARIO_TYPE="optimized"
NSYS_PROFILE=false
NSYS_OUTPUT=""
DRY_RUN=false

log_info() { printf '[INFO] %s\n' "$1"; }
log_error() { printf '[ERROR] %s\n' "$1" >&2; }

usage() {
    cat <<'EOF'
用法:
  ./scripts/sglang-nsys-workflow.sh --model <config后缀> --nsys [选项]

选项:
  --model NAME          使用 config.yaml.NAME；省略时使用 config.yaml
  --nsys                启用独立 Nsight Systems profiling（必需）
  --nsys-output PREFIX  输出文件名前缀或路径，可带 .nsys-rep 后缀
  --scenario TYPE       optimized|full|shape，默认 optimized
  --dry-run             校验配置并打印命令，不执行 nsys/SGLang
  -h, --help            显示帮助

默认输出:
  results/<model>/nsys/<model>-<scenario>-<timestamp>.nsys-rep
EOF
}

require_value() {
    local option="$1"
    local value="${2-}"
    if [[ -z "$value" || "$value" == --* ]]; then
        log_error "${option} 需要一个值"
        exit 2
    fi
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)
                require_value "$1" "${2-}"
                MODEL_CONFIG="$2"
                shift 2
                ;;
            --scenario)
                require_value "$1" "${2-}"
                SCENARIO_TYPE="$2"
                shift 2
                ;;
            --nsys)
                NSYS_PROFILE=true
                shift
                ;;
            --nsys-output)
                require_value "$1" "${2-}"
                NSYS_OUTPUT="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                exit 2
                ;;
        esac
    done
}

check_dependencies() {
    local missing=()
    command -v yq >/dev/null 2>&1 || missing+=("yq")
    command -v "$PYTHON_EXECUTABLE" >/dev/null 2>&1 || missing+=("$PYTHON_EXECUTABLE")
    if [[ "$DRY_RUN" != "true" ]]; then
        command -v nsys >/dev/null 2>&1 || missing+=("nsys")
    fi
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "缺少必需依赖: ${missing[*]}"
        exit 1
    fi
}

resolve_config() {
    if [[ -n "$MODEL_CONFIG" ]]; then
        CONFIG_FILE="${PROJECT_ROOT}/config.yaml.${MODEL_CONFIG}"
    fi
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "配置文件不存在: $CONFIG_FILE"
        exit 1
    fi
}

yq_read() {
    yq -r "$1" "$CONFIG_FILE"
}

empty_if_null() {
    if [[ "$1" == "null" ]]; then
        printf ''
    else
        printf '%s' "$1"
    fi
}

append_option() {
    local option="$1"
    local value="$2"
    if [[ -n "$value" && "$value" != "null" ]]; then
        sglang_args+=("$option" "$value")
    fi
}

split_extra_args() {
    local raw="$1"
    extra_tokens=()
    if [[ -z "$raw" || "$raw" == "null" ]]; then
        return 0
    fi
    while IFS= read -r -d '' token; do
        extra_tokens+=("$token")
    done < <(
        "$PYTHON_EXECUTABLE" -c \
            'import shlex,sys; [sys.stdout.buffer.write(x.encode()+b"\0") for x in shlex.split(sys.argv[1])]' \
            "$raw"
    )
    for token in "${extra_tokens[@]}"; do
        if [[ "$token" == "--profile" || "$token" == --profile=* || "$token" == "--profiler-config" || "$token" == --profiler-config=* ]]; then
            log_error "sglang.extra_args 不允许包含 ${token}；Nsight workflow 不启用 Torch Profiler"
            exit 1
        fi
    done
}

print_command() {
    local item
    printf '[DRY-RUN] '
    for item in "$@"; do
        if [[ "$item" =~ ^[A-Za-z0-9_./:=,@+-]+$ ]]; then
            printf '%s ' "$item"
        else
            printf '%q ' "$item"
        fi
    done
    printf '\n'
}

validate_model_family() {
    local model_name="$1"
    local model_path="$2"
    local tensor_parallel="$3"
    local combined lower
    combined="${MODEL_CONFIG} ${model_name} ${model_path}"
    lower=$(printf '%s' "$combined" | tr '[:upper:]' '[:lower:]')

    if [[ "$lower" == *"qwen3.6-35b-a3b-fp8"* ]]; then
        if [[ "$tensor_parallel" != "4" ]]; then
            log_error "Qwen3.6-35B-A3B-FP8 仅支持 TP4，当前 tensor_parallel_size=${tensor_parallel}"
            exit 1
        fi
    elif [[ "$lower" == *"deepseek-v4-flash-fp8"* ]]; then
        if [[ "$tensor_parallel" != "8" ]]; then
            log_error "DeepSeek-V4-Flash-FP8 仅支持 TP8，当前 tensor_parallel_size=${tensor_parallel}"
            exit 1
        fi
    else
        log_error "不支持的 Nsight 模型: ${model_name}"
        exit 1
    fi
}

resolve_output_prefix() {
    local model_name="$1"
    local scenario_name="$2"
    local scenario_count="$3"
    local timestamp="$4"
    local default_dir="${PROJECT_ROOT}/results/${model_name}/nsys"
    local requested clean prefix

    if [[ -z "$NSYS_OUTPUT" ]]; then
        prefix="${default_dir}/${model_name}-${scenario_name}-${timestamp}"
    else
        requested="$NSYS_OUTPUT"
        clean="${requested%.nsys-rep}"
        if [[ "$clean" == /* ]]; then
            prefix="$clean"
        elif [[ "$clean" == */* ]]; then
            prefix="${PROJECT_ROOT}/${clean}"
        else
            prefix="${default_dir}/${clean}"
        fi
        if (( scenario_count > 1 )); then
            prefix="${prefix}-${scenario_name}"
        fi
    fi
    printf '%s' "$prefix"
}

run_scenario() {
    local index="$1"
    local scenario_count="$2"
    local timestamp="$3"
    local model_path="$4"
    local model_name="$5"
    local tokenizer_path="$6"
    local tensor_parallel="$7"
    local dataset_name="$8"
    local dataset_path="$9"
    local trust_remote_code="${10}"
    local dtype="${11}"
    local mem_fraction="${12}"
    local context_length="${13}"
    local quantization="${14}"
    local load_format="${15}"
    local extra_args="${16}"
    local base scenario_name input_len output_len num_prompts
    local output_prefix expected_report

    base=".benchmark.scenarios.${SCENARIO_TYPE}[${index}]"
    scenario_name=$(yq_read "${base}.name // \"scenario-$((index + 1))\"")
    input_len=$(yq_read "${base}.input_len // 4096")
    output_len=$(yq_read "${base}.output_len // 1024")
    num_prompts=$(yq_read "${base}.num_prompts // ${base}.concurrency // 1")

    sglang_args=(
        --model-path "$model_path"
        --tokenizer-path "$tokenizer_path"
        --dataset-name "$dataset_name"
    )
    if [[ -n "$dataset_path" ]]; then
        sglang_args+=(--dataset-path "$dataset_path")
    fi
    case "$dataset_name" in
        random)
            sglang_args+=(--random-input-len "$input_len" --random-output-len "$output_len")
            ;;
        sharegpt)
            sglang_args+=(--sharegpt-output-len "$output_len")
            ;;
        *)
            log_error "SGLang Nsight workflow 仅支持 benchmark.dataset_name=random|sharegpt，当前值: ${dataset_name}"
            exit 1
            ;;
    esac
    sglang_args+=(--num-prompts "$num_prompts" --tp-size "$tensor_parallel")
    if [[ "$trust_remote_code" == "true" ]]; then
        sglang_args+=(--trust-remote-code)
    fi
    append_option --mem-fraction-static "$mem_fraction"
    append_option --context-length "$context_length"
    append_option --dtype "$dtype"
    append_option --quantization "$quantization"
    append_option --load-format "$load_format"
    split_extra_args "$extra_args"
    if [[ ${#extra_tokens[@]} -gt 0 ]]; then
        sglang_args+=("${extra_tokens[@]}")
    fi

    output_prefix=$(resolve_output_prefix "$model_name" "$scenario_name" "$scenario_count" "$timestamp")
    expected_report="${output_prefix}.nsys-rep"
    cmd=(
        nsys profile
        --trace=cuda,nvtx,osrt
        --sample=none
        --cpuctxsw=none
        --capture-range=cudaProfilerApi
        --force-overwrite=true
        --output "$output_prefix"
        "$PYTHON_EXECUTABLE" "${SCRIPT_DIR}/tools/cuda_profiler_launcher.py"
        --module sglang.bench_offline_throughput --
        "${sglang_args[@]}"
    )

    log_info "场景: ${scenario_name}"
    log_info "Nsight 输出: ${expected_report}"
    if [[ "$DRY_RUN" == "true" ]]; then
        print_command "${cmd[@]}"
        return 0
    fi

    mkdir -p "$(dirname "$output_prefix")"
    "${cmd[@]}"
    if [[ ! -f "$expected_report" ]]; then
        log_error "nsys 命令完成但未找到报告: ${expected_report}"
        exit 1
    fi
}

main() {
    parse_args "$@"
    if [[ "$NSYS_PROFILE" != "true" ]]; then
        log_error "必须指定 --nsys 才会启动 Nsight Systems profiling"
        exit 2
    fi
    case "$SCENARIO_TYPE" in
        optimized|full|shape) ;;
        *)
            log_error "--scenario 仅支持 optimized|full|shape，当前值: ${SCENARIO_TYPE}"
            exit 2
            ;;
    esac

    check_dependencies
    resolve_config

    local model_path model_name tokenizer_path tensor_parallel
    local dataset_name dataset_path trust_remote_code
    local dtype mem_fraction context_length quantization load_format extra_args
    local scenario_count timestamp index
    model_path=$(empty_if_null "$(yq_read '.model.path // ""')")
    model_name=$(empty_if_null "$(yq_read '.model.name // ""')")
    tokenizer_path=$(empty_if_null "$(yq_read '.model.tokenizer_path // .model.path // ""')")
    tensor_parallel=$(yq_read '.model.tensor_parallel_size // .serve.tensor_parallel_size // 1')
    dataset_name=$(empty_if_null "$(yq_read '.benchmark.dataset_name // "random"')")
    dataset_path=$(empty_if_null "$(yq_read '.benchmark.dataset_path // ""')")
    trust_remote_code=$(yq_read '.serve.trust_remote_code // .sglang.trust_remote_code // false')
    dtype=$(empty_if_null "$(yq_read '.sglang.dtype // ""')")
    mem_fraction=$(empty_if_null "$(yq_read '.sglang.mem_fraction_static // ""')")
    context_length=$(empty_if_null "$(yq_read '.sglang.context_length // ""')")
    quantization=$(empty_if_null "$(yq_read '.sglang.quantization // ""')")
    load_format=$(empty_if_null "$(yq_read '.sglang.load_format // ""')")
    extra_args=$(empty_if_null "$(yq_read '.sglang.extra_args // .serve.extra_args // ""')")

    if [[ -z "$model_path" || -z "$model_name" ]]; then
        log_error "配置中的 model.path 和 model.name 不能为空"
        exit 1
    fi
    validate_model_family "$model_name" "$model_path" "$tensor_parallel"

    scenario_count=$(yq_read ".benchmark.scenarios.${SCENARIO_TYPE} | length")
    if [[ ! "$scenario_count" =~ ^[1-9][0-9]*$ ]]; then
        log_error "benchmark.scenarios.${SCENARIO_TYPE} 未配置有效场景"
        exit 1
    fi

    timestamp=$(date '+%Y%m%d-%H%M%S')
    log_info "配置: ${CONFIG_FILE}"
    log_info "模型: ${model_name} (TP${tensor_parallel})"
    for ((index = 0; index < scenario_count; index++)); do
        run_scenario \
            "$index" "$scenario_count" "$timestamp" \
            "$model_path" "$model_name" "$tokenizer_path" "$tensor_parallel" \
            "$dataset_name" "$dataset_path" "$trust_remote_code" \
            "$dtype" "$mem_fraction" "$context_length" "$quantization" \
            "$load_format" "$extra_args"
    done
}

main "$@"
