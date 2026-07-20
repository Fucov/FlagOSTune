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
PARSE_REPORT=false
PARSE_TOP=20
PARSE_OUTPUT_DIR=""
FORCE_PARSE_EXPORT=false
ANALYZE_DEPENDENCIES=false
ANALYZE_COMMUNICATION=false
DEPENDENCY_TRACE=false
CAPTURE_MODE="server-full"
PROFILE_READY_TIMEOUT=3600
CUDA_GRAPH_TRACE="node"
LAYERWISE_NVTX="auto"

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
  --capture-mode MODE   server-full|full-offline，默认 server-full
  --profile-ready-timeout N server readiness/benchmark 超时秒数，默认 3600
  --cuda-graph-trace M graph|node|none，默认 node；none 表示不传该 Nsight 选项
  --layerwise-nvtx M   auto|true|false，默认 auto
  --parse               采集成功后自动运行 parse_nsys.py
  --parse-top N         Markdown 表格行数，默认 20
  --parse-output-dir D  parser 输出目录，默认报告旁的 summary/
  --force-parse-export  强制 parser 重新导出 SQLite
  --analyze-dependencies 生成 same-stream 时序邻接分析（需要 trace report）
  --analyze-communication 生成通信 overlap/chain/fusion 候选分析
  --dependency-trace    启用昂贵的 --cuda-event-trace=true，默认关闭
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
            --capture-mode)
                require_value "$1" "${2-}"
                CAPTURE_MODE="$2"
                shift 2
                ;;
            --profile-ready-timeout)
                require_value "$1" "${2-}"
                PROFILE_READY_TIMEOUT="$2"
                shift 2
                ;;
            --cuda-graph-trace)
                require_value "$1" "${2-}"
                CUDA_GRAPH_TRACE="$2"
                shift 2
                ;;
            --layerwise-nvtx)
                require_value "$1" "${2-}"
                LAYERWISE_NVTX="$2"
                shift 2
                ;;
            --parse)
                PARSE_REPORT=true
                shift
                ;;
            --parse-top)
                require_value "$1" "${2-}"
                PARSE_TOP="$2"
                shift 2
                ;;
            --parse-output-dir)
                require_value "$1" "${2-}"
                PARSE_OUTPUT_DIR="$2"
                shift 2
                ;;
            --force-parse-export)
                FORCE_PARSE_EXPORT=true
                shift
                ;;
            --analyze-dependencies)
                ANALYZE_DEPENDENCIES=true
                shift
                ;;
            --analyze-communication)
                ANALYZE_COMMUNICATION=true
                shift
                ;;
            --dependency-trace)
                DEPENDENCY_TRACE=true
                shift
                ;;
            --profile|--torch-profile|--torch-profiler)
                log_error "Nsight Systems workflow 不能同时开启 Torch Profiler（收到 $1）"
                exit 2
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

extra_arg_value() {
    local option="$1"
    shift
    local index token
    for ((index = 1; index <= $#; index++)); do
        token="${!index}"
        if [[ "$token" == "$option" && $index -lt $# ]]; then
            index=$((index + 1))
            printf '%s' "${!index}"
            return 0
        fi
        if [[ "$token" == "${option}="* ]]; then
            printf '%s' "${token#*=}"
            return 0
        fi
    done
    return 1
}

extra_args_disable_cuda_graph() {
    local token
    for token in "$@"; do
        case "$token" in
            --disable-cuda-graph|--cuda-graph-backend-decode=disabled|--cuda-graph-backend-prefill=disabled)
                return 0
                ;;
        esac
    done
    local decode_backend prefill_backend
    decode_backend=$(extra_arg_value --cuda-graph-backend-decode "$@" || true)
    prefill_backend=$(extra_arg_value --cuda-graph-backend-prefill "$@" || true)
    [[ "$decode_backend" == "disabled" || "$prefill_backend" == "disabled" ]]
}

print_command() {
    local label="$1"
    shift
    local item
    printf '%s ' "$label"
    for item in "$@"; do
        if [[ "$item" =~ ^[A-Za-z0-9_./:=,@+-]+$ ]]; then
            printf '%s ' "$item"
        else
            printf '%q ' "$item"
        fi
    done
    printf '\n'
}

build_parse_command() {
    local report_path="$1"
    parse_cmd=(
        "$PYTHON_EXECUTABLE" "${SCRIPT_DIR}/tools/parse_nsys.py"
        "$report_path" --top "$PARSE_TOP"
    )
    if [[ -n "$PARSE_OUTPUT_DIR" ]]; then
        parse_cmd+=(--output-dir "$PARSE_OUTPUT_DIR")
    fi
    if [[ "$FORCE_PARSE_EXPORT" == "true" ]]; then
        parse_cmd+=(--force-export)
    fi
    if [[ "$ANALYZE_DEPENDENCIES" == "true" ]]; then
        parse_cmd+=(--analyze-dependencies)
    fi
    if [[ "$ANALYZE_COMMUNICATION" == "true" ]]; then
        parse_cmd+=(--analyze-communication)
    fi
}

write_capture_metadata() {
    local metadata_path="$1"
    local report_path="$2"
    local model_name="$3"
    local model_path="$4"
    local scenario_name="$5"
    local dataset_name="$6"
    local tensor_parallel="$7"
    local output_prefix="$8"
    local nsys_log="$9"
    shift 9
    local input_tokens="$1"
    local output_tokens="$2"
    local num_prompts="$3"
    local concurrency="$4"
    local cuda_graph_enabled="$5"
    local layerwise_nvtx_enabled="$6"
    shift 6
    local visible_devices="${CUDA_VISIBLE_DEVICES:-}"
    local git_commit nsys_version
    git_commit=$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || true)
    nsys_version=$(nsys --version 2>&1 || true)
    "$PYTHON_EXECUTABLE" - \
        "$metadata_path" "$report_path" "$model_name" "$model_path" \
        "$scenario_name" "$dataset_name" "$tensor_parallel" "$visible_devices" \
        "$CONFIG_FILE" "$output_prefix" "$nsys_log" "$CAPTURE_MODE" \
        "$git_commit" "$PROJECT_ROOT" "${SCRIPT_DIR}/sglang-nsys-workflow.sh" \
        "${SCRIPT_DIR}/tools/parse_nsys.py" "$nsys_version" \
        "${CAPTURE_START_WALL:-}" "${CAPTURE_END_WALL:-}" \
        "${CAPTURE_DURATION_SECONDS:-}" full \
        "$input_tokens" "$output_tokens" "$num_prompts" "$concurrency" \
        "$cuda_graph_enabled" "$layerwise_nvtx_enabled" "$@" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

(
    metadata_path, report_path, model_name, model_path, scenario, workload,
    tp_size, visible_devices, config_path, output_prefix, nsys_log,
    capture_mode, git_commit, project_root, workflow_script, parser_script,
    nsys_version, capture_start_wall, capture_end_wall, capture_duration,
    requested_phase, input_tokens, output_tokens, num_prompts, concurrency,
    cuda_graph_enabled, layerwise_nvtx_enabled,
) = sys.argv[1:28]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

dirty = subprocess.run(
    ["git", "-C", project_root, "status", "--porcelain"],
    capture_output=True,
    text=True,
).stdout.strip()
log_text = ""
try:
    with open(nsys_log, encoding="utf-8", errors="replace") as handle:
        log_text = handle.read().lower()
except OSError:
    pass
value = {
    "capture_status": "PASS",
    "input_report": report_path,
    "model": model_name,
    "model_path": model_path,
    "scenario": scenario,
    "workload": workload,
    "tp_size": int(tp_size),
    "visible_devices": visible_devices or None,
    "config_path": config_path,
    "output_prefix": output_prefix,
    "nsys_log": nsys_log,
    "capture_mode": capture_mode,
    "capture_scope": "startup_and_full_process",
    "inference_scope": "startup_and_full_process",
    "requested_phase": requested_phase,
    "profile_phase": "full_process",
    "steady_state_guaranteed": False,
    "num_prompts": int(num_prompts),
    "input_tokens": int(input_tokens),
    "output_tokens": int(output_tokens),
    "concurrency": int(concurrency),
    "benchmark_throughput": None,
    "cuda_graph_enabled": cuda_graph_enabled == "true",
    "layerwise_nvtx_enabled": layerwise_nvtx_enabled == "true",
    "deepgemm_jit_detected": "deepgemm" in log_text and "jit" in log_text,
    "moe_config_fallback_detected": "moe" in log_text and "fallback" in log_text,
    "git_commit": git_commit or None,
    "git_dirty": bool(dirty),
    "workflow_sha256": sha256(workflow_script),
    "parser_sha256": sha256(parser_script),
    "nsys_version": nsys_version or None,
    "capture_start_wall_time": capture_start_wall or None,
    "capture_end_wall_time": capture_end_wall or None,
    "capture_duration_seconds": float(capture_duration) if capture_duration else None,
    "benchmark_start_wall_time": capture_start_wall or None,
    "benchmark_end_wall_time": capture_end_wall or None,
    "benchmark_duration_seconds": float(capture_duration) if capture_duration else None,
    "command": sys.argv[28:],
    "generated_time": datetime.now(timezone.utc).astimezone().isoformat(),
}
temporary = metadata_path + ".tmp"
with open(temporary, "w", encoding="utf-8") as handle:
    json.dump(value, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, metadata_path)
PY
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
    local benchmark_num_runs="${17}"
    local base scenario_name input_len output_len num_prompts scenario_concurrency
    local host port configured_port graph_disabled layerwise_enabled
    local output_prefix expected_report nsys_log server_log benchmark_log metadata_path report_size nsys_status

    base=".benchmark.scenarios.${SCENARIO_TYPE}[${index}]"
    scenario_name=$(yq_read "${base}.name // \"scenario-$((index + 1))\"")
    input_len=$(yq_read "${base}.input_len // 4096")
    output_len=$(yq_read "${base}.output_len // 1024")
    num_prompts=$(yq_read "${base}.num_prompts // ${base}.concurrency // 1")
    scenario_concurrency=$(yq_read "${base}.concurrency // 1")
    host=$(empty_if_null "$(yq_read '.benchmark.host // "127.0.0.1"')")
    port=$(yq_read '.benchmark.port // .benchmark.port_base // 30000')

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
    nsys_log="${output_prefix}.nsys.log"
    server_log="${output_prefix}.server.log"
    benchmark_log="${output_prefix}.benchmark.log"
    metadata_path="${expected_report}.metadata.json"
    graph_disabled=false
    if extra_args_disable_cuda_graph "${extra_tokens[@]}"; then
        graph_disabled=true
    fi
    if [[ "$CAPTURE_MODE" == "full-offline" ]]; then
        cmd=(
            nsys profile
            --trace-fork-before-exec=true
            --trace=cuda,nvtx,osrt
            --sample=none
            --cpuctxsw=none
            --capture-range=cudaProfilerApi
            --capture-range-end=stop
            --force-overwrite=true
            --output "$output_prefix"
            "$PYTHON_EXECUTABLE" "${SCRIPT_DIR}/tools/cuda_profiler_launcher.py"
            --module sglang.bench_offline_throughput --
            "${sglang_args[@]}"
        )
        if [[ "$DEPENDENCY_TRACE" == "true" ]]; then
            cmd=("${cmd[@]:0:8}" --cuda-event-trace=true "${cmd[@]:8}")
        fi
    else
        configured_port=$(extra_arg_value --port "${extra_tokens[@]}" || true)
        if [[ -n "$configured_port" ]]; then
            port="$configured_port"
        fi
        server_args=(
            --model-path "$model_path"
            --tokenizer-path "$tokenizer_path"
            --tp-size "$tensor_parallel"
            --host "$host"
        )
        if [[ -z "$configured_port" ]]; then
            server_args+=(--port "$port")
        fi
        if [[ "$trust_remote_code" == "true" ]]; then
            server_args+=(--trust-remote-code)
        fi
        [[ -n "$mem_fraction" ]] && server_args+=(--mem-fraction-static "$mem_fraction")
        [[ -n "$context_length" ]] && server_args+=(--context-length "$context_length")
        [[ -n "$dtype" ]] && server_args+=(--dtype "$dtype")
        [[ -n "$quantization" ]] && server_args+=(--quantization "$quantization")
        [[ -n "$load_format" ]] && server_args+=(--load-format "$load_format")
        server_args+=("${extra_tokens[@]}")

        layerwise_enabled=false
        case "$LAYERWISE_NVTX" in
            true) layerwise_enabled=true ;;
            auto)
                if [[ "$graph_disabled" == "true" ]]; then
                    layerwise_enabled=true
                else
                    log_info "CUDA Graph enabled: layerwise NVTX auto mode is disabled"
                fi
                ;;
        esac
        if [[ "$layerwise_enabled" == "true" ]]; then
            server_args+=(--enable-layerwise-nvtx-marker)
            if [[ "$graph_disabled" != "true" ]]; then
                log_info "WARNING: layerwise NVTX was forced while CUDA Graph is enabled"
            fi
        fi

        client_common=(
            --backend sglang
            --host "$host"
            --port "$port"
            --model "$model_path"
            --tokenizer "$tokenizer_path"
            --dataset-name "$dataset_name"
        )
        if [[ -n "$dataset_path" ]]; then
            client_common+=(--dataset-path "$dataset_path")
        fi
        if [[ "$dataset_name" == "random" ]]; then
            client_common+=(--random-input-len "$input_len" --random-output-len "$output_len")
        else
            client_common+=(--sharegpt-output-len "$output_len")
        fi
        benchmark_cmd=(
            "$PYTHON_EXECUTABLE" -m sglang.bench_serving
            "${client_common[@]}"
            --num-prompts "$num_prompts"
            --max-concurrency "$scenario_concurrency"
        )
        nsys_cmd=(
            nsys profile
            --trace-fork-before-exec=true
            --trace=cuda,nvtx,osrt
            --sample=none
            --cpuctxsw=none
            --capture-range=cudaProfilerApi
            --capture-range-end=stop
        )
        if [[ "$DEPENDENCY_TRACE" == "true" ]]; then
            nsys_cmd+=(--cuda-event-trace=true)
        fi
        if [[ "$CUDA_GRAPH_TRACE" != "none" ]]; then
            nsys_cmd+=("--cuda-graph-trace=${CUDA_GRAPH_TRACE}")
        fi
        nsys_cmd+=(
            --force-overwrite=true
            --output "$output_prefix"
            "$PYTHON_EXECUTABLE" "${SCRIPT_DIR}/tools/sglang_server_capture.py"
            exec-server --log "$server_log" --
            "$PYTHON_EXECUTABLE" -m sglang.launch_server
            "${server_args[@]}"
        )
        cmd=(
            "$PYTHON_EXECUTABLE" "${SCRIPT_DIR}/tools/sglang_server_capture.py" run
            --output-prefix "$output_prefix"
            --report "$expected_report"
            --metadata "$metadata_path"
            --server-log "$server_log"
            --nsys-log "$nsys_log"
            --benchmark-log "$benchmark_log"
            --base-url "http://${host}:${port}"
            --total-runs "$benchmark_num_runs"
            --concurrency "$scenario_concurrency"
            --profile-ready-timeout "$PROFILE_READY_TIMEOUT"
            --cuda-graph-enabled "$([[ "$graph_disabled" == "true" ]] && printf false || printf true)"
            --cuda-graph-trace "$CUDA_GRAPH_TRACE"
            --layerwise-nvtx-enabled "$layerwise_enabled"
            --project-root "$PROJECT_ROOT"
            --workflow-script "${SCRIPT_DIR}/sglang-nsys-workflow.sh"
            --parser-script "${SCRIPT_DIR}/tools/parse_nsys.py"
            --model "$model_name"
            --model-path "$model_path"
            --tokenizer-path "$tokenizer_path"
            --scenario "$scenario_name"
            --dataset "$dataset_name"
            --num-prompts "$num_prompts"
            --input-tokens "$input_len"
            --output-tokens "$output_len"
            --tp-size "$tensor_parallel"
            --nsys-command "${nsys_cmd[@]}"
            --benchmark-command "${benchmark_cmd[@]}"
        )
    fi

    log_info "场景: ${scenario_name}"
    log_info "模型配置: ${CONFIG_FILE}"
    log_info "模型路径: ${model_path}"
    log_info "设备: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not-set>}"
    log_info "TP size: ${tensor_parallel}"
    log_info "Nsight 输出: ${expected_report}"
    log_info "Nsight 日志: ${nsys_log}"
    build_parse_command "$expected_report"
    if [[ "$DRY_RUN" == "true" ]]; then
        print_command "[DRY-RUN]" "${cmd[@]}"
        if [[ "$PARSE_REPORT" == "true" ]]; then
            print_command "[DRY-RUN PARSE]" "${parse_cmd[@]}"
        fi
        return 0
    fi

    mkdir -p "$(dirname "$output_prefix")"
    # A retry must never leave an earlier PASS metadata file beside a failed
    # capture using the same explicit output prefix.
    "$PYTHON_EXECUTABLE" -c \
        'import pathlib,sys; pathlib.Path(sys.argv[1]).unlink(missing_ok=True)' \
        "$metadata_path"
    : > "$nsys_log"
    print_command "[COMMAND]" "${cmd[@]}" | tee -a "$nsys_log"
    CAPTURE_START_WALL=$("$PYTHON_EXECUTABLE" -c 'from datetime import datetime; print(datetime.now().astimezone().isoformat())')
    CAPTURE_START_SECONDS=$("$PYTHON_EXECUTABLE" -c 'import time; print(time.time())')
    set +e
    "${cmd[@]}" 2>&1 | tee -a "$nsys_log"
    nsys_status=${PIPESTATUS[0]}
    set -e
    CAPTURE_END_WALL=$("$PYTHON_EXECUTABLE" -c 'from datetime import datetime; print(datetime.now().astimezone().isoformat())')
    CAPTURE_END_SECONDS=$("$PYTHON_EXECUTABLE" -c 'import time; print(time.time())')
    CAPTURE_DURATION_SECONDS=$("$PYTHON_EXECUTABLE" -c 'import sys; print(max(0.0, float(sys.argv[2]) - float(sys.argv[1])))' "$CAPTURE_START_SECONDS" "$CAPTURE_END_SECONDS")
    if [[ "$nsys_status" -ne 0 ]]; then
        log_error "nsys profile 失败，退出码 ${nsys_status}；日志: ${nsys_log}"
        return "$nsys_status"
    fi
    if [[ ! -f "$expected_report" ]]; then
        log_error "nsys 命令完成但未找到报告: ${expected_report}"
        exit 1
    fi
    report_size=$(wc -c < "$expected_report" | tr -d '[:space:]')
    if [[ ! "$report_size" =~ ^[1-9][0-9]*$ ]]; then
        log_error "nsys 报告为空: ${expected_report}"
        exit 1
    fi
    log_info "Report size: ${report_size} bytes (${expected_report})"
    if [[ "$CAPTURE_MODE" == "full-offline" ]]; then
        write_capture_metadata \
            "$metadata_path" "$expected_report" "$model_name" "$model_path" \
            "$scenario_name" "$dataset_name" "$tensor_parallel" "$output_prefix" \
            "$nsys_log" "$input_len" "$output_len" "$num_prompts" \
            "$scenario_concurrency" \
            "$([[ "$graph_disabled" == "true" ]] && printf false || printf true)" \
            false "${cmd[@]}"
    elif [[ ! -s "$metadata_path" ]]; then
        log_error "server-full 完成但缺少成功 metadata: ${metadata_path}"
        exit 1
    fi
    log_info "采集元数据: ${metadata_path}"
    if [[ "$PARSE_REPORT" == "true" ]]; then
        print_command "[PARSE COMMAND]" "${parse_cmd[@]}"
        "${parse_cmd[@]}"
    fi
}

main() {
    parse_args "$@"
    if [[ "$NSYS_PROFILE" != "true" ]]; then
        log_error "必须指定 --nsys 才会启动 Nsight Systems profiling"
        exit 2
    fi
    case "$CAPTURE_MODE" in
        server-full|full-offline) ;;
        *)
            log_error "--capture-mode 仅支持 server-full|full-offline，当前值: ${CAPTURE_MODE}"
            exit 2
            ;;
    esac
    case "$CUDA_GRAPH_TRACE" in
        graph|node|none) ;;
        *)
            log_error "--cuda-graph-trace 仅支持 graph|node|none，当前值: ${CUDA_GRAPH_TRACE}"
            exit 2
            ;;
    esac
    case "$LAYERWISE_NVTX" in
        auto|true|false) ;;
        *)
            log_error "--layerwise-nvtx 仅支持 auto|true|false，当前值: ${LAYERWISE_NVTX}"
            exit 2
            ;;
    esac
    if [[ ! "$PROFILE_READY_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
        log_error "--profile-ready-timeout 必须是正整数"
        exit 2
    fi
    if [[ ! "$PARSE_TOP" =~ ^[1-9][0-9]*$ ]]; then
        log_error "--parse-top 必须是正整数"
        exit 2
    fi
    if [[ "$PARSE_REPORT" != "true" && ( "$FORCE_PARSE_EXPORT" == "true" || "$ANALYZE_DEPENDENCIES" == "true" || "$ANALYZE_COMMUNICATION" == "true" || -n "$PARSE_OUTPUT_DIR" ) ]]; then
        log_error "parser 相关参数要求同时指定 --parse"
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
    local benchmark_num_runs scenario_count timestamp index
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
    benchmark_num_runs=$(yq_read '.benchmark.num_runs // 1')

    if [[ -z "$model_path" || -z "$model_name" ]]; then
        log_error "配置中的 model.path 和 model.name 不能为空"
        exit 1
    fi
    validate_model_family "$model_name" "$model_path" "$tensor_parallel"
    if [[ ! "$benchmark_num_runs" =~ ^[1-9][0-9]*$ ]]; then
        log_error "benchmark.num_runs 必须是正整数"
        exit 1
    fi

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
            "$load_format" "$extra_args" "$benchmark_num_runs"
    done
}

main "$@"
