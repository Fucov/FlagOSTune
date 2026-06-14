#!/usr/bin/env bash
#
# patch-vllm-per-token-group-fp8.sh - 将 vLLM 的 per_token_group_quant_fp8
#   替换为 FlagGems Triton 实现
#
# 补丁点:
#   vllm/model_executor/layers/quantization/utils/fp8_utils.py
#     _triton_per_token_group_quant_fp8_impl -> vLLM original per-token helper
#     per_token_group_quant_fp8 -> flag_gems.ops.per_token_group_quant_fp8
#
# 行为:
#   用 FlagGems per-token group FP8 quantization 替换 vLLM 原函数。
#
# 用法:
#   ./patch-vllm-per-token-group-fp8.sh --apply
#   ./patch-vllm-per-token-group-fp8.sh --restore
#   ./patch-vllm-per-token-group-fp8.sh --status
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
log_error() { echo -e "${RED}[ERROR]${NC} $1" >&2; }

PATCH_MARKER="# >>> FLAGGEMS PER_TOKEN_GROUP_FP8 PATCH >>>"
PATCH_END="# <<< FLAGGEMS PER_TOKEN_GROUP_FP8 PATCH <<<"
ROUTE_PATCH_MARKER="# >>> FLAGGEMS PER_TOKEN_GROUP_FP8 VLLM_ROUTE PATCH >>>"
ROUTE_PATCH_END="# <<< FLAGGEMS PER_TOKEN_GROUP_FP8 VLLM_ROUTE PATCH <<<"
BAK_SUFFIX=".ptg_fp8_bak"

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
    case $1 in
        --apply)   set_action "apply"; shift ;;
        --restore) set_action "restore"; shift ;;
        --status)  set_action "status"; shift ;;
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
import importlib.util
import os

spec = importlib.util.find_spec('vllm')
if spec is None or spec.origin is None:
    raise SystemExit(1)
print(os.path.dirname(spec.origin))
" 2>/dev/null || { log_error "无法定位 vllm"; flagtune_emit_result "TARGET_MISMATCH"; exit 1; }
}

if ! VLLM_DIR=$(detect_vllm_path); then
    log_error "无法定位 vllm"
    flagtune_emit_result "TARGET_MISMATCH"
    exit 1
fi

FP8_UTILS="${VLLM_DIR}/model_executor/layers/quantization/utils/fp8_utils.py"
TARGETS=("$FP8_UTILS")
OTHER_SHARED_MARKERS=(
    "# >>> FLAGGEMS W8A8 PATCH >>>"
)

target_matches() {
    [[ -f "$FP8_UTILS" ]] && \
        grep -qE "^def per_token_group_quant_fp8\(" "$FP8_UTILS" 2>/dev/null && \
        grep -qE "^def _triton_per_token_group_quant_fp8_impl\(" "$FP8_UTILS" 2>/dev/null
}

is_patched() { flagtune_has_marker_pair "$1" "$PATCH_MARKER" "$PATCH_END"; }
route_is_patched() { flagtune_has_marker_pair "$1" "$ROUTE_PATCH_MARKER" "$ROUTE_PATCH_END"; }

route_patch_correct() {
    local f="$1"
    route_is_patched "$f" && \
        flagtune_has_all "$f" \
            "def _flagtune_vllm_per_token_group_quant_fp8(" \
            "def _triton_per_token_group_quant_fp8_impl(" \
            "return _flagtune_vllm_per_token_group_quant_fp8("
}

patch_correct() {
    local f="$1"
    route_patch_correct "$f" && \
        is_patched "$f" && \
        flagtune_has_all "$f" \
            "def per_token_group_quant_fp8(" \
            "from flag_gems.ops import per_token_group_quant_fp8 as _gems_per_token_group_quant_fp8" \
            "scale_ue8m0=use_ue8m0"
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
         flagtune_has_any_marker "$f" "$ROUTE_PATCH_MARKER" "$ROUTE_PATCH_END" || \
         grep -qF "_gems_per_token_group_quant_fp8" "$f" 2>/dev/null; then
        echo "patched_invalid"
    else
        echo "clean"
    fi
}

backup() {
    local f="$1"
    if [[ ! -f "${f}${BAK_SUFFIX}" ]]; then
        cp "$f" "${f}${BAK_SUFFIX}"
        log_info "备份: ${f}${BAK_SUFFIX}"
    fi
}

restore() {
    local f="$1"
    if [[ -f "${f}${BAK_SUFFIX}" ]]; then
        cp "${f}${BAK_SUFFIX}" "$f"
        rm -f "${f}${BAK_SUFFIX}"
        log_info "已还原: $f"
    else
        log_warn "无备份: $f"
    fi
}

check_status() {
    echo "=== per_token_group_quant_fp8 补丁状态 ==="
    for f in "${TARGETS[@]}"; do
        local name
        name=$(basename "$f")
        if patch_correct "$f"; then
            echo -e "  $name: ${GREEN}已补丁${NC}"
        elif is_patched "$f" || route_is_patched "$f"; then
            echo -e "  $name: ${YELLOW}部分补丁${NC}"
        else
            echo -e "  $name: ${YELLOW}未补丁${NC}"
        fi
        if [[ -f "${f}${BAK_SUFFIX}" ]]; then
            echo -e "    备份: ${GREEN}存在${NC}"
        fi
        if "$PYTHON_EXECUTABLE" -c \
            'import ast, pathlib, sys; p = pathlib.Path(sys.argv[1]); ast.parse(p.read_text(), filename=str(p))' \
            "$f" 2>/dev/null; then
            echo -e "    语法: ${GREEN}正常${NC}"
        else
            echo -e "    语法: ${RED}错误${NC}"
        fi
    done
}

patch_per_token_group_fp8() {
    local f="$FP8_UTILS"
    local state
    state="$(patch_state "$f")"
    case "$state" in
        patched_correct)
            log_warn "fp8_utils.py 已有正确 per_token_group_fp8 补丁，跳过"
            flagtune_emit_result "ALREADY_PATCHED"
            return 0
            ;;
        patched_correct_backup_missing)
            log_warn "fp8_utils.py 已有正确 per_token_group_fp8 补丁，但备份缺失"
            flagtune_emit_result "ALREADY_PATCHED_BACKUP_MISSING"
            return 0
            ;;
        patched_invalid)
            log_warn "fp8_utils.py 存在旧版或部分 per_token_group_fp8 补丁，将尝试补齐"
            ;;
        target_mismatch)
            log_error "fp8_utils.py 和 per_token_group_fp8 补丁脚本预期不匹配"
            flagtune_emit_result "TARGET_MISMATCH"
            return 1
            ;;
    esac
    if [[ "$state" == "patched_invalid" ]] && \
       ! flagtune_backup_exists "$f" "$BAK_SUFFIX"; then
        log_error "检测到旧版或部分 per_token_group_fp8 补丁，但缺少原始备份: ${f}${BAK_SUFFIX}"
        flagtune_emit_result "BACKUP_MISSING"
        return 1
    fi
    backup "$f"

    "$PYTHON_EXECUTABLE" - "$f" "${f}${BAK_SUFFIX}" \
        "$PATCH_MARKER" "$PATCH_END" "$ROUTE_PATCH_MARKER" "$ROUTE_PATCH_END" << 'PYEOF'
import ast
import os
import sys
import tempfile

filepath = sys.argv[1]
backup_path = sys.argv[2]
marker = sys.argv[3]
end_marker = sys.argv[4]
route_marker = sys.argv[5]
route_end_marker = sys.argv[6]

def read(path):
    with open(path, "r") as f:
        return f.read()

content = read(filepath)
backup_content = read(backup_path)

try:
    tree = ast.parse(content, filename=filepath)
except SyntaxError as exc:
    print(f"ERROR: 补丁前文件存在语法错误: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    backup_tree = ast.parse(backup_content, filename=backup_path)
except SyntaxError as exc:
    print(f"ERROR: 备份文件存在语法错误: {exc}", file=sys.stderr)
    sys.exit(1)

def top_function(tree, name):
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(nodes) != 1:
        print(f"ERROR: 找到 {len(nodes)} 个 {name} 函数，期望 1 个", file=sys.stderr)
        sys.exit(1)
    return nodes[0]

def node_text(source, node):
    lines = source.splitlines(keepends=True)
    return "".join(lines[node.lineno - 1:node.end_lineno])

def replace_node(source, node, replacement):
    lines = source.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: node.lineno - 1])
    end = sum(len(line) for line in lines[: node.end_lineno])
    return source[:start] + replacement + source[end:]

def replace_marker_block(source, start_marker, stop_marker, replacement):
    start = source.find(start_marker)
    if start == -1:
        return None
    end = source.find(stop_marker, start + len(start_marker))
    if end == -1:
        print(f"ERROR: 找不到补丁结束 marker: {stop_marker}", file=sys.stderr)
        sys.exit(1)
    line_start = source.rfind("\n", 0, start) + 1
    end += len(stop_marker)
    if end < len(source) and source[end:end + 2] == "\n\n":
        end += 2
    elif end < len(source) and source[end] == "\n":
        end += 1
    return source[:line_start] + replacement + source[end:]

backup_per = top_function(backup_tree, "per_token_group_quant_fp8")
backup_impl = top_function(backup_tree, "_triton_per_token_group_quant_fp8_impl")

helper_text = node_text(backup_content, backup_per).replace(
    "def per_token_group_quant_fp8(",
    "def _flagtune_vllm_per_token_group_quant_fp8(",
    1,
)

route_replacement = f'''{route_marker}
{helper_text}

def _triton_per_token_group_quant_fp8_impl(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _flagtune_vllm_per_token_group_quant_fp8(
        x, group_size, column_major_scales=False, use_ue8m0=False
    )
{route_end_marker}

'''

route_replaced = replace_marker_block(
    content, route_marker, route_end_marker, route_replacement
)
if route_replaced is None:
    current_tree = ast.parse(content, filename=filepath)
    impl_node = top_function(current_tree, "_triton_per_token_group_quant_fp8_impl")
    content = replace_node(content, impl_node, route_replacement)
else:
    content = route_replaced

tree = ast.parse(content, filename=filepath)

replacement = f'''{marker}
def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    tma_aligned_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from flag_gems.ops import per_token_group_quant_fp8 as _gems_per_token_group_quant_fp8

    # FlagGems does not support tma_aligned_scales or out_q yet;
    # fall back to original if tma_aligned_scales is requested.
    if tma_aligned_scales:
        raise NotImplementedError(
            "FlagGems per_token_group_quant_fp8 does not support "
            "tma_aligned_scales=True yet. Restore the original with "
            "--restore if needed."
        )

    if use_ue8m0 is None:
        use_ue8m0 = is_deep_gemm_e8m0_used()

    x_q, x_s = _gems_per_token_group_quant_fp8(
        x,
        group_size,
        eps=eps,
        dtype=dtype,
        column_major_scales=column_major_scales,
        scale_ue8m0=use_ue8m0,
    )

    # If caller provided out_q, copy into it
    if out_q is not None:
        out_q.copy_(x_q)
        return out_q, x_s

    return x_q, x_s
{end_marker}

'''

public_replaced = replace_marker_block(content, marker, end_marker, replacement)
if public_replaced is None:
    node = top_function(tree, "per_token_group_quant_fp8")
    new_content = replace_node(content, node, replacement)
else:
    new_content = public_replaced

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

print("OK: per_token_group_quant_fp8 已替换为 FlagGems，Triton custom op 保留 vLLM 原生 per")
PYEOF

    if ! "$PYTHON_EXECUTABLE" -m py_compile "$f"; then
        log_error "补丁后 py_compile 校验失败，正在从备份恢复"
        cp "${f}${BAK_SUFFIX}" "$f"
        flagtune_emit_result "PATCH_INVALID"
        exit 1
    fi
    flagtune_emit_result "APPLIED"
}

restore_per_token_group_fp8() {
    local f="$FP8_UTILS"
    local backup="${f}${BAK_SUFFIX}"
    if [[ ! -f "$backup" ]]; then
        log_error "备份丢失: $backup"
        return 1
    fi

    "$PYTHON_EXECUTABLE" - "$f" "$backup" \
        "$PATCH_MARKER" "$PATCH_END" "$ROUTE_PATCH_MARKER" "$ROUTE_PATCH_END" << 'PYEOF'
import ast
import os
import sys
import tempfile

target, backup, marker, end_marker, route_marker, route_end_marker = sys.argv[1:]

def read(path):
    with open(path, "r") as f:
        return f.read()

target_content = read(target)
backup_content = read(backup)

try:
    backup_tree = ast.parse(backup_content, filename=backup)
except SyntaxError as exc:
    print(f"ERROR: 备份文件语法错误: {exc}", file=sys.stderr)
    sys.exit(1)

def top_function(tree, name):
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(nodes) != 1:
        print(f"ERROR: 备份中找到 {len(nodes)} 个 {name} 函数，期望 1 个",
              file=sys.stderr)
        sys.exit(1)
    return nodes[0]

def node_text(source, node):
    lines = source.splitlines(keepends=True)
    text = "".join(lines[node.lineno - 1:node.end_lineno])
    if not text.endswith("\n"):
        text += "\n"
    return text + "\n"

def replace_marker_block(source, start_marker, stop_marker, replacement):
    start = source.find(start_marker)
    if start == -1:
        return source
    end = source.find(stop_marker, start + len(start_marker))
    if end == -1:
        print(f"ERROR: 找不到补丁结束 marker: {stop_marker}", file=sys.stderr)
        sys.exit(1)
    line_start = source.rfind("\n", 0, start) + 1
    end += len(stop_marker)
    if end < len(source) and source[end:end + 2] == "\n\n":
        end += 2
    elif end < len(source) and source[end] == "\n":
        end += 1
    return source[:line_start] + replacement + source[end:]

original_impl = node_text(
    backup_content, top_function(backup_tree, "_triton_per_token_group_quant_fp8_impl")
)
original_public = node_text(
    backup_content, top_function(backup_tree, "per_token_group_quant_fp8")
)

new_content = replace_marker_block(
    target_content, route_marker, route_end_marker, original_impl
)
new_content = replace_marker_block(
    new_content, marker, end_marker, original_public
)

try:
    ast.parse(new_content, filename=target)
except SyntaxError as exc:
    print(f"ERROR: 还原后语法校验失败: {exc}", file=sys.stderr)
    sys.exit(1)

tmp_path = None
try:
    fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(target) + ".",
        suffix=".tmp",
        dir=os.path.dirname(target),
    )
    with os.fdopen(fd, "w") as f:
        f.write(new_content)
        f.flush()
        os.fsync(f.fileno())
    os.chmod(tmp_path, os.stat(target).st_mode)
    os.replace(tmp_path, target)
except Exception:
    if tmp_path is not None:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
    raise
PYEOF
}

# 主逻辑
if [[ "$ACTION" == "status" ]]; then
    check_status
    case "$(patch_state "$FP8_UTILS")" in
        patched_correct) flagtune_emit_result "ALREADY_PATCHED" ;;
        patched_correct_backup_missing) flagtune_emit_result "ALREADY_PATCHED_BACKUP_MISSING" ;;
        patched_invalid) flagtune_emit_result "PATCH_INVALID"; exit 1 ;;
        target_mismatch) flagtune_emit_result "TARGET_MISMATCH"; exit 1 ;;
        *) flagtune_emit_result "ALREADY_RESTORED" ;;
    esac
    exit 0
fi

if [[ "$ACTION" == "restore" ]]; then
    state="$(patch_state "$FP8_UTILS")"
    case "$state" in
        clean)
            log_info "未检测到 per_token_group_fp8 补丁，无需还原"
            flagtune_emit_result "ALREADY_RESTORED"
            ;;
        patched_correct)
            if restore_per_token_group_fp8; then
                rm -f "${FP8_UTILS}${BAK_SUFFIX}"
                log_info "已还原 per_token_group_fp8 补丁: $FP8_UTILS"
                flagtune_emit_result "RESTORED"
            else
                flagtune_emit_result "PATCH_INVALID"
                exit 1
            fi
            ;;
        patched_correct_backup_missing)
            log_error "per_token_group_fp8 补丁已存在，但备份丢失: ${FP8_UTILS}${BAK_SUFFIX}"
            flagtune_emit_result "BACKUP_MISSING"
            exit 1
            ;;
        patched_invalid)
            if restore_per_token_group_fp8; then
                rm -f "${FP8_UTILS}${BAK_SUFFIX}"
                log_info "已还原 per_token_group_fp8 补丁: $FP8_UTILS"
                flagtune_emit_result "RESTORED"
            else
                flagtune_emit_result "PATCH_INVALID"
                exit 1
            fi
            ;;
        target_mismatch)
            log_error "fp8_utils.py 和 per_token_group_fp8 补丁脚本预期不匹配"
            flagtune_emit_result "TARGET_MISMATCH"
            exit 1
            ;;
    esac
    exit 0
fi

log_info "vLLM 路径: $VLLM_DIR"
log_info "补丁目标:"
log_info "  fp8_utils.py - triton_per_token_group_quant_fp8 保留 vLLM 原生 per"
log_info "  fp8_utils.py - per_token_group_quant_fp8 → FlagGems Triton"

patch_per_token_group_fp8

log_info "补丁完成！还原命令: $0 --restore"
