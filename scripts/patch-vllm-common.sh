#!/usr/bin/env bash
#
# Common helpers for FlagTune vLLM patch scripts.

FLAGTUNE_RESULT_EMITTED=0
FLAGTUNE_FAILURE_RESULT="${FLAGTUNE_FAILURE_RESULT:-PATCH_INVALID}"

flagtune_emit_result() {
    FLAGTUNE_RESULT_EMITTED=1
    echo "FLAGTUNE_RESULT=$1"
}

flagtune_emit_unhandled_result() {
    local rc=$?
    if [[ $rc -ne 0 && "${FLAGTUNE_RESULT_EMITTED:-0}" != "1" ]]; then
        echo "FLAGTUNE_RESULT=${FLAGTUNE_FAILURE_RESULT:-PATCH_INVALID}"
    fi
}

trap flagtune_emit_unhandled_result EXIT

flagtune_has_marker_pair() {
    local file="$1"
    local start_marker="$2"
    local end_marker="$3"
    grep -qF "$start_marker" "$file" 2>/dev/null && \
        grep -qF "$end_marker" "$file" 2>/dev/null
}

flagtune_has_any_marker() {
    local file="$1"
    local start_marker="$2"
    local end_marker="$3"
    grep -qF "$start_marker" "$file" 2>/dev/null || \
        grep -qF "$end_marker" "$file" 2>/dev/null
}

flagtune_has_all() {
    local file="$1"
    shift
    local needle
    for needle in "$@"; do
        if ! grep -qF "$needle" "$file" 2>/dev/null; then
            return 1
        fi
    done
    return 0
}

flagtune_backup_exists() {
    local file="$1"
    local suffix="$2"
    [[ -f "${file}${suffix}" ]]
}

flagtune_backup_file() {
    local file="$1"
    local suffix="$2"
    local backup="${file}${suffix}"
    if [[ ! -f "$backup" ]]; then
        cp "$file" "$backup"
        log_info "备份: $backup"
    else
        log_warn "备份已存在: $backup"
    fi
}

flagtune_safe_restore_file() {
    local file="$1"
    local suffix="$2"
    shift 2
    local backup="${file}${suffix}"
    local marker

    if [[ ! -f "$backup" ]]; then
        log_error "备份丢失: $backup"
        return 1
    fi

    for marker in "$@"; do
        if grep -qF "$marker" "$file" 2>/dev/null && \
           ! grep -qF "$marker" "$backup" 2>/dev/null; then
            log_error "拒绝还原: 备份不包含当前文件中的其他补丁 marker: $marker"
            return 2
        fi
    done

    cp "$backup" "$file"
    rm -f "$backup"
    log_info "已还原: $file"
}

flagtune_restore_function_from_backup() {
    local file="$1"
    local suffix="$2"
    local start_marker="$3"
    local end_marker="$4"
    local function_name="$5"
    local backup="${file}${suffix}"

    if [[ ! -f "$backup" ]]; then
        log_error "备份丢失: $backup"
        return 1
    fi

    "$PYTHON_EXECUTABLE" - "$file" "$backup" "$start_marker" "$end_marker" "$function_name" << 'PYEOF'
import ast
import os
import sys
import tempfile

target, backup, marker, end_marker, function_name = sys.argv[1:]

def read(path):
    with open(path, "r") as f:
        return f.read()

target_content = read(target)
backup_content = read(backup)

start = target_content.find(marker)
if start == -1:
    print(f"ERROR: 找不到补丁起始 marker: {marker}", file=sys.stderr)
    sys.exit(1)

end = target_content.find(end_marker, start + len(marker))
if end == -1:
    print(f"ERROR: 找不到补丁结束 marker: {end_marker}", file=sys.stderr)
    sys.exit(1)
end += len(end_marker)

try:
    backup_tree = ast.parse(backup_content, filename=backup)
except SyntaxError as exc:
    print(f"ERROR: 备份文件语法错误: {exc}", file=sys.stderr)
    sys.exit(1)

nodes = [
    node for node in ast.walk(backup_tree)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name == function_name
]
if len(nodes) != 1:
    print(
        f"ERROR: 备份中找到 {len(nodes)} 个 {function_name} 函数，期望 1 个",
        file=sys.stderr,
    )
    sys.exit(1)

node = nodes[0]
backup_lines = backup_content.splitlines(keepends=True)
func_text = "".join(backup_lines[node.lineno - 1:node.end_lineno])

prefix = target_content[:start]
suffix = target_content[end:]
separator = ""
if prefix and not prefix.endswith("\n"):
    separator += "\n"
replacement = separator + func_text
if replacement and not replacement.endswith("\n"):
    replacement += "\n"
if suffix and not suffix.startswith("\n"):
    replacement += "\n"

new_content = prefix + replacement + suffix

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
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        return $rc
    fi

    rm -f "$backup"
    log_info "已还原函数 ${function_name}: $file"
}
