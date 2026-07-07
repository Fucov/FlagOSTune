#!/usr/bin/env python3
"""Patch SGLang Qwen3 VL vision_config dict compatibility.

Some Qwen3.5 multimodal configs provide ``vision_config`` as a plain dict while
SGLang's Qwen3 VL vision model reads it through attribute access.  This patch
keeps the normal multimodal path and converts only that config object at the
Qwen3VLMoeVisionModel boundary.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple


START_MARKER = "# FLAGOSTUNE_QWEN3_VISION_CONFIG_PATCH_BEGIN"
END_MARKER = "# FLAGOSTUNE_QWEN3_VISION_CONFIG_PATCH_END"
CONVERSION_MARKER = "# FLAGOSTUNE_QWEN3_VISION_CONFIG_PATCH_APPLY"
BACKUP_SUFFIX = ".flagostune_qwen3_vision_config.bak"

HELPER_BLOCK = f'''{START_MARKER}
class _FlagOSTuneQwen3VisionAttrConfig:
    def __init__(self, values):
        for key, value in values.items():
            setattr(self, key, _flagostune_qwen3_vision_config_to_attr_config(value))


def _flagostune_qwen3_vision_config_to_attr_config(value):
    if isinstance(value, dict):
        return _FlagOSTuneQwen3VisionAttrConfig(value)
    if isinstance(value, list):
        return [_flagostune_qwen3_vision_config_to_attr_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_flagostune_qwen3_vision_config_to_attr_config(item) for item in value)
    return value
{END_MARKER}

'''


def locate_qwen3_vl_path() -> Path:
    spec = importlib.util.find_spec("sglang.srt.models.qwen3_vl")
    if spec is None or spec.origin is None:
        raise SystemExit("[ERROR] 找不到 sglang.srt.models.qwen3_vl，请确认当前 Python 环境已安装 SGLang")
    return Path(spec.origin)


def _find_target_nodes(tree: ast.Module) -> Tuple[ast.ClassDef, ast.FunctionDef]:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3VLMoeVisionModel":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return node, item
            raise ValueError("Qwen3VLMoeVisionModel.__init__ not found")
    raise ValueError("Qwen3VLMoeVisionModel class not found")


def _line_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _first_runtime_statement_line(init_node: ast.FunctionDef) -> int:
    body = list(init_node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        return init_node.end_lineno or init_node.lineno
    return body[0].lineno


def patch_source(source: str) -> Tuple[str, bool]:
    if START_MARKER in source and CONVERSION_MARKER in source:
        return source, False

    tree = ast.parse(source)
    class_node, init_node = _find_target_nodes(tree)
    lines = source.splitlines(keepends=True)

    if START_MARKER not in source:
        class_index = class_node.lineno - 1
        lines.insert(class_index, HELPER_BLOCK)
        helper_line_count = HELPER_BLOCK.count("\n")
    else:
        helper_line_count = 0

    conversion_line_no = _first_runtime_statement_line(init_node) + helper_line_count
    insertion_index = conversion_line_no - 1
    indent = _line_indent(lines[insertion_index])
    conversion_line = (
        f"{indent}vision_config = "
        f"_flagostune_qwen3_vision_config_to_attr_config(vision_config)  "
        f"{CONVERSION_MARKER}\n"
    )
    lines.insert(insertion_index, conversion_line)
    return "".join(lines), True


def restore_source(source: str) -> Tuple[str, bool]:
    changed = False
    if START_MARKER in source:
        start = source.find(START_MARKER)
        end = source.find(END_MARKER, start)
        if end == -1:
            raise ValueError("patch start marker exists but end marker is missing")
        end += len(END_MARKER)
        while end < len(source) and source[end] in "\r\n":
            end += 1
        source = source[:start] + source[end:]
        changed = True

    lines = []
    for line in source.splitlines(keepends=True):
        if CONVERSION_MARKER in line:
            changed = True
            continue
        lines.append(line)
    return "".join(lines), changed


def apply_patch(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    patched, changed = patch_source(source)
    if not changed:
        print(f"[INFO] SGLang Qwen3 vision_config patch already applied: {path}")
        return False

    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"[INFO] Backup: {backup}")
    path.write_text(patched, encoding="utf-8")
    print(f"[INFO] Applied SGLang Qwen3 vision_config patch: {path}")
    return True


def restore_patch(path: Path) -> bool:
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if backup.exists():
        shutil.copy2(backup, path)
        backup.unlink()
        print(f"[INFO] Restored from backup: {path}")
        return True

    source = path.read_text(encoding="utf-8")
    restored, changed = restore_source(source)
    if changed:
        path.write_text(restored, encoding="utf-8")
        print(f"[INFO] Removed SGLang Qwen3 vision_config patch markers: {path}")
    else:
        print(f"[INFO] SGLang Qwen3 vision_config patch is not applied: {path}")
    return changed


def patch_status(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    applied = START_MARKER in source and CONVERSION_MARKER in source
    print(f"[INFO] SGLang Qwen3 vision_config patch status: {'applied' if applied else 'not_applied'}")
    print(f"[INFO] Target: {path}")
    return applied


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch SGLang Qwen3 VL vision_config dict compatibility")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--apply", action="store_true", help="apply the compatibility patch")
    action.add_argument("--restore", action="store_true", help="restore the patched file")
    action.add_argument("--status", action="store_true", help="show patch status")
    parser.add_argument("--target", type=Path, default=None, help="explicit qwen3_vl.py path")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    target = args.target or locate_qwen3_vl_path()
    if not target.exists():
        raise SystemExit(f"[ERROR] 目标文件不存在: {target}")

    if args.apply:
        apply_patch(target)
    elif args.restore:
        restore_patch(target)
    else:
        patch_status(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
