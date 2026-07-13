#!/usr/bin/env python3
"""Collect SGLang profiling run metadata with per-field source tags."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def field(value: Any, source: str) -> Dict[str, Any]:
    return {"value": value, "source": source}


def run_text(cmd: Iterable[str], cwd: Optional[Path] = None) -> str:
    try:
        return subprocess.check_output(list(cmd), cwd=str(cwd) if cwd else None, text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        return ""


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        return parse_simple_yaml(path.read_text(encoding="utf-8"))
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def scalar(value: str) -> Any:
    text = value.strip().strip('"').strip("'")
    if text in {"true", "false"}:
        return text == "true"
    if text in {"null", "None", ""}:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Tiny fallback for local tests when PyYAML is absent; not a general YAML parser."""
    data: Dict[str, Any] = {}
    current: Optional[str] = None
    nested: Optional[str] = None
    list_key: Optional[str] = None
    list_item: Optional[Dict[str, Any]] = None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if indent == 0 and line.endswith(":"):
            current = line[:-1]
            data[current] = {}
            nested = None
            list_key = None
            continue
        if current is None:
            continue
        if indent == 2 and line.endswith(":"):
            nested = line[:-1]
            data[current][nested] = {}
            continue
        if indent == 4 and line.endswith(":") and isinstance(data[current].get(nested or ""), dict):
            list_key = line[:-1]
            data[current][nested][list_key] = []
            continue
        if line.startswith("- "):
            list_item = {}
            if nested and list_key:
                data[current][nested][list_key].append(list_item)
            rest = line[2:]
            if ":" in rest:
                key, value = rest.split(":", 1)
                list_item[key.strip()] = scalar(value)
            continue
        if list_item is not None and indent >= 8 and ":" in line:
            key, value = line.split(":", 1)
            list_item[key.strip()] = scalar(value)
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            if nested:
                data[current][nested][key.strip()] = scalar(value)
            else:
                data[current][key.strip()] = scalar(value)
    return data


def first_scenario(config: Dict[str, Any]) -> Dict[str, Any]:
    scenarios = ((config.get("benchmark") or {}).get("scenarios") or {})
    for name in ("optimized", "full", "shape"):
        values = scenarios.get(name)
        if isinstance(values, list) and values:
            return dict(values[0] or {})
    for values in scenarios.values():
        if isinstance(values, list) and values:
            return dict(values[0] or {})
    return {}


def import_attr(module_name: str, attr: str = "__version__") -> Dict[str, Any]:
    try:
        module = __import__(module_name)
        value = getattr(module, attr, None)
        module_file = getattr(module, "__file__", None)
        if value is None and attr == "__version__":
            value = "installed"
        return {"version": value, "file": module_file}
    except Exception:
        return {"version": None, "file": None}


def collect_gpu_info() -> Dict[str, Any]:
    try:
        import torch  # type: ignore

        devices = []
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                free = used = None
                try:
                    free_bytes, total_free_context_bytes = torch.cuda.mem_get_info(idx)
                    free = int(free_bytes / 1024 / 1024)
                    used = int((props.total_memory - free_bytes) / 1024 / 1024)
                except Exception:
                    pass
                devices.append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_memory_mb": int(props.total_memory / 1024 / 1024),
                        "free_memory_mb": free,
                        "used_memory_mb": used,
                    }
                )
        return {
            "torch_cuda_device_count": field(torch.cuda.device_count() if torch.cuda.is_available() else 0, "torch.cuda"),
            "visible_gpus": field(devices, "torch.cuda"),
        }
    except Exception:
        return {
            "torch_cuda_device_count": field(None, "torch_unavailable"),
            "visible_gpus": field([], "torch_unavailable"),
        }


def discover_trace_files(trace_dir: Optional[Path]) -> list[dict[str, Any]]:
    if not trace_dir or not trace_dir.exists():
        return []
    files = sorted(
        set(
            list(trace_dir.glob("*.pt.trace.json"))
            + list(trace_dir.glob("*.trace.json"))
            + list(trace_dir.glob("*.pt.trace.json.gz"))
            + list(trace_dir.glob("*.trace.json.gz"))
        )
    )
    out = []
    for path in files:
        stat = path.stat()
        out.append(
            {
                "path": str(path),
                "size_bytes": stat.st_size,
                "size_gb": round(stat.st_size / 1024 / 1024 / 1024, 6),
                "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return out


def parse_bench_metrics(log_paths: list[Path]) -> Dict[str, Any]:
    text = ""
    for path in log_paths:
        try:
            text += "\n" + path.read_text(encoding="utf-8", errors="ignore")[-200000:]
        except OSError:
            pass

    def match_float(pattern: str) -> Any:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return float(match.group(1)) if match else None

    return {
        "throughput": field(match_float(r"(?:throughput|request throughput).*?([0-9]+(?:\.[0-9]+)?)"), "bench_log" if text else "not_found_in_bench_log"),
        "output_throughput": field(match_float(r"output.*?throughput.*?([0-9]+(?:\.[0-9]+)?)"), "bench_log" if text else "not_found_in_bench_log"),
        "mean_latency": field(match_float(r"mean.*?latency.*?([0-9]+(?:\.[0-9]+)?)"), "bench_log" if text else "not_found_in_bench_log"),
        "ttft": field(match_float(r"TTFT.*?([0-9]+(?:\.[0-9]+)?)"), "bench_log" if text else "not_found_in_bench_log"),
        "itl": field(match_float(r"ITL.*?([0-9]+(?:\.[0-9]+)?)"), "bench_log" if text else "not_found_in_bench_log"),
        "tpot": field(match_float(r"TPOT.*?([0-9]+(?:\.[0-9]+)?)"), "bench_log" if text else "not_found_in_bench_log"),
    }


def collect_metadata(
    *,
    model_name: str,
    config_path: Path,
    output_dir: Path,
    report_dir: Path,
    trace_dir: Optional[Path],
    selected_rank: str,
    phase: str,
    workflow_command: str = "",
    processing_command: str = "",
    nvidia_smi_text: str = "",
) -> Dict[str, Any]:
    config = load_config(config_path)
    model = config.get("model", {}) or {}
    runtime = config.get("runtime", {}) or {}
    sglang_cfg = config.get("sglang", {}) or {}
    bench_cfg = config.get("benchmark", {}) or {}
    scenario = first_scenario(config)
    root = Path.cwd()

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if nvidia_smi_text:
        (output_dir / f"nvidia_smi_{phase}.txt").write_text(nvidia_smi_text, encoding="utf-8")

    log_paths = []
    for base in (root / "results" / model_name).glob("**/*.log") if (root / "results" / model_name).exists() else []:
        log_paths.append(base)

    torch_info = import_attr("torch")
    sglang_info = import_attr("sglang")
    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "phase": phase,
        "environment": {
            "timestamp": field(datetime.now(timezone.utc).isoformat(), "collector"),
            "hostname": field(socket.gethostname(), "socket"),
            "user": field(getpass.getuser(), "getpass"),
            "pwd": field(str(root), "cwd"),
            "git_commit": field(run_text(["git", "rev-parse", "HEAD"], root) or None, "git"),
            "git_branch": field(run_text(["git", "branch", "--show-current"], root) or None, "git"),
            "git_status_short": field(run_text(["git", "status", "--short"], root), "git"),
            "python_executable": field(sys.executable, "sys"),
            "python_version": field(platform.python_version(), "platform"),
            "torch_version": field(torch_info["version"], "python_import:torch"),
            "torch_file": field(torch_info["file"], "python_import:torch"),
            "sglang_version": field(sglang_info["version"], "python_import:sglang"),
            "sglang_file": field(sglang_info["file"], "python_import:sglang"),
            "transformers_version": field(import_attr("transformers")["version"], "python_import:transformers"),
            "triton_version": field(import_attr("triton")["version"], "python_import:triton"),
            "flashinfer_version": field(import_attr("flashinfer")["version"], "python_import:flashinfer"),
            "deep_gemm_version": field(import_attr("deep_gemm")["version"], "python_import:deep_gemm"),
        },
        "gpu": {
            "cuda_visible_devices": field(os.environ.get("CUDA_VISIBLE_DEVICES"), "env"),
            **collect_gpu_info(),
        },
        "model": {
            "model_name": field(model.get("name", model_name), str(config_path)),
            "model_path": field(model.get("path"), str(config_path)),
            "tokenizer_path": field(model.get("tokenizer_path"), str(config_path)),
            "tp_size": field(model.get("tensor_parallel_size"), str(config_path)),
            "dtype": field(runtime.get("dtype", sglang_cfg.get("dtype")), str(config_path)),
            "context_length": field(sglang_cfg.get("context_length"), str(config_path)),
            "server_args": field(sglang_cfg.get("extra_args"), str(config_path)),
        },
        "benchmark": {
            "scenario_name": field(scenario.get("name"), str(config_path) if scenario else "not_found_in_config"),
            "input_len": field(scenario.get("input_len"), str(config_path) if scenario else "not_found_in_config"),
            "output_len": field(scenario.get("output_len"), str(config_path) if scenario else "not_found_in_config"),
            "concurrency": field(scenario.get("concurrency"), str(config_path) if scenario else "not_found_in_config"),
            "num_runs": field(bench_cfg.get("num_runs"), str(config_path)),
            "bench_log_paths": field([str(path) for path in log_paths], "results_logs"),
            **parse_bench_metrics(log_paths),
        },
        "trace": {
            "trace_dir": field(str(trace_dir) if trace_dir else None, "argument"),
            "trace_files": field(discover_trace_files(trace_dir), "trace_dir"),
            "selected_rank": field(selected_rank, "argument"),
            "workflow_command": field(workflow_command, "argument"),
            "processing_command": field(processing_command, "argument"),
            "profiler_config": {
                "detail": os.environ.get("SGLANG_TORCH_PROFILER_DETAIL", "light"),
                "light": os.environ.get("SGLANG_TORCH_PROFILER_LIGHT"),
                "with_stack": os.environ.get("SGLANG_TORCH_PROFILER_WITH_STACK"),
                "record_shapes": os.environ.get("SGLANG_TORCH_PROFILER_RECORD_SHAPES"),
                "profile_memory": os.environ.get("SGLANG_TORCH_PROFILER_PROFILE_MEMORY"),
                "with_modules": os.environ.get("SGLANG_TORCH_PROFILER_WITH_MODULES"),
            },
        },
    }

    payload = json.dumps(metadata, ensure_ascii=False, indent=2)
    (output_dir / "run_metadata.json").write_text(payload, encoding="utf-8")
    (report_dir / "run_metadata.json").write_text(payload, encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect SGLang run metadata")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--trace-dir", default="")
    parser.add_argument("--rank", default="0")
    parser.add_argument("--phase", default="snapshot")
    parser.add_argument("--workflow-command", default="")
    parser.add_argument("--processing-command", default="")
    parser.add_argument("--nvidia-smi-file", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    nvidia_smi_text = ""
    if args.nvidia_smi_file:
        smi_path = Path(args.nvidia_smi_file)
        if smi_path.exists():
            nvidia_smi_text = smi_path.read_text(encoding="utf-8", errors="ignore")
    collect_metadata(
        model_name=args.model,
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        report_dir=Path(args.report_dir),
        trace_dir=Path(args.trace_dir) if args.trace_dir else None,
        selected_rank=args.rank,
        phase=args.phase,
        workflow_command=args.workflow_command,
        processing_command=args.processing_command,
        nvidia_smi_text=nvidia_smi_text,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
