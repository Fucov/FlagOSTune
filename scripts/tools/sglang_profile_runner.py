#!/usr/bin/env python3
"""SGLang Torch profiler runner.

This runner mirrors the existing vLLM torch profiling flow, but it uses
SGLang offline throughput profiling and writes one native SGLang report side.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml


DEFAULT_SCENARIO = {
    "name": "p4096d1024",
    "input_len": 4096,
    "output_len": 1024,
    "concurrency": 16,
}


@dataclass(frozen=True)
class RunPaths:
    log_dir: Path
    profile_dir: Path


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Path | None = None) -> Dict[str, Any]:
    cfg = config_path or (get_project_root() / "scripts" / "tools" / "sglang_tool_config.yaml")
    if not cfg.exists():
        raise SystemExit(f"[ERROR] 配置文件不存在: {cfg}")
    data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"[ERROR] 配置文件格式非法: {cfg}")
    return data


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else get_project_root() / path


def get_scenarios(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    current_run = config.get("current_run", {}) or {}
    scenario_type = current_run.get("scenario_type") or "optimized"
    scenarios = (
        (config.get("benchmark", {}) or {})
        .get("scenarios", {})
        .get(scenario_type, [])
    )
    if not scenarios:
        return [dict(DEFAULT_SCENARIO)]
    return [dict(item) for item in scenarios]


def resolve_run_paths(config: Dict[str, Any]) -> RunPaths:
    paths = config.get("paths", {}) or {}
    log_dir = resolve_path(paths.get("log_dir", "results/sglang-bench-log/sglang_bench_logs"))
    torch_output_dir = resolve_path(paths.get("torch_output_dir", "results/sglang-torch-raw"))
    return RunPaths(log_dir=log_dir, profile_dir=torch_output_dir / "report-sglang")


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def first_non_empty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text != "null":
            return text
    return ""


def add_option(cmd: List[str], name: str, value: Any) -> None:
    if value is None:
        return
    text = str(value)
    if text == "" or text == "null":
        return
    cmd.extend([name, text])

def validate_extra_args(extra_args: str) -> List[str]:
    """Parse extra args and reject duplicated core options.

    extra_args 只允许放低频 SGLang 参数，不允许覆盖 runner 已经生成的核心参数。
    """
    if not extra_args:
        return []

    tokens = shlex.split(extra_args)

    forbidden_prefixes = {
        "--model-path",
        "--tokenizer-path",
        "--dataset-name",
        "--dataset-path",
        "--random-input-len",
        "--random-output-len",
        "--num-prompts",
        "--tp-size",
        "--tensor-parallel-size",
        "--trust-remote-code",
        "--mem-fraction-static",
        "--dtype",
        "--quantization",
        "--load-format",
        "--profile",
        "--language-only",
    }

    bad = []
    for token in tokens:
        key = token.split("=", 1)[0]
        if key in forbidden_prefixes:
            bad.append(token)

    if bad:
        raise SystemExit(
            "[ERROR] sglang.extra_args / serve.extra_args 中包含核心参数，"
            "请改到 config 的一级字段里，不要通过 extra_args 覆盖。\n"
            f"非法参数: {bad}"
        )

    return tokens

def build_sglang_command(
    scenario: Dict[str, Any],
    config: Dict[str, Any],
    profile_dir: Path,
    profile: bool,
) -> List[str]:
    model = config.get("model", {}) or {}
    serve = config.get("serve", {}) or {}
    sglang = config.get("sglang", {}) or {}
    bench_cfg = config.get("benchmark", {}) or {}

    model_path = first_non_empty(model.get("path"))
    if not model_path:
        raise SystemExit("[ERROR] model.path 不能为空")

    tokenizer_path = first_non_empty(model.get("tokenizer_path"), model_path)

    input_len = int(scenario.get("input_len", DEFAULT_SCENARIO["input_len"]))
    output_len = int(scenario.get("output_len", DEFAULT_SCENARIO["output_len"]))

    # offline throughput 没有“并发”概念，这里把 concurrency 映射成 num-prompts。
    # 也允许场景里显式写 num_prompts 覆盖。
    num_prompts = int(
        scenario.get(
            "num_prompts",
            scenario.get("concurrency", DEFAULT_SCENARIO["concurrency"]),
        )
    )

    tp_size = int(model.get("tensor_parallel_size", 1) or 1)

    dataset_name = str(bench_cfg.get("dataset_name", "random")).strip()
    valid_dataset_names = {"sharegpt", "random", "generated-shared-prefix"}
    if dataset_name not in valid_dataset_names:
        raise SystemExit(
            f"[ERROR] bench_offline_throughput 不支持 dataset_name={dataset_name!r}。\n"
            f"合法值: {sorted(valid_dataset_names)}\n"
            "注意：当前 offline throughput 不支持 random-ids。"
        )

    dataset_path = first_non_empty(bench_cfg.get("dataset_path"))

    # 在无外网服务器上，random 如果不指定 dataset_path，会尝试下载 HF ShareGPT。
    allow_hf_download = bool_value(bench_cfg.get("allow_hf_download", False))
    if dataset_name == "random" and not dataset_path and not allow_hf_download:
        raise SystemExit(
            "[ERROR] dataset_name=random 但没有配置 benchmark.dataset_path。\n"
            "SGLang bench_offline_throughput 的 random 会尝试下载 HuggingFace ShareGPT，"
            "无外网环境会失败。\n"
            "请先生成本地数据集，并在 config 中添加：\n"
            "benchmark:\n"
            "  dataset_name: random\n"
            "  dataset_path: /data/yangkw/datasets/sharegpt_synthetic_2048.json"
        )

    python_exec = os.environ.get("Python_EXECUTABLE", sys.executable or "python3")

    cmd = [
        python_exec,
        "-m",
        "sglang.bench_offline_throughput",
        "--model-path",
        model_path,
        "--tokenizer-path",
        tokenizer_path,
        "--dataset-name",
        dataset_name,
    ]

    if dataset_path:
        cmd.extend(["--dataset-path", dataset_path])

    if dataset_name == "random":
        cmd.extend(
            [
                "--random-input-len",
                str(input_len),
                "--random-output-len",
                str(output_len),
            ]
        )
    elif dataset_name == "sharegpt":
        # sharegpt 可以不指定 input_len，但 output_len 可控。
        cmd.extend(["--sharegpt-output-len", str(output_len)])
        sharegpt_context_len = bench_cfg.get("sharegpt_context_len")
        if sharegpt_context_len is not None:
            add_option(cmd, "--sharegpt-context-len", sharegpt_context_len)
    elif dataset_name == "generated-shared-prefix":
        # generated-shared-prefix 使用自己的参数体系。
        cmd.extend(
            [
                "--gsp-num-groups",
                str(scenario.get("gsp_num_groups", bench_cfg.get("gsp_num_groups", 1))),
                "--gsp-prompts-per-group",
                str(
                    scenario.get(
                        "gsp_prompts_per_group",
                        bench_cfg.get("gsp_prompts_per_group", max(1, num_prompts)),
                    )
                ),
                "--gsp-system-prompt-len",
                str(
                    scenario.get(
                        "gsp_system_prompt_len",
                        bench_cfg.get("gsp_system_prompt_len", input_len),
                    )
                ),
                "--gsp-question-len",
                str(
                    scenario.get(
                        "gsp_question_len",
                        bench_cfg.get("gsp_question_len", 16),
                    )
                ),
                "--gsp-output-len",
                str(
                    scenario.get(
                        "gsp_output_len",
                        bench_cfg.get("gsp_output_len", output_len),
                    )
                ),
            ]
        )

    cmd.extend(
        [
            "--num-prompts",
            str(num_prompts),
            "--tp-size",
            str(tp_size),
        ]
    )

    if bool_value(serve.get("trust_remote_code", False)):
        cmd.append("--trust-remote-code")

    add_option(cmd, "--mem-fraction-static", sglang.get("mem_fraction_static"))
    add_option(cmd, "--context-length", sglang.get("context_length"))
    add_option(cmd, "--dtype", sglang.get("dtype"))
    add_option(cmd, "--quantization", sglang.get("quantization"))
    add_option(cmd, "--load-format", sglang.get("load_format"))

    extra_args = first_non_empty(sglang.get("extra_args"), serve.get("extra_args"))
    cmd.extend(validate_extra_args(extra_args))

    if profile:
        cmd.append("--profile")
        profile_dir.mkdir(parents=True, exist_ok=True)

    return cmd


def run_one_command(
    cmd: List[str],
    log_file: Path,
    profile_dir: Path,
    profile: bool,
    config: Dict[str, Any],
) -> int:
    env = os.environ.copy()

    current_run = config.get("current_run", {}) or {}
    device = str(current_run.get("device", "")).strip()
    if device:
        env["CUDA_VISIBLE_DEVICES"] = device

    # 无外网服务器默认离线，避免 transformers / datasets / HF hub 隐式联网。
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    if profile:
        env["SGLANG_TORCH_PROFILER_DIR"] = str(profile_dir)

    log_file.parent.mkdir(parents=True, exist_ok=True)

    with log_file.open("w", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Command: {shlex.join(cmd)}\n")
        f.write(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '')}\n")
        f.write(f"HF_HUB_OFFLINE={env.get('HF_HUB_OFFLINE', '')}\n")
        f.write(f"HF_DATASETS_OFFLINE={env.get('HF_DATASETS_OFFLINE', '')}\n")
        f.write(f"TRANSFORMERS_OFFLINE={env.get('TRANSFORMERS_OFFLINE', '')}\n")
        if profile:
            f.write(f"SGLANG_TORCH_PROFILER_DIR={profile_dir}\n")
        f.flush()

        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        return int(result.returncode)


def run_profile(config: Dict[str, Any], dry_run: bool = False) -> int:
    paths = resolve_run_paths(config)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    paths.profile_dir.mkdir(parents=True, exist_ok=True)

    benchmark = config.get("benchmark", {}) or {}
    num_runs = int(benchmark.get("num_runs", 2) or 2)
    if num_runs <= 0:
        raise SystemExit("[ERROR] benchmark.num_runs 必须 >= 1")

    scenarios = get_scenarios(config)
    exit_code = 0

    for scenario in scenarios:
        name = str(scenario.get("name", "unknown"))
        for run_id in range(1, num_runs + 1):
            profile = run_id == num_runs
            cmd = build_sglang_command(
                scenario=scenario,
                config=config,
                profile_dir=paths.profile_dir,
                profile=profile,
            )
            log_file = paths.log_dir / f"{name}_run{run_id}.log"

            print(f"[INFO] SGLang scenario={name} run={run_id}/{num_runs} profile={profile}")
            print(f"[INFO] Log: {log_file}")
            print(f"[INFO] Command: {shlex.join(cmd)}")

            if dry_run:
                continue

            rc = run_one_command(
                cmd=cmd,
                log_file=log_file,
                profile_dir=paths.profile_dir,
                profile=profile,
                config=config,
            )
            if rc != 0:
                print(
                    f"[ERROR] Command failed with exit code {rc}: {log_file}",
                    file=sys.stderr,
                )
                exit_code = rc
                return exit_code

    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SGLang offline Torch profiler scenarios")
    parser.add_argument("--config", type=str, default=None, help="sglang_tool_config.yaml path")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing SGLang")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(Path(args.config) if args.config else None)
    return run_profile(config, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
