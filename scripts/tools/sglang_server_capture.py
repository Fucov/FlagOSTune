#!/usr/bin/env python3
"""Capture one complete measured SGLang inference window with Nsight Systems."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple


class CaptureError(RuntimeError):
    """A capture lifecycle requirement was not satisfied."""


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def profile_request_body() -> dict:
    # Omitting num_steps keeps profiling active until /stop_profile. This is
    # required to cover the measured request's prefill and decode in one trace.
    return {"activities": ["CUDA_PROFILER"]}


def _default_health_request(url: str, timeout: float) -> int:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0


def wait_ready(
    base_url: str,
    timeout: float,
    child_alive: Callable[[], None],
    *,
    request: Callable[[str, float], int] = _default_health_request,
    poll_interval: float = 0.5,
) -> str:
    endpoints = ("/health_generate", "/health", "/v1/models")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        child_alive()
        for endpoint in endpoints:
            status = request(base_url.rstrip("/") + endpoint, min(2.0, timeout))
            if 200 <= status < 300:
                return endpoint
        time.sleep(poll_interval)
    raise CaptureError(
        "server readiness timeout after "
        f"{timeout:g}s; tried {', '.join(endpoints)}"
    )


def http_json(
    method: str, url: str, body: Optional[Mapping[str, object]], timeout: float
) -> Tuple[int, object]:
    encoded = None
    headers = {"Accept": "application/json"}
    if body is not None:
        encoded = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=encoded, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                value = json.loads(raw) if raw else {}
            except ValueError:
                value = raw
            return int(response.status), value
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            value = json.loads(raw) if raw else {}
        except ValueError:
            value = raw
        return int(exc.code), value
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise CaptureError(f"HTTP request failed: {method} {url}: {exc}") from exc


def start_profile(base_url: str, timeout: float = 10.0) -> dict:
    body = profile_request_body()
    status, response = http_json(
        "POST", base_url.rstrip("/") + "/start_profile", body, timeout
    )
    if not 200 <= status < 300:
        raise CaptureError(
            f"/start_profile returned HTTP {status}: {response!r}; body={body!r}"
        )
    if isinstance(response, Mapping) and response.get("error"):
        raise CaptureError(f"/start_profile rejected request: {response!r}")
    return {"request": body, "status": status, "response": response}


def stop_profile(base_url: str, timeout: float = 10.0) -> dict:
    status, response = http_json(
        "POST", base_url.rstrip("/") + "/stop_profile", None, timeout
    )
    if not 200 <= status < 300:
        raise CaptureError(
            f"/stop_profile returned HTTP {status}: {response!r}"
        )
    if isinstance(response, Mapping) and response.get("error"):
        raise CaptureError(f"/stop_profile rejected request: {response!r}")
    return {"status": status, "response": response}


def terminate_process_group(
    process: subprocess.Popen, grace_seconds: float = 10.0
) -> None:
    if process.poll() is not None:
        return
    for sig, wait_seconds in (
        (signal.SIGINT, grace_seconds),
        (signal.SIGTERM, max(1.0, grace_seconds / 2)),
        (signal.SIGKILL, 1.0),
    ):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=wait_seconds)
            return
        except subprocess.TimeoutExpired:
            continue


GROUP_MARKERS = {
    "--nsys-command": "nsys",
    "--warmup-command": "warmup",
    "--benchmark-command": "benchmark",
}


def parse_command_groups(argv: Sequence[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    prefix: List[str] = []
    groups: Dict[str, List[str]] = {"nsys": [], "warmup": [], "benchmark": []}
    current: Optional[str] = None
    for token in argv:
        marker = GROUP_MARKERS.get(token)
        if marker is not None:
            current = marker
            continue
        if current is None:
            prefix.append(token)
        else:
            groups[current].append(token)
    missing = [name for name, command in groups.items() if not command]
    if missing:
        raise CaptureError("missing command group(s): " + ", ".join(missing))
    return prefix, groups


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_identity(root: Path) -> Tuple[Optional[str], bool]:
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    return (commit.stdout.strip() or None, bool(dirty.stdout.strip()))


def _nsys_version(command: Sequence[str]) -> Optional[str]:
    executable = command[0] if command else "nsys"
    result = subprocess.run(
        [executable, "--version"], capture_output=True, text=True
    )
    text = (result.stdout + "\n" + result.stderr).strip()
    return text or None


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _run_logged(command: Sequence[str], log_path: Path, label: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{label}] {json.dumps(list(command), ensure_ascii=False)}\n")
        handle.flush()
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise CaptureError(f"{label} command failed with exit {result.returncode}")


def _child_alive(process: subprocess.Popen, label: str) -> None:
    code = process.poll()
    if code is not None:
        raise CaptureError(f"{label} exited unexpectedly with code {code}")


def _wait_benchmark(
    benchmark: subprocess.Popen,
    nsys_process: subprocess.Popen,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        code = benchmark.poll()
        if code is not None:
            if code != 0:
                raise CaptureError(f"benchmark command failed with exit {code}")
            return
        _child_alive(nsys_process, "nsys/server")
        time.sleep(0.2)
    raise CaptureError(f"benchmark timeout after {timeout:g}s")


def _parse_throughput(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = (
        r"(?:request|throughput)[^0-9\n]*([0-9]+(?:\.[0-9]+)?)\s*(?:req/s|requests/s)",
        r"([0-9]+(?:\.[0-9]+)?)\s*requests/s",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def run_capture(args: argparse.Namespace, commands: Mapping[str, Sequence[str]]) -> dict:
    report = Path(args.report).absolute()
    metadata_path = Path(args.metadata).absolute()
    server_log = Path(args.server_log).absolute()
    nsys_log = Path(args.nsys_log).absolute()
    benchmark_log = Path(args.benchmark_log).absolute()
    for path in (server_log, nsys_log, benchmark_log, report):
        path.parent.mkdir(parents=True, exist_ok=True)
    for path in (server_log, nsys_log, benchmark_log):
        path.write_text("", encoding="utf-8")

    nsys_process: Optional[subprocess.Popen] = None
    benchmark_process: Optional[subprocess.Popen] = None
    ready_endpoint: Optional[str] = None
    profile_exchange: Optional[dict] = None
    stop_exchange: Optional[dict] = None
    capture_start_iso: Optional[str] = None
    capture_start_monotonic: Optional[float] = None
    capture_end_iso: Optional[str] = None
    capture_end_monotonic: Optional[float] = None
    benchmark_start_iso: Optional[str] = None
    benchmark_start_monotonic: Optional[float] = None
    benchmark_end_iso: Optional[str] = None
    try:
        with nsys_log.open("a", encoding="utf-8") as nsys_handle:
            nsys_handle.write(
                "[nsys] " + json.dumps(list(commands["nsys"]), ensure_ascii=False) + "\n"
            )
            nsys_handle.flush()
            nsys_process = subprocess.Popen(
                commands["nsys"],
                stdout=nsys_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            alive = lambda: _child_alive(nsys_process, "nsys/server")
            ready_endpoint = wait_ready(
                args.base_url, args.profile_ready_timeout, alive
            )

            _run_logged(commands["warmup"], benchmark_log, "warmup")

            capture_start_iso = now_iso()
            capture_start_monotonic = time.monotonic()
            profile_exchange = start_profile(args.base_url)

            benchmark_start_iso = now_iso()
            benchmark_start_monotonic = time.monotonic()
            with benchmark_log.open("a", encoding="utf-8") as benchmark_handle:
                benchmark_handle.write(
                    "[benchmark] "
                    + json.dumps(list(commands["benchmark"]), ensure_ascii=False)
                    + "\n"
                )
                benchmark_handle.flush()
                benchmark_process = subprocess.Popen(
                    commands["benchmark"],
                    stdout=benchmark_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                _wait_benchmark(
                    benchmark_process,
                    nsys_process,
                    max(args.profile_ready_timeout, 60.0) * 4,
                )
            benchmark_end_iso = now_iso()
            benchmark_end_monotonic = time.monotonic()
            benchmark_process = None
            stop_exchange = stop_profile(args.base_url)
            capture_end_iso = now_iso()
            capture_end_monotonic = time.monotonic()
            terminate_process_group(nsys_process, grace_seconds=15.0)
            nsys_process = None

        if not report.is_file() or report.stat().st_size <= 0:
            raise CaptureError(f"Nsight report is missing or empty: {report}")

        project_root = Path(args.project_root).absolute()
        workflow_script = Path(args.workflow_script).absolute()
        parser_script = Path(args.parser_script).absolute()
        git_commit, git_dirty = _git_identity(project_root)
        metadata = {
            "capture_status": "PASS",
            "input_report": str(report),
            "report_size": report.stat().st_size,
            "capture_mode": "server-full",
            "capture_scope": "measured_inference",
            "inference_scope": "prefill_and_decode",
            "steady_state_guaranteed": True,
            "profile_request": profile_exchange,
            "profile_stop_response": stop_exchange,
            "readiness_endpoint": ready_endpoint,
            "capture_start_wall_time": capture_start_iso,
            "capture_end_wall_time": capture_end_iso,
            "capture_duration_seconds": (
                capture_end_monotonic - capture_start_monotonic
                if capture_start_monotonic is not None
                else None
            ),
            "benchmark_start_wall_time": benchmark_start_iso,
            "benchmark_end_wall_time": benchmark_end_iso,
            "benchmark_duration_seconds": (
                benchmark_end_monotonic - benchmark_start_monotonic
                if benchmark_start_monotonic is not None
                else None
            ),
            "model": args.model,
            "model_path": args.model_path,
            "tokenizer_path": args.tokenizer_path,
            "scenario": args.scenario,
            "workload": args.dataset,
            "num_prompts": args.num_prompts,
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "concurrency": args.profile_concurrency,
            "tp_size": args.tp_size,
            "benchmark_throughput": _parse_throughput(benchmark_log),
            "cuda_graph_enabled": args.cuda_graph_enabled,
            "cuda_graph_trace": args.cuda_graph_trace,
            "layerwise_nvtx_enabled": args.layerwise_nvtx_enabled,
            "deepgemm_jit_detected": "deepgemm" in server_log.read_text(
                encoding="utf-8", errors="replace"
            ).lower() and "jit" in server_log.read_text(
                encoding="utf-8", errors="replace"
            ).lower(),
            "moe_config_fallback_detected": "fallback" in server_log.read_text(
                encoding="utf-8", errors="replace"
            ).lower(),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "workflow_sha256": _sha256(workflow_script),
            "parser_sha256": _sha256(parser_script),
            "nsys_version": _nsys_version(commands["nsys"]),
            "server_log": str(server_log),
            "nsys_log": str(nsys_log),
            "benchmark_log": str(benchmark_log),
            "nsys_command": list(commands["nsys"]),
            "warmup_command": list(commands["warmup"]),
            "benchmark_command": list(commands["benchmark"]),
            "generated_time": now_iso(),
        }
        _atomic_json(metadata_path, metadata)
        return metadata
    finally:
        if benchmark_process is not None:
            terminate_process_group(benchmark_process, grace_seconds=2.0)
        if profile_exchange is not None and stop_exchange is None and nsys_process is not None:
            try:
                stop_profile(args.base_url)
            except CaptureError:
                pass
        if nsys_process is not None:
            terminate_process_group(nsys_process, grace_seconds=5.0)


def _bool(value: str) -> bool:
    lowered = value.lower()
    if lowered not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected true or false")
    return lowered == "true"


def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sglang_server_capture.py run")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--nsys-log", required=True)
    parser.add_argument("--benchmark-log", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--profile-warmup-prompts", type=int, required=True)
    parser.add_argument("--profile-concurrency", type=int, required=True)
    parser.add_argument("--profile-ready-timeout", type=float, required=True)
    parser.add_argument("--cuda-graph-enabled", type=_bool, required=True)
    parser.add_argument("--cuda-graph-trace", choices=("graph", "node", "none"), required=True)
    parser.add_argument("--layerwise-nvtx-enabled", type=_bool, required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--workflow-script", required=True)
    parser.add_argument("--parser-script", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--tp-size", type=int, required=True)
    return parser


def exec_server(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="sglang_server_capture.py exec-server")
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("server command is required after --")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        str(args.log), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644
    )
    os.dup2(descriptor, sys.stdout.fileno())
    os.dup2(descriptor, sys.stderr.fileno())
    if descriptor not in (sys.stdout.fileno(), sys.stderr.fileno()):
        os.close(descriptor)
    os.execvp(command[0], command)
    return 127


def main(argv: Optional[Sequence[str]] = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        print("expected run or exec-server", file=sys.stderr)
        return 2
    mode = values.pop(0)
    try:
        if mode == "exec-server":
            return exec_server(values)
        if mode != "run":
            raise CaptureError(f"unknown mode: {mode}")
        prefix, commands = parse_command_groups(values)
        args = build_run_parser().parse_args(prefix)
        metadata = run_capture(args, commands)
        print(json.dumps(metadata, ensure_ascii=False))
        return 0
    except (CaptureError, OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
