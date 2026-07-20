#!/usr/bin/env python3
"""Supervise phase-selective SGLang Nsight Systems server captures."""

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
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple


class CaptureError(RuntimeError):
    """A capture lifecycle requirement was not satisfied."""


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def endpoint_metadata(
    base_url: str, *, visible_devices: Optional[str] = None
) -> dict:
    parsed = urllib.parse.urlsplit(base_url)
    return {
        "base_url": base_url,
        "host": parsed.hostname,
        "port": parsed.port,
        "visible_devices": visible_devices,
    }


def detect_log_flags(text: str) -> dict:
    lowered = text.lower()
    return {
        "deepgemm_jit_detected": "deepgemm" in lowered and "jit" in lowered,
        "moe_config_fallback_detected": "moe" in lowered and "fallback" in lowered,
    }


def prepare_capture_outputs(log_paths: Sequence[Path], metadata_path: Path) -> None:
    """Clear owned logs and prevent stale PASS metadata from surviving a retry."""
    metadata_path.unlink(missing_ok=True)
    for path in log_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def profile_request_body(num_steps: int) -> dict:
    """Build the SGLang scheduler-step profiler request.

    Phase selection is performed by deciding when to send the request.  The
    endpoint therefore receives a relative start step of zero, while the
    requested scheduler-step count controls automatic profiler termination.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    return {
        "start_step": 0,
        "num_steps": num_steps,
        "activities": ["CUDA_PROFILER"],
    }


class DecodeDetector:
    """Require both a decode marker and a positive running-request count."""

    _decode = re.compile(r"\bDecode batch\b", re.IGNORECASE)
    _running = re.compile(
        r"running(?:[_\s]+(?:reqs?|request(?:\(s\)|s)?))?[^0-9]{0,20}([0-9]+)",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.decode_line: Optional[str] = None
        self.running_line: Optional[str] = None

    def feed(self, line: str) -> bool:
        if self._decode.search(line):
            self.decode_line = line.strip()
        match = self._running.search(line)
        if match and int(match.group(1)) > 0:
            self.running_line = line.strip()
        return self.decode_line is not None and self.running_line is not None

    @property
    def evidence(self) -> str:
        values = []
        for value in (self.decode_line, self.running_line):
            if value and value not in values:
                values.append(value)
        return " | ".join(values)


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


def wait_for_decode(
    log_path: Path,
    timeout: float,
    child_alive: Callable[[], None],
    *,
    poll_interval: float = 0.2,
    start_offset: int = 0,
) -> str:
    """Incrementally monitor the server log for stable decode evidence."""
    detector = DecodeDetector()
    deadline = time.monotonic() + timeout
    offset = start_offset
    remainder = ""
    while time.monotonic() < deadline:
        if log_path.is_file():
            with log_path.open(encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
            if chunk:
                combined = remainder + chunk
                lines = combined.splitlines(keepends=True)
                remainder = ""
                if lines and not lines[-1].endswith(("\n", "\r")):
                    remainder = lines.pop()
                for line in lines:
                    if detector.feed(line):
                        return detector.evidence
                if remainder and detector.feed(remainder):
                    return detector.evidence
        child_alive()
        time.sleep(poll_interval)
    raise CaptureError(
        f"decode evidence timeout after {timeout:g}s; no stable Decode batch "
        "with positive running requests was observed"
    )


def wait_for_profile_completion(
    log_path: Path,
    start_offset: int,
    timeout: float,
    child_alive: Callable[[], None],
    *,
    poll_interval: float = 0.2,
) -> str:
    """Wait for SGLang's automatic num_steps profiler stop evidence."""
    completed = re.compile(r"\bProfiling done\b", re.IGNORECASE)
    deadline = time.monotonic() + timeout
    offset = start_offset
    remainder = ""
    while time.monotonic() < deadline:
        if log_path.is_file():
            with log_path.open(encoding="utf-8", errors="replace") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
            if chunk:
                combined = remainder + chunk
                lines = combined.splitlines(keepends=True)
                remainder = ""
                if lines and not lines[-1].endswith(("\n", "\r")):
                    remainder = lines.pop()
                for line in lines:
                    if completed.search(line):
                        return line.strip()
                if remainder and completed.search(remainder):
                    return remainder.strip()
        child_alive()
        time.sleep(poll_interval)
    raise CaptureError(
        f"profile completion timeout after {timeout:g}s; SGLang did not log "
        "automatic 'Profiling done' evidence for the requested num_steps"
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


def start_profile(base_url: str, num_steps: int, timeout: float = 10.0) -> dict:
    body = profile_request_body(num_steps)
    status, response = http_json(
        "POST", base_url.rstrip("/") + "/start_profile", body, timeout
    )
    if not 200 <= status < 300:
        raise CaptureError(
            f"/start_profile returned HTTP {status}: {response!r}; body={body!r}"
        )
    if isinstance(response, Mapping) and (
        response.get("error") or response.get("success") is False
    ):
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
    if isinstance(response, Mapping) and (
        response.get("error") or response.get("success") is False
    ):
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
    prepare_capture_outputs((server_log, nsys_log, benchmark_log), metadata_path)

    nsys_process: Optional[subprocess.Popen] = None
    benchmark_process: Optional[subprocess.Popen] = None
    ready_endpoint: Optional[str] = None
    phase_evidence = ""
    profile_completion_evidence = ""
    profile_log_offset = 0
    profile_exchange: Optional[dict] = None
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

            if args.profile_phase == "startup":
                profile_log_offset = server_log.stat().st_size if server_log.is_file() else 0
                capture_start_iso = now_iso()
                capture_start_monotonic = time.monotonic()
                profile_exchange = start_profile(
                    args.base_url, args.profile_num_steps
                )
                phase_evidence = (
                    "profile started after HTTP readiness and before workload warmup; "
                    "this is post-load startup/warmup evidence, not model-load capture"
                )

            _run_logged(commands["warmup"], benchmark_log, "warmup")

            if args.profile_phase == "startup":
                profile_completion_evidence = wait_for_profile_completion(
                    server_log,
                    profile_log_offset,
                    args.profile_ready_timeout,
                    alive,
                )
                capture_end_iso = now_iso()
                capture_end_monotonic = time.monotonic()

            if args.profile_phase in ("prefill", "full"):
                profile_log_offset = server_log.stat().st_size if server_log.is_file() else 0
                capture_start_iso = now_iso()
                capture_start_monotonic = time.monotonic()
                profile_exchange = start_profile(
                    args.base_url, args.profile_num_steps
                )
                phase_evidence = (
                    "profile started before measured benchmark "
                    f"for requested {args.profile_phase} phase via {ready_endpoint}"
                )

            benchmark_start_iso = now_iso()
            benchmark_start_monotonic = time.monotonic()
            measured_log_offset = server_log.stat().st_size if server_log.is_file() else 0
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
                if args.profile_phase == "decode":
                    phase_evidence = wait_for_decode(
                        server_log,
                        args.profile_ready_timeout,
                        lambda: (
                            _child_alive(nsys_process, "nsys/server"),
                            _child_alive(benchmark_process, "benchmark"),
                        ),
                        start_offset=measured_log_offset,
                    )
                    profile_log_offset = server_log.stat().st_size if server_log.is_file() else 0
                    capture_start_iso = now_iso()
                    capture_start_monotonic = time.monotonic()
                    profile_exchange = start_profile(
                        args.base_url, args.profile_num_steps
                    )
                if args.profile_phase != "startup":
                    profile_completion_evidence = wait_for_profile_completion(
                        server_log,
                        profile_log_offset,
                        args.profile_ready_timeout,
                        lambda: (
                            _child_alive(nsys_process, "nsys/server"),
                            _child_alive(benchmark_process, "benchmark"),
                        ),
                    )
                    capture_end_iso = now_iso()
                    capture_end_monotonic = time.monotonic()
                _wait_benchmark(
                    benchmark_process,
                    nsys_process,
                    max(args.profile_ready_timeout, 60.0) * 4,
                )
            benchmark_end_iso = now_iso()
            benchmark_end_monotonic = time.monotonic()
            benchmark_process = None
            # num_steps makes SGLang call cudaProfilerStop automatically.  Do
            # not issue /stop_profile here: doing so would turn a selective
            # capture into a benchmark-duration capture and may race the
            # scheduler's automatic stop.
            terminate_process_group(nsys_process, grace_seconds=15.0)
            nsys_process = None

        if not report.is_file() or report.stat().st_size <= 0:
            raise CaptureError(f"Nsight report is missing or empty: {report}")

        project_root = Path(args.project_root).absolute()
        workflow_script = Path(args.workflow_script).absolute()
        parser_script = Path(args.parser_script).absolute()
        git_commit, git_dirty = _git_identity(project_root)
        server_log_text = server_log.read_text(encoding="utf-8", errors="replace")
        metadata = {
            "capture_status": "PASS",
            "input_report": str(report),
            "report_size": report.stat().st_size,
            "capture_mode": "server-steps",
            "capture_scope": "scheduler_steps",
            "inference_scope": "phase_selective_scheduler_steps",
            "steady_state_guaranteed": args.profile_phase in ("prefill", "decode"),
            "requested_phase": args.profile_phase,
            "profile_phase": args.profile_phase,
            "detected_phase": args.profile_phase.upper(),
            "phase_confidence": "HIGH" if args.profile_phase == "decode" else "MEDIUM",
            "phase_evidence": phase_evidence,
            "profile_completion_evidence": profile_completion_evidence,
            "capture_end_source": "server_log_automatic_profile_stop",
            "profile_start_step": args.profile_start_step,
            "effective_profile_start_step": 0,
            "profile_num_steps": args.profile_num_steps,
            "profile_warmup_prompts": args.profile_warmup_prompts,
            "profile_ready_timeout_seconds": args.profile_ready_timeout,
            "profile_request": profile_exchange,
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
            "benchmark_status": "PASS",
            "cuda_graph_enabled": args.cuda_graph_enabled,
            "cuda_graph_trace": args.cuda_graph_trace,
            "layerwise_nvtx_enabled": args.layerwise_nvtx_enabled,
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
        metadata.update(detect_log_flags(server_log_text))
        metadata.update(
            endpoint_metadata(
                args.base_url,
                visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            )
        )
        _atomic_json(metadata_path, metadata)
        return metadata
    finally:
        if benchmark_process is not None:
            terminate_process_group(benchmark_process, grace_seconds=2.0)
        if nsys_process is not None:
            terminate_process_group(nsys_process, grace_seconds=5.0)


def _bool(value: str) -> bool:
    lowered = value.lower()
    if lowered not in ("true", "false"):
        raise argparse.ArgumentTypeError("expected true or false")
    return lowered == "true"


def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sglang_server_steps.py run")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--server-log", required=True)
    parser.add_argument("--nsys-log", required=True)
    parser.add_argument("--benchmark-log", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--profile-phase", choices=("startup", "prefill", "decode", "full"), required=True)
    parser.add_argument("--profile-start-step", type=int, default=0)
    parser.add_argument("--profile-num-steps", type=int, required=True)
    parser.add_argument("--profile-warmup-prompts", type=int, required=True)
    parser.add_argument("--profile-concurrency", type=int, required=True)
    parser.add_argument("--profile-ready-timeout", type=float, required=True)
    parser.add_argument("--decode-log-pattern")
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
    parser = argparse.ArgumentParser(prog="sglang_server_steps.py exec-server")
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
