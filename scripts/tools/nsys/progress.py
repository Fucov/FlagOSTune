"""Observable stage and subprocess progress for long Nsight operations."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Optional, Sequence

from .utils import format_bytes


class ProgressReporter:
    def __init__(self, total_stages: int, stream=None, log_path: Optional[Path] = None):
        self.total_stages = max(1, total_stages)
        self.stream = stream if stream is not None else sys.stderr
        self.log_path = log_path
        self.stage = 0
        self._lock = threading.Lock()
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, message: str) -> None:
        line = message.rstrip("\n")
        with self._lock:
            print(line, file=self.stream, flush=True)
            if self.log_path is not None:
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
                    handle.flush()

    def begin(
        self,
        name: str,
        command: Optional[Sequence[str]] = None,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> float:
        self.stage += 1
        started = time.monotonic()
        stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        self.emit(f"[{self.stage}/{self.total_stages}] {name} | STARTED {stamp}")
        if command:
            self.emit("  command: " + shlex.join([str(value) for value in command]))
        if input_path is not None:
            size = input_path.stat().st_size if input_path.exists() else None
            self.emit(f"  input: {input_path} ({format_bytes(size)})")
        if output_path is not None:
            size = output_path.stat().st_size if output_path.exists() else 0
            self.emit(f"  output: {output_path} ({format_bytes(size)})")
        return started

    def finish(
        self,
        name: str,
        started: float,
        status: str = "SUCCESS",
        output_path: Optional[Path] = None,
        detail: Optional[str] = None,
    ) -> None:
        elapsed = time.monotonic() - started
        size = output_path.stat().st_size if output_path and output_path.exists() else None
        suffix = f" | output={format_bytes(size)}" if output_path is not None else ""
        if detail:
            suffix += f" | {detail}"
        self.emit(f"  {status} {name} | elapsed={elapsed:.1f}s{suffix}")

    def warning(self, message: str) -> None:
        self.emit(f"  WARNING: {message}")


def run_streaming_command(
    command: Sequence[str],
    stdout_path: Optional[Path],
    stderr_log_path: Path,
    progress: ProgressReporter,
    heartbeat_seconds: float = 10.0,
    monitored_output: Optional[Path] = None,
    popen_factory=subprocess.Popen,
    merge_stderr: bool = False,
) -> int:
    """Run a child while forwarding stderr and reporting output growth."""
    stderr_log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle: Optional[IO[bytes]] = None
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_handle = stdout_path.open("wb")
    monitor = monitored_output if monitored_output is not None else stdout_path
    started = time.monotonic()
    try:
        try:
            process = popen_factory(
                [str(value) for value in command],
                stdout=stdout_handle if stdout_handle is not None else subprocess.DEVNULL,
                stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
                bufsize=0,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"executable not found: {command[0]}") from exc

        def forward_stderr() -> None:
            if process.stderr is None:
                return
            with stderr_log_path.open("ab") as log:
                for raw in iter(process.stderr.readline, b""):
                    log.write(raw)
                    log.flush()
                    progress.emit(raw.decode("utf-8", errors="replace").rstrip("\n"))

        reader = None
        if not merge_stderr:
            reader = threading.Thread(target=forward_stderr, daemon=True)
            reader.start()
        interval = max(0.01, heartbeat_seconds)
        while True:
            try:
                return_code = process.wait(timeout=interval)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started
                size = monitor.stat().st_size if monitor and monitor.exists() else 0
                progress.emit(
                    f"  heartbeat | elapsed={elapsed:.1f}s | output={format_bytes(size)}"
                )
        if reader is not None:
            reader.join(timeout=2.0)
        if process.stderr is not None:
            process.stderr.close()
        return int(return_code)
    except KeyboardInterrupt:
        if "process" in locals() and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        if stdout_handle is not None:
            stdout_handle.flush()
            os.fsync(stdout_handle.fileno())
            stdout_handle.close()
