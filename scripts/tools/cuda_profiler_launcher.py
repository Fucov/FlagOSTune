#!/usr/bin/env python3
"""Run a Python module inside a CUDA Profiler API capture range."""

from __future__ import annotations

import argparse
import runpy
import sys
from typing import Any, Callable, List, Optional


ModuleRunner = Callable[[str, str], Any]


def cuda_call_succeeded(result: object) -> bool:
    """Return whether a CUDA runtime API result represents success."""
    try:
        return int(result) == 0  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return result is None


def load_torch_cudart() -> Any:
    """Load the CUDA runtime exposed by the active PyTorch installation."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to control the CUDA profiler range") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; Nsight profiling requires a CUDA device")
    return torch.cuda.cudart()


def run_module_with_cuda_profiler(
    module: str,
    module_args: List[str],
    *,
    cudart: Optional[Any] = None,
    module_runner: ModuleRunner = runpy.run_module,
) -> Any:
    """Execute *module* in-process between cudaProfilerStart/Stop calls."""
    runtime = cudart if cudart is not None else load_torch_cudart()
    old_argv = list(sys.argv)
    sys.argv = [module, *module_args]
    started = False

    try:
        start_result = runtime.cudaProfilerStart()
        if not cuda_call_succeeded(start_result):
            raise RuntimeError(f"cudaProfilerStart failed: {start_result}")
        started = True
        return module_runner(module, "__main__")
    finally:
        try:
            if started:
                stop_result = runtime.cudaProfilerStop()
                if not cuda_call_succeeded(stop_result) and sys.exc_info()[0] is None:
                    raise RuntimeError(f"cudaProfilerStop failed: {stop_result}")
        finally:
            sys.argv = old_argv


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Python module inside a CUDA Profiler API range"
    )
    parser.add_argument(
        "--module",
        default="sglang.bench_offline_throughput",
        help="Python module to execute",
    )
    parser.add_argument("module_args", nargs=argparse.REMAINDER)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    module_args = list(args.module_args)
    if module_args[:1] == ["--"]:
        module_args.pop(0)

    try:
        run_module_with_cuda_profiler(args.module, module_args)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
