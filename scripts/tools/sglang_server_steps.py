#!/usr/bin/env python3
"""Compatibility entry point for phase-selective SGLang server captures."""

from scripts.tools.sglang_server_capture import *  # noqa: F401,F403
from scripts.tools.sglang_server_capture import main


if __name__ == "__main__":
    raise SystemExit(main())
