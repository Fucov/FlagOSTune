"""Validation and one-time atomic export of `.nsys-rep` files."""

from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path
from typing import Optional

from .progress import ProgressReporter, run_streaming_command


class ExportError(RuntimeError):
    pass


def is_valid_sqlite(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = connection.execute("PRAGMA quick_check").fetchone()
            return bool(row and str(row[0]).lower() == "ok")
        finally:
            connection.close()
    except sqlite3.Error:
        return False


def resolve_sqlite(
    input_path: Path,
    output_dir: Path,
    nsys_path: str,
    force_export: bool = False,
    reuse_sqlite: bool = True,
    progress: Optional[ProgressReporter] = None,
) -> Path:
    input_path = input_path.absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    reporter = progress or ProgressReporter(2, log_path=output_dir / "progress.log")
    started = reporter.begin("Validate input", input_path=input_path)
    if not input_path.is_file():
        reporter.finish("Validate input", started, "FAILED", detail="input does not exist")
        raise ExportError(f"input does not exist: {input_path}")
    if input_path.suffix == ".sqlite":
        if force_export or not reuse_sqlite:
            reporter.finish("Validate input", started, "FAILED", detail="export flags invalid")
            raise ExportError("export/cache flags are not valid for direct SQLite input")
        if not is_valid_sqlite(input_path):
            reporter.finish("Validate input", started, "FAILED", detail="invalid SQLite")
            raise ExportError(f"invalid SQLite database: {input_path}")
        reporter.finish("Validate input", started, output_path=input_path)
        return input_path
    if not input_path.name.endswith(".nsys-rep"):
        reporter.finish("Validate input", started, "FAILED", detail="unsupported suffix")
        raise ExportError(f"expected .nsys-rep or .sqlite input: {input_path}")
    reporter.finish("Validate input", started, output_path=input_path)

    sqlite_path = input_path.with_name(input_path.name[: -len(".nsys-rep")] + ".sqlite")
    current = (
        reuse_sqlite
        and not force_export
        and is_valid_sqlite(sqlite_path)
        and sqlite_path.stat().st_mtime >= input_path.stat().st_mtime
    )
    if current:
        started = reporter.begin("Export report to SQLite", input_path=input_path, output_path=sqlite_path)
        reporter.finish("Export report to SQLite", started, "REUSED", output_path=sqlite_path)
        return sqlite_path

    if shutil.which(nsys_path) is None and not Path(nsys_path).is_file():
        raise ExportError(f"Nsight Systems executable not found: {nsys_path}")
    free_bytes = shutil.disk_usage(str(sqlite_path.parent)).free
    if free_bytes < max(input_path.stat().st_size, 1):
        raise ExportError(
            f"insufficient disk space for SQLite export: free={free_bytes}, input={input_path.stat().st_size}"
        )

    temporary = Path(str(sqlite_path) + ".tmp")
    if temporary.exists():
        temporary.unlink()
    command = [
        nsys_path,
        "export",
        "--type",
        "sqlite",
        "--output",
        str(temporary),
        str(input_path),
    ]
    started = reporter.begin(
        "Export report to SQLite", command=command, input_path=input_path, output_path=temporary
    )
    try:
        return_code = run_streaming_command(
            command,
            None,
            output_dir / "export_sqlite.log",
            reporter,
            monitored_output=temporary,
        )
        if return_code != 0:
            raise ExportError(
                f"nsys export failed with exit {return_code}; see {output_dir / 'export_sqlite.log'}"
            )
        if not is_valid_sqlite(temporary):
            raise ExportError("nsys export completed but produced an invalid SQLite database")
        os.replace(str(temporary), str(sqlite_path))
        reporter.finish("Export report to SQLite", started, output_path=sqlite_path)
        return sqlite_path
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        reporter.finish("Export report to SQLite", started, "FAILED", output_path=sqlite_path)
        raise
