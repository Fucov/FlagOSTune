import io
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from scripts.tools.nsys.export_report import ExportError, resolve_sqlite
from scripts.tools.nsys.progress import ProgressReporter, run_streaming_command


FAKE_NSYS = """#!/usr/bin/env python3
import os, sqlite3, sys
args = sys.argv[1:]
with open(os.environ['FAKE_NSYS_CALLS'], 'a', encoding='utf-8') as log:
    log.write(' '.join(args) + '\\n')
if args[0] == 'export':
    output = args[args.index('--output') + 1]
    print('exporting fixture', file=sys.stderr, flush=True)
    connection = sqlite3.connect(output)
    connection.execute('create table exported(value integer)')
    connection.commit()
    connection.close()
"""


class NsysExportReportTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.report = self.root / "capture.nsys-rep"
        self.report.write_bytes(b"report")
        self.fake_nsys = self.root / "nsys"
        self.fake_nsys.write_text(FAKE_NSYS, encoding="utf-8")
        self.fake_nsys.chmod(0o755)
        self.calls = self.root / "calls.txt"
        self.old_calls = os.environ.get("FAKE_NSYS_CALLS")
        os.environ["FAKE_NSYS_CALLS"] = str(self.calls)
        self.stderr = io.StringIO()
        self.progress = ProgressReporter(3, stream=self.stderr, log_path=self.root / "progress.log")

    def tearDown(self):
        if self.old_calls is None:
            os.environ.pop("FAKE_NSYS_CALLS", None)
        else:
            os.environ["FAKE_NSYS_CALLS"] = self.old_calls
        self.temp_dir.cleanup()

    def make_sqlite(self, path):
        connection = sqlite3.connect(str(path))
        connection.execute("create table fixture(value integer)")
        connection.commit()
        connection.close()

    def test_missing_sqlite_is_exported_once_and_promoted(self):
        result = resolve_sqlite(
            self.report, self.root / "summary", str(self.fake_nsys), progress=self.progress
        )

        self.assertEqual(result, self.root / "capture.sqlite")
        self.assertTrue(result.is_file())
        self.assertFalse(Path(str(result) + ".tmp").exists())
        self.assertEqual(self.calls.read_text().count("export "), 1)
        self.assertIn("Export report to SQLite", self.stderr.getvalue())
        self.assertIn("exporting fixture", self.stderr.getvalue())

    def test_current_sqlite_is_reused(self):
        sqlite_path = self.root / "capture.sqlite"
        self.make_sqlite(sqlite_path)
        newer = self.report.stat().st_mtime + 10
        os.utime(sqlite_path, (newer, newer))

        result = resolve_sqlite(
            self.report, self.root / "summary", str(self.fake_nsys), progress=self.progress
        )

        self.assertEqual(result, sqlite_path)
        self.assertFalse(self.calls.exists())
        self.assertIn("REUSED", self.stderr.getvalue())

    def test_force_export_refreshes_current_sqlite(self):
        sqlite_path = self.root / "capture.sqlite"
        self.make_sqlite(sqlite_path)
        os.utime(sqlite_path, (time.time() + 10, time.time() + 10))

        resolve_sqlite(
            self.report,
            self.root / "summary",
            str(self.fake_nsys),
            force_export=True,
            progress=self.progress,
        )

        self.assertEqual(self.calls.read_text().count("export "), 1)

    def test_failed_export_preserves_report_and_existing_database(self):
        sqlite_path = self.root / "capture.sqlite"
        self.make_sqlite(sqlite_path)
        original = sqlite_path.read_bytes()
        failing = self.root / "failing-nsys"
        failing.write_text("#!/bin/sh\necho disk-full >&2\nexit 7\n", encoding="utf-8")
        failing.chmod(0o755)

        with self.assertRaisesRegex(ExportError, "exit 7"):
            resolve_sqlite(
                self.report,
                self.root / "summary",
                str(failing),
                force_export=True,
                progress=self.progress,
            )

        self.assertEqual(self.report.read_bytes(), b"report")
        self.assertEqual(sqlite_path.read_bytes(), original)
        self.assertFalse(Path(str(sqlite_path) + ".tmp").exists())

    def test_streaming_command_heartbeats_and_propagates_exit_code(self):
        script = self.root / "slow"
        script.write_text(
            "#!/bin/sh\necho live-error >&2\nsleep 0.1\necho csv-output\nexit 4\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        output = self.root / "output.csv"
        log = self.root / "command.log"

        result = run_streaming_command(
            [str(script)], output, log, self.progress, heartbeat_seconds=0.02
        )

        self.assertEqual(result, 4)
        self.assertEqual(output.read_text(), "csv-output\n")
        self.assertIn("live-error", self.stderr.getvalue())
        self.assertIn("heartbeat", self.stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
