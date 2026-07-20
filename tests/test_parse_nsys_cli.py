import json
import os
import sqlite3
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PARSER = ROOT / "scripts" / "tools" / "parse_nsys.py"

FAKE_NSYS = """#!/usr/bin/env python3
import os, sqlite3, sys
a=sys.argv[1:]
with open(os.environ['NSYS_CALLS'], 'a', encoding='utf-8') as h: h.write(' '.join(a)+'\\n')
if a[0] == '--version': print('NVIDIA Nsight Systems version 2026.1'); raise SystemExit
if a[0] == 'export':
 p=a[a.index('--output')+1]; c=sqlite3.connect(p)
 c.execute('create table CUPTI_ACTIVITY_KIND_KERNEL (deviceId integer, globalPid integer, start integer, end integer, name text)')
 c.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (0,1,0,100,'gemm')")
 c.commit(); c.close(); print('export progress', file=sys.stderr, flush=True); raise SystemExit
if '--help-reports' in a and os.environ.get('NSYS_HELP_EMPTY'):
 print('', end=''); raise SystemExit(1)
if '--help-reports' in a and os.environ.get('NSYS_HELP_EXIT_ONE'):
 print('The following built-in reports are available:')
 print('cuda_gpu_kern_sum cuda_api_sum nvtx_sum')
 raise SystemExit(1)
if '--help-reports' in a:
 print('cuda_gpu_kern_sum cuda_gpu_kern_sum:base cuda_gpu_kern_gb_sum cuda_kern_exec_sum:base cuda_api_sum nvtx_sum nvtx_gpu_proj_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum cuda_gpu_trace:base cuda_kern_exec_trace:base nvtx_kern_sum:base nvtx_gpu_proj_trace'); raise SystemExit
r=a[a.index('--report')+1]
if r == os.environ.get('NSYS_FAIL'): print('missing table', file=sys.stderr); raise SystemExit(8)
print('Time (%),Total Time (ns),Instances,Avg (ns),Name')
print('100,1000,2,500,' + ('gemm_kernel' if 'kern' in r else r))
"""


class ParseNsysCliTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.report = self.root / "capture.nsys-rep"
        self.report.write_bytes(b"fixture report")
        self.nsys = self.root / "nsys"
        self.nsys.write_text(FAKE_NSYS, encoding="utf-8")
        self.nsys.chmod(0o755)
        self.calls = self.root / "calls"

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_parser(self, *args, extra_env=None):
        env = os.environ.copy()
        env["NSYS_CALLS"] = str(self.calls)
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["python3", str(PARSER), str(self.report), "--nsys", str(self.nsys), *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_exports_once_uses_sqlite_for_all_stats_and_separates_streams(self):
        output = self.root / "custom-summary"
        result = self.run_parser("--output-dir", str(output), "--top", "1")

        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls.read_text().splitlines()
        self.assertEqual(sum(line.startswith("export ") for line in calls), 1)
        stats = [line for line in calls if line.startswith("stats --report")]
        self.assertTrue(stats)
        self.assertTrue(all(str(self.root / "capture.sqlite") in line for line in stats))
        self.assertTrue(all("force-export" not in line and "nsys-rep" not in line for line in stats))
        self.assertIn("FlagOSTune Nsight Systems Analysis", result.stdout)
        self.assertNotIn("[1/", result.stdout)
        self.assertIn("[1/", result.stderr)
        self.assertTrue((output / "cuda_gpu_kern_sum.csv").is_file())
        self.assertTrue((output / "nsys_analysis.md").is_file())
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["input_report"], str(self.report))
        self.assertIn("cuda_gpu_kern_sum", metadata["successful_reports"])

    def test_reuses_newer_sqlite_and_force_export_refreshes(self):
        first = self.run_parser("--reports", "cuda_api_sum")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.calls.unlink()
        reused = self.run_parser("--reports", "cuda_api_sum")
        self.assertEqual(reused.returncode, 0, reused.stderr)
        self.assertFalse(any(line.startswith("export ") for line in self.calls.read_text().splitlines()))
        self.calls.unlink()
        forced = self.run_parser("--reports", "cuda_api_sum", "--force-export")
        self.assertEqual(forced.returncode, 0, forced.stderr)
        self.assertEqual(sum(line.startswith("export ") for line in self.calls.read_text().splitlines()), 1)

    def test_resume_reuses_valid_native_stats_csv(self):
        output = self.root / "resume-summary"
        first = self.run_parser("--output-dir", str(output), "--reports", "cuda_api_sum")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.calls.unlink()

        resumed = self.run_parser(
            "--output-dir", str(output), "--reports", "cuda_api_sum", "--resume"
        )

        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        calls = self.calls.read_text().splitlines()
        self.assertFalse(any(line.startswith("stats ") for line in calls))
        self.assertIn("REUSED Detect supported reports", resumed.stderr)
        self.assertIn("REUSED Generate cuda_gpu_kern_sum", resumed.stderr)
        self.assertIn("REUSED Generate cuda_api_sum", resumed.stderr)
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["base_stats_status"], "REUSED")

    def test_optional_failure_warns_but_core_failure_is_nonzero(self):
        optional = self.run_parser("--reports", "nvtx_sum", extra_env={"NSYS_FAIL": "nvtx_sum"})
        self.assertEqual(optional.returncode, 0, optional.stderr)
        self.assertIn("WARNING", optional.stderr)

        self.calls.unlink(missing_ok=True)
        core = self.run_parser(extra_env={"NSYS_FAIL": "cuda_gpu_kern_sum"})
        self.assertNotEqual(core.returncode, 0)
        self.assertIn("cuda_gpu_kern_sum", core.stderr)

    def test_direct_sqlite_and_cache_flags_are_validated(self):
        first = self.run_parser("--reports", "cuda_api_sum")
        self.assertEqual(first.returncode, 0, first.stderr)
        sqlite_path = self.root / "capture.sqlite"
        result = subprocess.run(
            ["python3", str(PARSER), str(sqlite_path), "--nsys", str(self.nsys), "--force-export"],
            cwd=ROOT,
            env={**os.environ, "NSYS_CALLS": str(self.calls)},
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("direct SQLite", result.stderr)

    def test_help_failure_probes_reports_and_reuses_qwen_sqlite(self):
        self.report = self.root / "qwen-tp4-full.nsys-rep"
        self.report.write_bytes(b"existing report")
        sqlite_path = self.root / "qwen-tp4-full.sqlite"
        connection = sqlite3.connect(str(sqlite_path))
        connection.execute(
            "create table CUPTI_ACTIVITY_KIND_KERNEL "
            "(deviceId integer, globalPid integer, start integer, end integer, name text)"
        )
        connection.execute(
            "insert into CUPTI_ACTIVITY_KIND_KERNEL values (0, 1, 0, 100, 'gemm')"
        )
        connection.commit()
        connection.close()
        newer = self.report.stat().st_mtime + 10
        os.utime(sqlite_path, (newer, newer))

        result = self.run_parser(
            "--reports", "cuda_api_sum", extra_env={"NSYS_HELP_EMPTY": "1"}
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls.read_text().splitlines()
        self.assertFalse(any(line.startswith("export ") for line in calls))
        self.assertTrue(any("--report cuda_gpu_kern_sum" in line for line in calls))
        self.assertTrue(any("--report cuda_api_sum" in line for line in calls))
        self.assertIn("WARNING", result.stderr)
        metadata = json.loads((self.root / "summary" / "metadata.json").read_text())
        self.assertTrue(any("help-reports" in warning["message"] for warning in metadata["warnings"]))

    def test_fallback_optional_failure_warns_and_core_failure_is_fatal(self):
        optional = self.run_parser(
            "--reports",
            "nvtx_sum",
            extra_env={"NSYS_HELP_EMPTY": "1", "NSYS_FAIL": "nvtx_sum"},
        )
        self.assertEqual(optional.returncode, 0, optional.stderr)
        self.assertIn("WARNING", optional.stderr)

        self.calls.unlink(missing_ok=True)
        core = self.run_parser(
            extra_env={"NSYS_HELP_EMPTY": "1", "NSYS_FAIL": "cuda_gpu_kern_sum"}
        )
        self.assertNotEqual(core.returncode, 0)
        self.assertIn("cuda_gpu_kern_sum", core.stderr)

    def test_valid_nonzero_help_warning_is_saved_in_metadata(self):
        result = self.run_parser(
            "--reports", "cuda_api_sum", extra_env={"NSYS_HELP_EXIT_ONE": "1"}
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((self.root / "summary" / "metadata.json").read_text())
        messages = [warning["message"] for warning in metadata["warnings"]]
        self.assertTrue(any("exit 1" in message and "help body is valid" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
