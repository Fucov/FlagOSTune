import io
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.tools.nsys.collect_stats import (
    CoreReportError,
    collect_reports,
    detect_supported_reports,
    parse_help_report_names,
    report_filename,
    report_candidates,
    select_reports,
)
from scripts.tools.nsys.progress import ProgressReporter


FAKE_NSYS = """#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
with open(os.environ['FAKE_NSYS_CALLS'], 'a', encoding='utf-8') as handle:
    handle.write(' '.join(args) + '\\n')
if '--help-reports' in args:
    text = os.environ.get(
        'FAKE_NSYS_HELP_TEXT',
        'The following built-in reports are available:\\n'
        'cuda_gpu_kern_sum cuda_gpu_kern_sum:base cuda_api_sum nvtx_sum',
    )
    stream = sys.stderr if os.environ.get('FAKE_NSYS_HELP_STREAM') == 'stderr' else sys.stdout
    if text:
        print(text, file=stream, flush=True)
    raise SystemExit(int(os.environ.get('FAKE_NSYS_HELP_EXIT', '0')))
report = args[args.index('--report') + 1]
if report == os.environ.get('FAKE_NSYS_FAIL'):
    print('missing table', file=sys.stderr, flush=True)
    raise SystemExit(9)
print('Time (%),Total Time (ns),Instances,Avg (ns),Name')
print('100,1000,1,1000,' + report)
"""


class NsysCollectStatsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.summary = self.root / "summary"
        self.summary.mkdir()
        self.sqlite = self.root / "capture.sqlite"
        connection = sqlite3.connect(str(self.sqlite))
        connection.execute("create table fixture(value integer)")
        connection.commit()
        connection.close()
        self.nsys = self.root / "nsys"
        self.nsys.write_text(FAKE_NSYS, encoding="utf-8")
        self.nsys.chmod(0o755)
        self.calls = self.root / "calls"
        self.old_env = dict(os.environ)
        os.environ["FAKE_NSYS_CALLS"] = str(self.calls)
        self.stderr = io.StringIO()
        self.progress = ProgressReporter(20, self.stderr, self.summary / "progress.log")

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)
        self.temp_dir.cleanup()

    def test_report_filename_is_filesystem_safe(self):
        self.assertEqual(report_filename("cuda_gpu_kern_sum:base"), "cuda_gpu_kern_sum_base.csv")

    def test_default_reports_avoid_unsupported_2025_base_variants(self):
        selected = select_reports(None, True, True)
        for report in (
            "cuda_gpu_kern_sum:base",
            "cuda_kern_exec_sum:base",
            "cuda_gpu_trace:base",
            "cuda_kern_exec_trace:base",
            "nvtx_kern_sum:base",
        ):
            self.assertNotIn(report, selected)

    def test_event_report_fallback_candidates_are_ordered(self):
        self.assertEqual(
            report_candidates("cuda_gpu_trace"),
            ("cuda_gpu_trace:nvtx-name", "cuda_gpu_trace"),
        )
        self.assertEqual(
            report_candidates("cuda_kern_exec_trace"),
            ("cuda_kern_exec_trace:nvtx-name", "cuda_kern_exec_trace"),
        )
        self.assertEqual(report_candidates("nvtx_kern_sum"), ("nvtx_kern_sum",))

    def test_detects_supported_reports_without_force_export(self):
        supported = detect_supported_reports(str(self.nsys), self.summary, self.progress)

        self.assertIn("cuda_gpu_kern_sum", supported)
        calls = self.calls.read_text()
        self.assertIn("stats --help-reports", calls)
        self.assertNotIn("force-export", calls)

    def test_nonzero_exit_with_valid_help_is_accepted_with_warning(self):
        os.environ["FAKE_NSYS_HELP_EXIT"] = "1"

        supported = detect_supported_reports(str(self.nsys), self.summary, self.progress)

        self.assertIn("cuda_gpu_kern_sum", supported)
        self.assertIn("cuda_api_sum", supported)
        self.assertIn("WARNING", self.stderr.getvalue())
        self.assertIn("exit 1", self.stderr.getvalue())

    def test_nonzero_exit_accepts_valid_report_body_without_canonical_heading(self):
        os.environ["FAKE_NSYS_HELP_EXIT"] = "1"
        os.environ["FAKE_NSYS_HELP_TEXT"] = (
            "cuda_gpu_kern_sum cuda_api_sum nvtx_sum cuda_gpu_trace[:nvtx-name]"
        )
        supported = detect_supported_reports(
            str(self.nsys), self.summary, self.progress
        )
        self.assertIn("cuda_gpu_kern_sum", supported)
        self.assertIn("cuda_gpu_trace", supported)

    def test_nonzero_exit_with_empty_help_is_rejected(self):
        os.environ["FAKE_NSYS_HELP_EXIT"] = "1"
        os.environ["FAKE_NSYS_HELP_TEXT"] = ""

        with self.assertRaisesRegex(CoreReportError, "help-reports"):
            detect_supported_reports(str(self.nsys), self.summary, self.progress)

    def test_valid_help_written_to_stderr_is_accepted(self):
        os.environ["FAKE_NSYS_HELP_EXIT"] = "1"
        os.environ["FAKE_NSYS_HELP_STREAM"] = "stderr"

        supported = detect_supported_reports(str(self.nsys), self.summary, self.progress)

        self.assertIn("cuda_gpu_kern_sum", supported)
        self.assertIn("cuda_api_sum", supported)

    def test_optional_help_grammar_parses_base_report_name(self):
        supported = parse_help_report_names(
            "The following built-in reports are available:\n"
            "nvtx_sum[:nvtx-name][:base|:mangled]\n"
            "cuda_gpu_kern_sum:base\n"
            "cuda_gpu_trace:nvtx-name\n"
        )

        self.assertIn("nvtx_sum", supported)
        self.assertIn("cuda_gpu_kern_sum:base", supported)
        self.assertIn("cuda_gpu_trace:nvtx-name", supported)
        self.assertNotIn("nvtx-name", supported)
        self.assertNotIn("mangled", supported)

    def test_collects_each_report_from_sqlite_and_persists_csv(self):
        result = collect_reports(
            self.sqlite,
            ["cuda_gpu_kern_sum", "cuda_api_sum"],
            {"cuda_gpu_kern_sum", "cuda_api_sum"},
            str(self.nsys),
            self.summary,
            self.progress,
        )

        self.assertEqual(set(result.successful), {"cuda_gpu_kern_sum", "cuda_api_sum"})
        self.assertTrue((self.summary / "cuda_gpu_kern_sum.csv").is_file())
        self.assertTrue((self.summary / "cuda_api_sum.csv").is_file())
        for line in self.calls.read_text().splitlines():
            self.assertIn(str(self.sqlite), line)
            self.assertNotIn("nsys-rep", line)
            self.assertNotIn("force-export", line)

    def test_none_supported_set_directly_probes_selected_reports(self):
        result = collect_reports(
            self.sqlite,
            ["cuda_gpu_kern_sum", "cuda_api_sum"],
            None,
            str(self.nsys),
            self.summary,
            self.progress,
        )

        self.assertEqual(set(result.successful), {"cuda_gpu_kern_sum", "cuda_api_sum"})
        calls = self.calls.read_text()
        self.assertIn("--report cuda_gpu_kern_sum", calls)
        self.assertIn("--report cuda_api_sum", calls)

    def test_optional_report_failure_warns_and_core_failure_is_fatal(self):
        os.environ["FAKE_NSYS_FAIL"] = "nvtx_sum"
        optional = collect_reports(
            self.sqlite,
            ["cuda_gpu_kern_sum", "nvtx_sum"],
            {"cuda_gpu_kern_sum", "nvtx_sum"},
            str(self.nsys),
            self.summary,
            self.progress,
        )
        self.assertIn("nvtx_sum", optional.failed)
        self.assertIn("WARNING", self.stderr.getvalue())

        os.environ["FAKE_NSYS_FAIL"] = "cuda_gpu_kern_sum"
        with self.assertRaisesRegex(CoreReportError, "cuda_gpu_kern_sum"):
            collect_reports(
                self.sqlite,
                ["cuda_gpu_kern_sum"],
                {"cuda_gpu_kern_sum"},
                str(self.nsys),
                self.summary,
                self.progress,
            )

    def test_parser_can_defer_core_failure_to_sqlite_fallback(self):
        os.environ["FAKE_NSYS_FAIL"] = "cuda_gpu_kern_sum"
        result = collect_reports(
            self.sqlite,
            ["cuda_gpu_kern_sum", "cuda_api_sum"],
            {"cuda_gpu_kern_sum", "cuda_api_sum"},
            str(self.nsys),
            self.summary,
            self.progress,
            allow_core_fallback=True,
        )
        self.assertNotIn("cuda_gpu_kern_sum", result.successful)
        self.assertIn("cuda_gpu_kern_sum", result.failed)
        self.assertIn("cuda_api_sum", result.successful)

    def test_fallback_uses_plain_trace_when_nvtx_name_candidate_fails(self):
        os.environ["FAKE_NSYS_FAIL"] = "cuda_gpu_trace:nvtx-name"
        result = collect_reports(
            self.sqlite,
            ["cuda_gpu_kern_sum", "cuda_gpu_trace"],
            {"cuda_gpu_kern_sum", "cuda_gpu_trace"},
            str(self.nsys),
            self.summary,
            self.progress,
        )
        self.assertIn("cuda_gpu_trace", result.successful)
        self.assertEqual(result.selected_sources["cuda_gpu_trace"], "cuda_gpu_trace")
        calls = self.calls.read_text()
        self.assertLess(
            calls.index("--report cuda_gpu_trace:nvtx-name"),
            calls.index("--report cuda_gpu_trace --format"),
        )

    def test_unsupported_optional_warns_and_core_is_always_selected(self):
        selected = select_reports("cuda_api_sum", False, False)
        self.assertEqual(selected[0], "cuda_gpu_kern_sum")
        result = collect_reports(
            self.sqlite,
            ["cuda_gpu_kern_sum", "nvtx_gpu_proj_sum"],
            {"cuda_gpu_kern_sum"},
            str(self.nsys),
            self.summary,
            self.progress,
        )
        self.assertEqual(result.unsupported, ["nvtx_gpu_proj_sum"])
        self.assertIn("unsupported", self.stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
