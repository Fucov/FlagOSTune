import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.tools.parse_nsys import (
    kernel_rows_with_percentage,
    parse_summary_csv,
    run_nsys_report,
)


CUDA_CSV = """Processing report...
Time (%),Total Time (ns),Instances,Avg (ns),Name
60.0,600,3,200,kernel_a
40.0,400,2,200,kernel_b
"""


class ParseNsysTest(unittest.TestCase):
    def test_parses_preamble_and_calculates_kernel_percentages(self):
        rows = parse_summary_csv(CUDA_CSV, name_aliases=("Name",))
        kernels = kernel_rows_with_percentage(rows)

        self.assertEqual([row.name for row in kernels], ["kernel_a", "kernel_b"])
        self.assertAlmostEqual(kernels[0].time_percentage, 60.0)
        self.assertAlmostEqual(kernels[1].time_percentage, 40.0)
        self.assertEqual(kernels[0].instances, 3)
        self.assertEqual(kernels[0].avg_ns, 200.0)

    def test_accepts_nvtx_range_and_cuda_api_call_aliases(self):
        nvtx = parse_summary_csv(
            "Time (%),Total Time (ns),Instances,Avg (ns),Range\n"
            "100,500,1,500,decode\n",
            name_aliases=("Range", "Name"),
        )
        api = parse_summary_csv(
            "Time (%),Total Time (ns),Num Calls,Avg (ns),Name\n"
            "100,250,5,50,cudaLaunchKernel\n",
            name_aliases=("Name",),
        )

        self.assertEqual((nvtx[0].name, nvtx[0].instances), ("decode", 1))
        self.assertEqual((api[0].name, api[0].instances), ("cudaLaunchKernel", 5))

    def test_rows_are_sorted_by_total_time(self):
        rows = parse_summary_csv(
            "Total Time (ns),Calls,Avg (ns),Name,Time (%)\n"
            "100,1,100,small,10\n900,3,300,large,90\n",
            name_aliases=("Name",),
        )

        self.assertEqual([row.name for row in rows], ["large", "small"])

    def test_malformed_csv_reports_missing_columns(self):
        with self.assertRaisesRegex(ValueError, "summary header"):
            parse_summary_csv("Name,Calls\nkernel,1\n", name_aliases=("Name",))

    def test_empty_and_zero_kernel_tables_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no CUDA kernel"):
            kernel_rows_with_percentage([])

        rows = parse_summary_csv(
            "Total Time (ns),Calls,Avg (ns),Name,Time (%)\n"
            "0,1,0,empty,0\n",
            name_aliases=("Name",),
        )
        with self.assertRaisesRegex(ValueError, "zero total CUDA kernel"):
            kernel_rows_with_percentage(rows)

    def test_run_nsys_report_builds_expected_command(self):
        seen = []

        def runner(cmd, **kwargs):
            seen.append((cmd, kwargs))
            return subprocess.CompletedProcess(cmd, 0, stdout=CUDA_CSV, stderr="")

        output = run_nsys_report(
            Path("capture.sqlite"),
            "cuda_gpu_kern_sum",
            nsys="/opt/nsys",
            runner=runner,
        )

        self.assertEqual(output, CUDA_CSV)
        self.assertEqual(
            seen[0][0],
            [
                "/opt/nsys",
                "stats",
                "--report",
                "cuda_gpu_kern_sum",
                "--format",
                "csv",
                "capture.sqlite",
            ],
        )
        self.assertEqual(seen[0][1], {"capture_output": True, "text": True})

    def test_run_nsys_report_includes_report_name_and_stderr_on_failure(self):
        def runner(cmd, **_kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="bad schema")

        with self.assertRaisesRegex(
            RuntimeError,
            "cuda_api_sum.*bad schema",
        ):
            run_nsys_report(
                Path("capture.nsys-rep"),
                "cuda_api_sum",
                runner=runner,
            )

    def test_cli_prints_all_three_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "capture.nsys-rep"
            report.touch()
            fake_nsys = Path(tmp) / "nsys"
            fake_nsys.write_text(
                "#!/usr/bin/env python3\n"
                "import sqlite3,sys\n"
                "a=sys.argv[1:]\n"
                "if a[0]=='--version': print('Nsight Systems 2026.1'); raise SystemExit\n"
                "if a[0]=='export':\n"
                " p=a[a.index('--output')+1]; c=sqlite3.connect(p); "
                "c.execute('create table CUPTI_ACTIVITY_KIND_KERNEL (deviceId integer, globalPid integer, start integer, end integer, name text)'); "
                "c.commit(); c.close(); raise SystemExit\n"
                "if '--help-reports' in a:\n"
                " print('cuda_gpu_kern_sum cuda_gpu_kern_sum:base cuda_gpu_kern_gb_sum cuda_kern_exec_sum:base cuda_api_sum nvtx_sum nvtx_gpu_proj_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum'); raise SystemExit\n"
                "r=a[a.index('--report')+1]\n"
                "name='Range' if r=='nvtx_sum' else 'Name'\n"
                "print(f'Time (%),Total Time (ns),Instances,Avg (ns),{name}')\n"
                "print('100,1000,1,1000,item')\n",
                encoding="utf-8",
            )
            fake_nsys.chmod(0o755)

            result = subprocess.run(
                [
                    "python3",
                    "scripts/tools/parse_nsys.py",
                    str(report),
                    "--nsys",
                    str(fake_nsys),
                    "--top",
                    "1",
                ],
                cwd=Path(__file__).resolve().parent.parent,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Top Kernel Variants", result.stdout)
        self.assertIn("NVTX and Module Attribution", result.stdout)
        self.assertIn("CUDA API and Launch/Execution", result.stdout)
        self.assertIn("100.00", result.stdout)
        self.assertIn("[1/", result.stderr)


if __name__ == "__main__":
    unittest.main()
