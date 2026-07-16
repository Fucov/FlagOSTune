import tempfile
import unittest
from pathlib import Path

from scripts.tools.nsys.analyze_phases import attribute_phase
from scripts.tools.nsys.models import (
    AnalysisData,
    DeviceSummary,
    KernelSummary,
    ReportCollection,
    WarningRecord,
)
from scripts.tools.nsys.render_markdown import SECTION_TITLES, render_markdown


class NsysRenderMarkdownTest(unittest.TestCase):
    def test_phase_evidence_priority_and_unknown_fallback(self):
        explicit = attribute_phase({"profile_phase": "decode"}, ["prefill range"])
        self.assertEqual(explicit.phase, "PREFILL")
        self.assertEqual(explicit.source, "NVTX")

        metadata = attribute_phase({"profile_phase": "decode"}, [])
        self.assertEqual(metadata.phase, "DECODE")
        self.assertEqual(metadata.source, "workflow metadata")

        unknown = attribute_phase({}, [])
        self.assertEqual(unknown.phase, "UNKNOWN")

    def test_report_has_fixed_sections_and_scope_caveats(self):
        reports = ReportCollection()
        reports.unsupported.append("nvtx_sum")
        data = AnalysisData(
            metadata={
                "input_report": "/reports/capture.nsys-rep",
                "input_size": 1024,
                "input_mtime": "2026-07-16T00:00:00+08:00",
                "sqlite_path": "/reports/capture.sqlite",
                "sqlite_size": 2048,
                "nsys_version": "2026.1",
                "capture_mode": "full-offline",
                "profile_phase": "full",
                "integrity_ok": True,
            },
            reports=reports,
            kernels=[KernelSummary("complete_kernel_name", 1000, 5, 100.0, 200)],
            base_kernels=[KernelSummary("base_family", 1000, 5, 100.0, 200)],
            devices=[DeviceSummary(0, 1, 5, 1000, 800, 200, 20.0, "base_family", 1.0, 0.0, "GPU-0")],
            native_tables={
                "cuda_gpu_kern_gb_sum": [{"Grid XYZ": "1 2 3", "Block XYZ": "128 1 1"}],
                "cuda_api_sum": [{"Name": "cudaLaunchKernel", "Time (%)": "100"}],
                "cuda_kern_exec_sum:base": [{"API Total Time (ns)": "20", "Queue Total Time (ns)": "10", "Kernel Total Time (ns)": "1000"}],
                "nvtx_gpu_proj_sum": [],
                "cuda_gpu_mem_time_sum": [{"Operation": "memcpy", "Total Time (ns)": "10"}],
            },
            warnings=[WarningRecord("nvtx", "NVTX data is empty")],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nsys_analysis.md"
            markdown = render_markdown(data, top=20, output_path=path)
            self.assertEqual(path.read_text(), markdown)

        positions = [markdown.index(title) for title in SECTION_TITLES]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("complete_kernel_name", markdown)
        self.assertIn("Kernel instances | 5", markdown)
        self.assertIn("GPU-0", markdown)
        self.assertIn("NVTX data is empty", markdown)
        required = (
            "sum of all kernel durations",
            "not wall-clock",
            "sum of CUDA API durations",
            "not critical-path communication overhead",
            "not a Tensor data dependency",
            "not Tensor shape",
            "FlagOSTune-derived timeline metric",
            "full-run trace is not decode-only",
            "CUDA-graph-disabled",
            "captured workload only",
        )
        for phrase in required:
            self.assertIn(phrase, markdown)

    def test_missing_optional_data_is_na_not_zero(self):
        data = AnalysisData(metadata={}, reports=ReportCollection())
        markdown = render_markdown(data, top=20)
        self.assertIn("N/A", markdown)


if __name__ == "__main__":
    unittest.main()
