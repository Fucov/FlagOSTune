import tempfile
import unittest
from pathlib import Path

from scripts.tools.nsys.analyze_phases import attribute_phase
from scripts.tools.nsys.models import (
    AnalysisData,
    ClassifiedKernel,
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
                "raw_report_integrity": "PASS",
                "analysis_completeness": "PARTIAL",
                "analysis_completeness_reasons": ["event trace unavailable"],
            },
            reports=reports,
            kernels=[KernelSummary("complete_kernel_name", 1000, 5, 100.0, 200)],
            base_kernels=[KernelSummary("base_family", 1000, 5, 100.0, 200)],
            operator_hotspots=[
                ClassifiedKernel(
                    "fused_allreduce_rmsnorm_kernel",
                    "fused_allreduce_rmsnorm_kernel",
                    "Fused Communication-Compute",
                    "explicit fused communication and normalization tokens",
                    "HIGH",
                    1000,
                    5,
                    100.0,
                    "YES",
                    "communication-compute",
                    "one kernel contains allreduce + rmsnorm + fused",
                )
            ],
            devices=[DeviceSummary(0, 1, 5, 1000, 800, 200, 20.0, "base_family", 1.0, 0.0, "GPU-0")],
            native_tables={
                "cuda_gpu_kern_gb_sum": [{"Grid XYZ": "1 2 3", "Block XYZ": "128 1 1"}],
                "cuda_api_sum": [{"Name": "cudaLaunchKernel", "Time (%)": "100"}],
                "cuda_kern_exec_sum": [{"API Total Time (ns)": "20", "Queue Total Time (ns)": "10", "Kernel Total Time (ns)": "1000"}],
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
        self.assertIn("Full-Inference Operator Hotspots", markdown)
        self.assertIn("fused_allreduce_rmsnorm_kernel", markdown)
        self.assertIn("communication-compute", markdown)
        self.assertIn("YES", markdown)
        self.assertIn("Kernel instances | 5", markdown)
        self.assertIn("GPU-0", markdown)
        self.assertIn("NVTX data is empty", markdown)
        self.assertIn("Raw report integrity: **PASS**", markdown)
        self.assertIn("Analysis completeness: **PARTIAL**", markdown)
        self.assertNotIn("['event trace unavailable']", markdown)
        self.assertIn("includes model initialization", markdown)
        self.assertIn("cannot be used for stable decode communication share", markdown)
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

    def test_nvtx_diagnostics_explain_skips_without_event_type_semantics(self):
        data = AnalysisData(
            metadata={
                "nvtx_load_status": "PASS_WITH_WARNINGS",
                "nvtx_load_stats": {
                    "valid_closed_ranges": 2946116,
                    "null_start_rows": 0,
                    "null_end_rows": 16,
                    "invalid_interval_rows": 0,
                    "load_duration_seconds": 1.5,
                    "estimated_memory_bytes": 1024,
                    "counts_by_event_type": {59: 2946116, 75: 8, 39: 8},
                    "skipped_by_event_type": {75: 8, 39: 8},
                },
            },
            reports=ReportCollection(),
        )
        markdown = render_markdown(data)
        self.assertIn("### NVTX Diagnostics", markdown)
        self.assertIn("NULL end rows skipped: `16`", markdown)
        self.assertIn("No end timestamp is synthesized", markdown)
        self.assertIn("not assigned universal semantics", markdown)

    def test_mapping_table_does_not_render_python_list_literals(self):
        data = AnalysisData(
            metadata={"analysis_completeness_reasons": ["one", "two"]},
            reports=ReportCollection(),
            native_tables={"cuda_api_sum": [{"Names": ["a", "b"]}]},
        )
        markdown = render_markdown(data, top=20)
        self.assertNotIn("['a', 'b']", markdown)
        self.assertIn('["a", "b"]', markdown)


if __name__ == "__main__":
    unittest.main()
