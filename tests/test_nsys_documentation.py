import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class NsysDocumentationTest(unittest.TestCase):
    def test_readme_documents_capture_parse_monitoring_and_artifacts(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        required = (
            "Qwen3.6-35B-A3B-FP8-TP4-P128D16",
            "DeepSeek-V4-Flash-FP8-TP8-Profile-P2048D32C64",
            "--capture-mode server-steps",
            "--profile-phase prefill",
            "--profile-phase decode",
            "--profile-start-step 0",
            "--profile-num-steps",
            "--parse-output-dir",
            "--force-export",
            "--analyze-dependencies",
            "progress.log",
            "export_sqlite.log",
            "cuda_gpu_kern_sum.csv",
            "operator_hotspots.csv",
            "kernel_adjacency.csv",
            "communication_chains.csv",
            "nsys_analysis.md",
            "/start_profile",
            "Profiling done",
            "full-offline",
            "startup_and_full_process",
            "steady_state_guaranteed=false",
            "YES / NO / UNKNOWN",
            "cuda_gpu_trace:nvtx-name",
            "raw_report_integrity",
            "analysis_completeness",
        )
        for value in required:
            self.assertIn(value, text)
        self.assertNotIn("--capture-mode server-full", text)


if __name__ == "__main__":
    unittest.main()
