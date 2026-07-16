import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class NsysDocumentationTest(unittest.TestCase):
    def test_readme_documents_capture_parse_monitoring_and_artifacts(self):
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        required = (
            "Qwen3.6-35B-A3B-FP8-TP4-P128D16",
            "DeepSeek-V4-Flash-FP8-TP8-Profile-P2048D32C64",
            "--capture-mode full-offline",
            "--parse-output-dir",
            "--force-export",
            "--analyze-dependencies",
            "progress.log",
            "export_sqlite.log",
            "cuda_gpu_kern_sum.csv",
            "kernel_adjacency.csv",
            "communication_chains.csv",
            "nsys_analysis.md",
            "server-steps",
            "decode-only",
        )
        for value in required:
            self.assertIn(value, text)


if __name__ == "__main__":
    unittest.main()
