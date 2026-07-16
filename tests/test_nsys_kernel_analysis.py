import tempfile
import unittest
from pathlib import Path

from scripts.tools.nsys.classify_kernels import classify_kernel, classify_kernels, write_classification
from scripts.tools.nsys.models import KernelSummary


class NsysKernelAnalysisTest(unittest.TestCase):
    def row(self, name, total=100, calls=1):
        return KernelSummary(name, total, calls, 0.0)

    def test_ordered_rules_prevent_generic_misclassification(self):
        self.assertEqual(classify_kernel("ncclKernel_AllReduce_RING_LL")[0], "NCCL Communication")
        self.assertEqual(classify_kernel("moe_expert_gemm_kernel")[0], "MoE Expert GEMM")
        self.assertEqual(classify_kernel("flash_mla_gemm")[0], "Attention/MLA")
        self.assertEqual(classify_kernel("plain_gemm")[0], "Dense GEMM")
        self.assertEqual(classify_kernel("mystery_xyz")[0], "Unknown")

    def test_classification_preserves_unknown_and_recomputes_shares(self):
        rows = [self.row("ncclAllReduce", 60, 3), self.row("mystery_xyz", 40, 2)]
        classified = classify_kernels(rows)

        self.assertEqual(classified[1].category, "Unknown")
        self.assertAlmostEqual(classified[0].time_percentage, 60.0)
        self.assertEqual(classified[0].instances, 3)

    def test_writes_classification_and_unknown_artifacts(self):
        rows = classify_kernels([self.row("gemm", 80), self.row("mystery", 20)])
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_classification(rows, output)
            classification = (output / "kernel_classification.csv").read_text()
            unknown = (output / "unknown_kernels.csv").read_text()

        self.assertIn("base_family,category,rule", classification)
        self.assertIn("mystery", unknown)


if __name__ == "__main__":
    unittest.main()
