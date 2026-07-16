import tempfile
import unittest
from pathlib import Path

from scripts.tools.nsys.analyze_communication import (
    analyze_communication,
    build_communication_chains,
    build_fusion_candidates,
    write_communication_analysis,
)
from scripts.tools.nsys.analyze_dependencies import build_adjacency
from scripts.tools.nsys.models import KernelEvent


class NsysCommunicationAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.compute_before = KernelEvent(1, 0, 1, 7, 0, 100, "gemm_a", "Dense GEMM", "gemm", "PREFILL", "layer.0")
        self.communication = KernelEvent(2, 0, 1, 7, 110, 210, "ncclAllReduce", "NCCL Communication", "allreduce", "PREFILL", "layer.0")
        self.compute_after = KernelEvent(3, 0, 1, 7, 220, 300, "gemm_b", "Dense GEMM", "gemm", "PREFILL", "layer.0")
        self.overlap_a = KernelEvent(4, 0, 2, 8, 120, 180, "attention", "Attention/MLA", "attention", "PREFILL", "layer.0")
        self.overlap_b = KernelEvent(5, 0, 2, 9, 150, 200, "gemm_c", "Dense GEMM", "gemm", "PREFILL", "layer.0")
        self.events = [self.compute_before, self.communication, self.compute_after, self.overlap_a, self.overlap_b]

    def test_adjacency_stays_within_device_context_stream(self):
        rows = build_adjacency(self.events)
        pairs = {(row.previous_kernel, row.next_kernel): row for row in rows}

        self.assertIn(("gemm_a", "ncclAllReduce"), pairs)
        self.assertIn(("ncclAllReduce", "gemm_b"), pairs)
        self.assertNotIn(("ncclAllReduce", "attention"), pairs)
        self.assertEqual(pairs[("gemm_a", "ncclAllReduce")].gap_ns, 10)
        self.assertEqual(pairs[("gemm_a", "ncclAllReduce")].relation_type, "temporal_adjacency")
        self.assertNotEqual(pairs[("gemm_a", "ncclAllReduce")].confidence, "HIGH")

    def test_compute_intervals_are_unioned_before_exposed_time(self):
        communication = analyze_communication(self.events)
        event = communication[0]

        self.assertEqual(event.duration_ns, 100)
        self.assertEqual(event.overlap_compute_ns, 80)
        self.assertEqual(event.exposed_communication_ns, 20)
        self.assertGreaterEqual(event.exposed_communication_ns, 0)
        self.assertLessEqual(event.exposed_communication_ns, event.duration_ns)

    def test_chains_use_local_denominator_and_transparent_confidence(self):
        adjacency = build_adjacency(self.events)
        communications = analyze_communication(self.events)
        chains = build_communication_chains(self.events, adjacency, communications)

        three_part = [row for row in chains if row.relation_type == "Compute→Communication→Compute"][0]
        self.assertEqual(three_part.denominator, 1)
        self.assertEqual(three_part.adjacency_rate, 1.0)
        self.assertIn("temporal", three_part.evidence)
        self.assertNotEqual(three_part.confidence, "HIGH")

    def test_fusion_candidates_expose_components_and_distributed_constraint(self):
        communications = analyze_communication(self.events)
        chains = build_communication_chains(self.events, build_adjacency(self.events), communications)
        candidates = build_fusion_candidates(communications, chains)

        candidate = candidates[0]
        self.assertGreaterEqual(candidate.total_score, 0)
        self.assertEqual(candidate.required_primitive, "distributed collective primitive")
        self.assertEqual(candidate.tle_feasibility, "UNKNOWN")
        self.assertNotIn("speedup", candidate.candidate.lower())

    def test_analysis_artifacts_are_written(self):
        adjacency = build_adjacency(self.events)
        communications = analyze_communication(self.events)
        chains = build_communication_chains(self.events, adjacency, communications)
        candidates = build_fusion_candidates(communications, chains)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_communication_analysis(adjacency, communications, chains, candidates, output)
            self.assertTrue((output / "kernel_adjacency.csv").is_file())
            self.assertTrue((output / "communication_events.csv").is_file())
            self.assertTrue((output / "communication_chains.csv").is_file())
            self.assertTrue((output / "fusion_candidates.csv").is_file())


if __name__ == "__main__":
    unittest.main()
