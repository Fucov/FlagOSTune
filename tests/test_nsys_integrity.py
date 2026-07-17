from __future__ import annotations

import unittest

from scripts.tools.nsys.evaluate_integrity import IntegrityInputs, evaluate_integrity


class NsysIntegrityTest(unittest.TestCase):
    def base(self, **changes):
        values = dict(
            report_size=4096,
            sqlite_size=8192,
            kernel_event_count=20,
            invalid_timestamp_count=0,
            requested_dependencies=False,
            requested_communication=False,
            event_trace_available=True,
            communication_capability=True,
            expected_tp=4,
            captured_devices=(0, 1, 2, 3),
            requested_phase="decode",
            detected_phase="DECODE",
            initialization_only=False,
            runtime_collective_count=8,
            capture_duration_seconds=1.0,
            benchmark_duration_seconds=1.0,
            kernel_time_ns=500_000_000,
            h2d_time_ns=1,
            memory_time_ns=100,
            largest_nvtx="Decode batch",
            capture_mode="server-steps",
            deepgemm_jit_detected=False,
        )
        values.update(changes)
        return IntegrityInputs(**values)

    def test_complete_decode_capture_passes_both_states(self):
        result = evaluate_integrity(self.base())
        self.assertEqual(result.raw_report_integrity, "PASS")
        self.assertEqual(result.analysis_completeness, "PASS")

    def test_requested_dependency_without_events_is_partial(self):
        result = evaluate_integrity(
            self.base(requested_dependencies=True, event_trace_available=False)
        )
        self.assertEqual(result.analysis_completeness, "PARTIAL")
        self.assertIn("dependency", " ".join(result.analysis_reasons).lower())

    def test_requested_communication_without_capability_is_partial(self):
        result = evaluate_integrity(
            self.base(requested_communication=True, communication_capability=False)
        )
        self.assertEqual(result.analysis_completeness, "PARTIAL")
        self.assertIn("communication", " ".join(result.analysis_reasons).lower())

    def test_tp4_missing_devices_cannot_pass(self):
        result = evaluate_integrity(self.base(captured_devices=()))
        self.assertEqual(result.analysis_completeness, "PARTIAL")
        self.assertIn("TP4", " ".join(result.analysis_reasons))

    def test_unknown_requested_phase_and_zero_collectives_cannot_pass(self):
        result = evaluate_integrity(
            self.base(
                requested_communication=True,
                detected_phase="UNKNOWN",
                runtime_collective_count=0,
            )
        )
        self.assertEqual(result.analysis_completeness, "PARTIAL")
        reasons = " ".join(result.analysis_reasons).lower()
        self.assertIn("phase", reasons)
        self.assertIn("collective", reasons)

    def test_empty_report_is_raw_failure(self):
        result = evaluate_integrity(self.base(report_size=0))
        self.assertEqual(result.raw_report_integrity, "FAIL")

    def test_decode_startup_and_h2d_dominance_are_suspicious(self):
        result = evaluate_integrity(
            self.base(
                initialization_only=True,
                largest_nvtx="ncclCommInitRank",
                h2d_time_ns=80,
                memory_time_ns=100,
            )
        )
        self.assertEqual(result.analysis_completeness, "PARTIAL")
        checks = " ".join(result.sanity_checks).lower()
        self.assertIn("initialization", checks)
        self.assertIn("h2d", checks)

    def test_duration_mismatch_and_tiny_kernel_window_are_suspicious(self):
        result = evaluate_integrity(
            self.base(
                capture_duration_seconds=10.0,
                benchmark_duration_seconds=1.0,
                kernel_time_ns=1_000,
            )
        )
        self.assertEqual(result.analysis_completeness, "PARTIAL")
        checks = " ".join(result.sanity_checks).lower()
        self.assertIn("duration", checks)
        self.assertIn("kernel", checks)

    def test_full_offline_deepgemm_sets_startup_contamination(self):
        result = evaluate_integrity(
            self.base(capture_mode="full-offline", deepgemm_jit_detected=True)
        )
        self.assertTrue(result.flags["startup_contaminated"])


if __name__ == "__main__":
    unittest.main()
