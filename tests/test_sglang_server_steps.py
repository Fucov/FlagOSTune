from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.tools.sglang_server_steps import (
    CaptureError,
    DecodeDetector,
    detect_log_flags,
    endpoint_metadata,
    parse_command_groups,
    prepare_capture_outputs,
    profile_request_body,
    start_profile,
    terminate_process_group,
    wait_for_decode,
    wait_for_profile_completion,
    wait_ready,
)


class SGLangServerStepsTest(unittest.TestCase):
    def test_prepare_capture_outputs_removes_stale_pass_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            logs = [root / "server.log", root / "nsys.log", root / "benchmark.log"]
            for log in logs:
                log.write_text("stale", encoding="utf-8")
            metadata = root / "capture.metadata.json"
            metadata.write_text('{"capture_status":"PASS"}', encoding="utf-8")

            prepare_capture_outputs(logs, metadata)

            self.assertFalse(metadata.exists())
            self.assertTrue(all(log.read_text() == "" for log in logs))

    def test_log_flags_require_specific_jit_and_moe_fallback_evidence(self):
        self.assertEqual(
            detect_log_flags("DeepGEMM JIT complete; MoE config fallback enabled"),
            {
                "deepgemm_jit_detected": True,
                "moe_config_fallback_detected": True,
            },
        )
        self.assertFalse(
            detect_log_flags("health endpoint fallback to v1/models")[
                "moe_config_fallback_detected"
            ]
        )

    def test_endpoint_metadata_records_host_port_and_visible_devices(self):
        self.assertEqual(
            endpoint_metadata(
                "http://127.0.0.1:30001",
                visible_devices="0,1,2,3",
            ),
            {
                "base_url": "http://127.0.0.1:30001",
                "host": "127.0.0.1",
                "port": 30001,
                "visible_devices": "0,1,2,3",
            },
        )

    def test_profile_request_body_uses_relative_step_zero_and_requested_count(self):
        self.assertEqual(
            profile_request_body(7),
            {
                "start_step": 0,
                "num_steps": 7,
                "activities": ["CUDA_PROFILER"],
            },
        )

    def test_profile_request_body_rejects_nonpositive_step_count(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            profile_request_body(0)

    def test_start_profile_http_error_is_fatal(self):
        with mock.patch(
            "scripts.tools.sglang_server_capture.http_json",
            return_value=(500, {"error": "profiler unavailable"}),
        ):
            with self.assertRaisesRegex(CaptureError, "HTTP 500"):
                start_profile("http://127.0.0.1:30001", 4)

    def test_start_profile_rejects_success_false_response(self):
        with mock.patch(
            "scripts.tools.sglang_server_capture.http_json",
            return_value=(200, {"success": False, "message": "profiler unavailable"}),
        ):
            with self.assertRaisesRegex(CaptureError, "rejected"):
                start_profile("http://127.0.0.1:30001", 4)

    def test_decode_detector_requires_decode_and_positive_running_requests(self):
        detector = DecodeDetector()
        self.assertFalse(detector.feed("Decode batch, running requests: 0"))
        self.assertTrue(detector.feed("Decode batch, running_reqs=4"))

    def test_wait_ready_falls_back_to_v1_models(self):
        calls = []

        def request(url, timeout):
            del timeout
            calls.append(url)
            return 200 if url.endswith("/v1/models") else 404

        endpoint = wait_ready(
            "http://127.0.0.1:30001",
            timeout=0.5,
            child_alive=lambda: None,
            request=request,
            poll_interval=0.001,
        )
        self.assertEqual(endpoint, "/v1/models")
        self.assertTrue(any(url.endswith("/health") for url in calls))

    def test_wait_ready_timeout_names_endpoints(self):
        with self.assertRaisesRegex(CaptureError, "readiness timeout"):
            wait_ready(
                "http://127.0.0.1:30001",
                timeout=0.01,
                child_alive=lambda: None,
                request=lambda _url, _timeout: 503,
                poll_interval=0.001,
            )

    def test_wait_for_decode_reads_log_evidence_without_fixed_sleep(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server.log"
            log.write_text(
                "running request(s): 2\nDecode batch size=2\n",
                encoding="utf-8",
            )
            evidence = wait_for_decode(
                log,
                timeout=0.2,
                child_alive=lambda: None,
                poll_interval=0.001,
            )
        self.assertIn("Decode batch", evidence)
        self.assertIn("running request", evidence)

    def test_wait_for_decode_times_out_without_decode_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server.log"
            log.write_text("prefill batch only\n", encoding="utf-8")
            with self.assertRaisesRegex(CaptureError, "decode evidence timeout"):
                wait_for_decode(
                    log,
                    timeout=0.01,
                    child_alive=lambda: None,
                    poll_interval=0.001,
                )

    def test_wait_for_decode_ignores_warmup_evidence_before_start_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server.log"
            log.write_text(
                "Decode batch, running requests: 2\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CaptureError, "decode evidence timeout"):
                wait_for_decode(
                    log,
                    timeout=0.01,
                    child_alive=lambda: None,
                    poll_interval=0.001,
                    start_offset=log.stat().st_size,
                )

    def test_wait_for_profile_completion_uses_automatic_stop_log_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server.log"
            log.write_text(
                "Stop profiling...\nProfiling done. Traces are saved.\n",
                encoding="utf-8",
            )
            evidence = wait_for_profile_completion(
                log,
                start_offset=0,
                timeout=0.2,
                child_alive=lambda: None,
                poll_interval=0.001,
            )
        self.assertIn("Profiling done", evidence)

    def test_command_groups_preserve_option_like_child_arguments(self):
        prefix, groups = parse_command_groups(
            [
                "--profile-phase", "decode",
                "--nsys-command", "nsys", "profile", "--trace=cuda",
                "--warmup-command", "python", "-m", "warmup", "--num-prompts", "2",
                "--benchmark-command", "python", "-m", "bench", "--num-prompts", "4",
            ]
        )
        self.assertEqual(prefix, ["--profile-phase", "decode"])
        self.assertEqual(groups["nsys"], ["nsys", "profile", "--trace=cuda"])
        self.assertEqual(groups["warmup"][-2:], ["--num-prompts", "2"])
        self.assertEqual(groups["benchmark"][-2:], ["--num-prompts", "4"])

    @unittest.skipIf(sys.platform == "win32", "process groups use POSIX signals")
    def test_terminate_process_group_stops_owned_child(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        terminate_process_group(process, grace_seconds=0.1)
        self.assertIsNotNone(process.poll())


if __name__ == "__main__":
    unittest.main()
