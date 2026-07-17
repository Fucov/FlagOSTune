from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

from scripts.tools.sglang_server_steps import (
    CaptureError,
    DecodeDetector,
    parse_command_groups,
    profile_request_body,
    start_profile,
    terminate_process_group,
    wait_for_decode,
    wait_ready,
)


class SGLangServerStepsTest(unittest.TestCase):
    def test_profile_request_body_uses_cuda_profiler_activity(self):
        self.assertEqual(
            profile_request_body(7),
            {
                "start_step": 0,
                "num_steps": 7,
                "activities": ["CUDA_PROFILER"],
            },
        )

    def test_start_profile_http_error_is_fatal(self):
        with mock.patch(
            "scripts.tools.sglang_server_steps.http_json",
            return_value=(500, {"error": "profiler unavailable"}),
        ):
            with self.assertRaisesRegex(CaptureError, "HTTP 500"):
                start_profile("http://127.0.0.1:30001", 4)

    def test_decode_detector_requires_decode_and_positive_running_requests(self):
        detector = DecodeDetector()
        self.assertFalse(detector.feed("running requests: 0"))
        self.assertFalse(detector.feed("running request(s): 3"))
        self.assertTrue(detector.feed("Decode batch size=3"))

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

    def test_wait_for_decode_reads_incrementally(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server.log"
            log.write_text("running request(s): 2\nDecode batch size=2\n", encoding="utf-8")
            evidence = wait_for_decode(
                log,
                timeout=0.2,
                child_alive=lambda: None,
                poll_interval=0.001,
            )
        self.assertIn("Decode batch", evidence)
        self.assertIn("running request", evidence)

    def test_wait_for_decode_times_out_without_guessing(self):
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
