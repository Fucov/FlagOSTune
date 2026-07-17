from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path

from scripts.tools.sglang_server_capture import (
    CaptureError,
    parse_command_groups,
    profile_request_body,
    start_profile,
    stop_profile,
    terminate_process_group,
    wait_ready,
)


class SGLangServerCaptureTest(unittest.TestCase):
    def test_profile_request_body_uses_cuda_profiler_activity(self):
        self.assertEqual(
            profile_request_body(),
            {"activities": ["CUDA_PROFILER"]},
        )

    def test_start_profile_http_error_is_fatal(self):
        with mock.patch(
            "scripts.tools.sglang_server_capture.http_json",
            return_value=(500, {"error": "profiler unavailable"}),
        ):
            with self.assertRaisesRegex(CaptureError, "HTTP 500"):
                start_profile("http://127.0.0.1:30001")

    def test_stop_profile_uses_manual_stop_endpoint(self):
        with mock.patch(
            "scripts.tools.sglang_server_capture.http_json",
            return_value=(200, {"status": "ok"}),
        ) as request:
            result = stop_profile("http://127.0.0.1:30001")
        self.assertEqual(result["status"], 200)
        request.assert_called_once_with(
            "POST", "http://127.0.0.1:30001/stop_profile", None, 10.0
        )

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

    def test_command_groups_preserve_option_like_child_arguments(self):
        prefix, groups = parse_command_groups(
            [
                "--profile-warmup-prompts", "2",
                "--nsys-command", "nsys", "profile", "--trace=cuda",
                "--warmup-command", "python", "-m", "warmup", "--num-prompts", "2",
                "--benchmark-command", "python", "-m", "bench", "--num-prompts", "4",
            ]
        )
        self.assertEqual(prefix, ["--profile-warmup-prompts", "2"])
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
