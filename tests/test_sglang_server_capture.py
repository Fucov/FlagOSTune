from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.tools import sglang_server_capture as capture_module
from scripts.tools.sglang_server_capture import (
    CaptureError,
    detect_log_flags,
    endpoint_metadata,
    parse_command_groups,
    prepare_capture_outputs,
    profile_request_body,
    run_capture,
    start_profile,
    stop_profile,
    terminate_process_group,
    wait_ready,
)


ROOT = Path(__file__).resolve().parent.parent
CAPTURE_SCRIPT = ROOT / "scripts" / "tools" / "sglang_server_capture.py"


class FakeProcess:
    pid = 12345

    def __init__(self) -> None:
        self.returncode = None

    def poll(self):
        return self.returncode


class SGLangServerCaptureTest(unittest.TestCase):
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

    def test_profile_request_body_covers_until_manual_stop(self):
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

    def test_start_profile_rejects_success_false_response(self):
        with mock.patch(
            "scripts.tools.sglang_server_capture.http_json",
            return_value=(200, {"success": False}),
        ):
            with self.assertRaisesRegex(CaptureError, "rejected"):
                start_profile("http://127.0.0.1:30001")

    def test_stop_profile_rejects_success_false_response(self):
        with mock.patch(
            "scripts.tools.sglang_server_capture.http_json",
            return_value=(200, {"success": False}),
        ):
            with self.assertRaisesRegex(CaptureError, "rejected"):
                stop_profile("http://127.0.0.1:30001")

    def test_flush_cache_rejects_success_false_response(self):
        flush_cache = getattr(capture_module, "flush_cache")
        with mock.patch(
            "scripts.tools.sglang_server_capture.http_json",
            return_value=(200, {"success": False}),
        ):
            with self.assertRaisesRegex(CaptureError, "rejected"):
                flush_cache("http://127.0.0.1:30001")

    def test_throughput_uses_the_captured_final_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "benchmark.log"
            log.write_text(
                "Request throughput: 1.25 requests/s\n"
                "Request throughput: 2.50 requests/s\n",
                encoding="utf-8",
            )
            value = capture_module._parse_throughput(log)
        self.assertEqual(value, 2.5)

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
                "--total-runs",
                "2",
                "--nsys-command",
                "nsys",
                "profile",
                "--trace=cuda",
                "--benchmark-command",
                "python",
                "-m",
                "bench",
                "--num-prompts",
                "64",
            ]
        )
        self.assertEqual(prefix, ["--total-runs", "2"])
        self.assertEqual(groups["nsys"], ["nsys", "profile", "--trace=cuda"])
        self.assertEqual(groups["benchmark"][-2:], ["--num-prompts", "64"])

    def _capture_args(self, root: Path) -> argparse.Namespace:
        report = root / "capture.nsys-rep"
        return argparse.Namespace(
            output_prefix=str(root / "capture"),
            report=str(report),
            metadata=str(report) + ".metadata.json",
            server_log=str(root / "server.log"),
            nsys_log=str(root / "nsys.log"),
            benchmark_log=str(root / "benchmark.log"),
            base_url="http://127.0.0.1:30001",
            total_runs=2,
            concurrency=64,
            profile_ready_timeout=30.0,
            cuda_graph_enabled=False,
            cuda_graph_trace="node",
            layerwise_nvtx_enabled=True,
            project_root=str(ROOT),
            workflow_script=str(ROOT / "scripts" / "sglang-nsys-workflow.sh"),
            parser_script=str(ROOT / "scripts" / "tools" / "parse_nsys.py"),
            model="Qwen3.6-35B-A3B-FP8-TP4-P32768D1024C64",
            model_path="/models/Qwen3.6-35B-A3B-FP8",
            tokenizer_path="/models/Qwen3.6-35B-A3B-FP8",
            scenario="p32768d1024_c64",
            dataset="sharegpt",
            num_prompts=64,
            input_tokens=32768,
            output_tokens=1024,
            tp_size=4,
        )

    def test_capture_orders_warmup_flush_start_measured_stop_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = self._capture_args(root)
            report = Path(args.report)
            events = []
            control_timeouts = {}

            def fake_http(_method, url, _body, _timeout):
                if url.endswith("/flush_cache"):
                    events.append("flush")
                    control_timeouts["flush"] = _timeout
                elif url.endswith("/start_profile"):
                    events.append("start")
                    control_timeouts["start"] = _timeout
                elif url.endswith("/stop_profile"):
                    events.append("stop")
                    control_timeouts["stop"] = _timeout
                return 200, {"success": True}

            def fake_run(_command, _log_path, label, **_kwargs):
                events.append(label)

            def fake_terminate(_process, grace_seconds=10.0):
                del grace_seconds
                report.write_bytes(b"nsys-report")

            patches = (
                mock.patch(
                    "scripts.tools.sglang_server_capture.subprocess.Popen",
                    return_value=FakeProcess(),
                ),
                mock.patch(
                    "scripts.tools.sglang_server_capture.wait_ready",
                    return_value="/health",
                ),
                mock.patch(
                    "scripts.tools.sglang_server_capture.http_json",
                    side_effect=fake_http,
                ),
                mock.patch(
                    "scripts.tools.sglang_server_capture._run_logged",
                    side_effect=fake_run,
                ),
                mock.patch(
                    "scripts.tools.sglang_server_capture.terminate_process_group",
                    side_effect=fake_terminate,
                ),
                mock.patch(
                    "scripts.tools.sglang_server_capture._nsys_version",
                    return_value="Nsight Systems 2025.3.1",
                ),
                mock.patch(
                    "scripts.tools.sglang_server_capture._git_identity",
                    return_value=("deadbeef", True),
                ),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
                metadata = run_capture(
                    args,
                    {"nsys": ["nsys", "profile"], "benchmark": ["bench"]},
                )

            self.assertEqual(
                events[:5],
                ["warmup-1", "flush", "start", "benchmark-2", "stop"],
            )
            self.assertEqual(metadata["capture_mode"], "server-full")
            self.assertEqual(metadata["capture_scope"], "measured_inference")
            self.assertEqual(metadata["inference_scope"], "prefill_and_decode")
            self.assertEqual(metadata["total_runs"], 2)
            self.assertEqual(metadata["warmup_runs"], 1)
            self.assertEqual(metadata["captured_run"], 2)
            self.assertEqual(metadata["num_prompts"], 64)
            self.assertEqual(metadata["concurrency"], 64)
            self.assertEqual(
                control_timeouts["stop"], args.profile_ready_timeout
            )
            stored = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
            self.assertEqual(stored["capture_status"], "PASS")

    def test_failed_warmup_does_not_flush_or_start_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._capture_args(Path(tmp))
            flush = mock.Mock()
            start = mock.Mock()
            with mock.patch(
                "scripts.tools.sglang_server_capture.subprocess.Popen",
                return_value=FakeProcess(),
            ), mock.patch(
                "scripts.tools.sglang_server_capture.wait_ready",
                return_value="/health",
            ), mock.patch(
                "scripts.tools.sglang_server_capture._run_logged",
                side_effect=CaptureError("warmup-1 command failed with exit 7"),
            ), mock.patch(
                "scripts.tools.sglang_server_capture.flush_cache", flush
            ), mock.patch(
                "scripts.tools.sglang_server_capture.start_profile", start
            ), mock.patch(
                "scripts.tools.sglang_server_capture.terminate_process_group"
            ):
                with self.assertRaisesRegex(CaptureError, "warmup-1"):
                    run_capture(
                        args,
                        {"nsys": ["nsys"], "benchmark": ["bench"]},
                    )
            flush.assert_not_called()
            start.assert_not_called()
            self.assertFalse(Path(args.metadata).exists())

    def test_failed_cache_flush_does_not_start_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._capture_args(Path(tmp))
            start = mock.Mock()
            with mock.patch(
                "scripts.tools.sglang_server_capture.subprocess.Popen",
                return_value=FakeProcess(),
            ), mock.patch(
                "scripts.tools.sglang_server_capture.wait_ready",
                return_value="/health",
            ), mock.patch(
                "scripts.tools.sglang_server_capture._run_logged"
            ), mock.patch(
                "scripts.tools.sglang_server_capture.flush_cache",
                side_effect=CaptureError("/flush_cache returned HTTP 500"),
            ), mock.patch(
                "scripts.tools.sglang_server_capture.start_profile", start
            ), mock.patch(
                "scripts.tools.sglang_server_capture.terminate_process_group"
            ):
                with self.assertRaisesRegex(CaptureError, "flush_cache"):
                    run_capture(
                        args,
                        {"nsys": ["nsys"], "benchmark": ["bench"]},
                    )
            start.assert_not_called()
            self.assertFalse(Path(args.metadata).exists())

    def test_failed_measured_run_stops_profile_and_writes_no_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._capture_args(Path(tmp))
            stop = mock.Mock(return_value={"status": 200, "response": {}})
            with mock.patch(
                "scripts.tools.sglang_server_capture.subprocess.Popen",
                return_value=FakeProcess(),
            ), mock.patch(
                "scripts.tools.sglang_server_capture.wait_ready",
                return_value="/health",
            ), mock.patch(
                "scripts.tools.sglang_server_capture._run_logged",
                side_effect=[None, CaptureError("benchmark-2 command failed")],
            ), mock.patch(
                "scripts.tools.sglang_server_capture.flush_cache",
                return_value={"status": 200, "response": {}},
            ), mock.patch(
                "scripts.tools.sglang_server_capture.start_profile",
                return_value={"request": profile_request_body(), "status": 200},
            ), mock.patch(
                "scripts.tools.sglang_server_capture.stop_profile", stop
            ), mock.patch(
                "scripts.tools.sglang_server_capture.terminate_process_group"
            ):
                with self.assertRaisesRegex(CaptureError, "benchmark-2"):
                    run_capture(
                        args,
                        {"nsys": ["nsys"], "benchmark": ["bench"]},
                    )
            stop.assert_called_once_with(
                args.base_url, timeout=args.profile_ready_timeout
            )
            self.assertFalse(Path(args.metadata).exists())

    def test_empty_report_after_normal_stop_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self._capture_args(Path(tmp))
            args.total_runs = 1
            with mock.patch(
                "scripts.tools.sglang_server_capture.subprocess.Popen",
                return_value=FakeProcess(),
            ), mock.patch(
                "scripts.tools.sglang_server_capture.wait_ready",
                return_value="/health",
            ), mock.patch(
                "scripts.tools.sglang_server_capture._run_logged"
            ), mock.patch(
                "scripts.tools.sglang_server_capture.flush_cache",
                return_value={"status": 200, "response": {}},
            ), mock.patch(
                "scripts.tools.sglang_server_capture.start_profile",
                return_value={"request": profile_request_body(), "status": 200},
            ), mock.patch(
                "scripts.tools.sglang_server_capture.stop_profile",
                return_value={"status": 200, "response": {}},
            ), mock.patch(
                "scripts.tools.sglang_server_capture.terminate_process_group"
            ):
                with self.assertRaisesRegex(CaptureError, "missing or empty"):
                    run_capture(
                        args,
                        {"nsys": ["nsys"], "benchmark": ["bench"]},
                    )
            self.assertFalse(Path(args.metadata).exists())

    def test_absolute_exec_server_entrypoint_does_not_require_repo_on_pythonpath(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(CAPTURE_SCRIPT), "exec-server", "--help"],
                cwd=tmp,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("exec-server", result.stdout)

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
