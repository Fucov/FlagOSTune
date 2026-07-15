import sys
import unittest

from scripts.tools.cuda_profiler_launcher import run_module_with_cuda_profiler


class FakeCudart:
    def __init__(self, start_result=0, stop_result=0):
        self.calls = []
        self.start_result = start_result
        self.stop_result = stop_result

    def cudaProfilerStart(self):
        self.calls.append("start")
        return self.start_result

    def cudaProfilerStop(self):
        self.calls.append("stop")
        return self.stop_result


class CudaProfilerLauncherTest(unittest.TestCase):
    def test_module_runs_inside_cuda_profiler_range(self):
        cudart = FakeCudart()
        seen = []

        def module_runner(module, run_name):
            seen.append((module, run_name, list(sys.argv)))

        run_module_with_cuda_profiler(
            "sglang.bench_offline_throughput",
            ["--model-path", "/models/qwen"],
            cudart=cudart,
            module_runner=module_runner,
        )

        self.assertEqual(cudart.calls, ["start", "stop"])
        self.assertEqual(
            seen,
            [
                (
                    "sglang.bench_offline_throughput",
                    "__main__",
                    [
                        "sglang.bench_offline_throughput",
                        "--model-path",
                        "/models/qwen",
                    ],
                )
            ],
        )

    def test_profiler_stops_and_argv_is_restored_when_module_fails(self):
        cudart = FakeCudart()
        original_argv = list(sys.argv)

        def fail(_module, _run_name):
            raise RuntimeError("boom")

        with self.assertRaisesRegex(RuntimeError, "boom"):
            run_module_with_cuda_profiler(
                "target",
                [],
                cudart=cudart,
                module_runner=fail,
            )

        self.assertEqual(cudart.calls, ["start", "stop"])
        self.assertEqual(sys.argv, original_argv)

    def test_nonzero_start_result_is_rejected_without_stop(self):
        cudart = FakeCudart(start_result=7)

        with self.assertRaisesRegex(RuntimeError, "cudaProfilerStart failed: 7"):
            run_module_with_cuda_profiler("target", [], cudart=cudart)

        self.assertEqual(cudart.calls, ["start"])

    def test_nonzero_stop_result_is_rejected_after_success(self):
        cudart = FakeCudart(stop_result=8)

        with self.assertRaisesRegex(RuntimeError, "cudaProfilerStop failed: 8"):
            run_module_with_cuda_profiler(
                "target",
                [],
                cudart=cudart,
                module_runner=lambda _module, _run_name: None,
            )

        self.assertEqual(cudart.calls, ["start", "stop"])


if __name__ == "__main__":
    unittest.main()
