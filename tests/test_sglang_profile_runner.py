import tempfile
import unittest
from pathlib import Path

from scripts.tools.sglang_profile_runner import (
    build_sglang_command,
    get_scenarios,
    resolve_run_paths,
)


def sample_config(tmp_path: Path) -> dict:
    return {
        "current_run": {
            "scenario_type": "optimized",
            "torch_profile": True,
        },
        "model": {
            "path": "/models/DeepSeek-V4-Flash",
            "name": "DeepSeek-V4-Flash",
            "tokenizer_path": None,
            "tensor_parallel_size": 8,
        },
        "serve": {
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.85,
            "extra_args": "--disable-cuda-graph --disable-radix-cache",
        },
        "sglang": {
            "extra_args": "--mem-frac 0.82 --attention-backend flashinfer",
        },
        "benchmark": {
            "num_runs": 2,
            "scenarios": {
                "optimized": [
                    {
                        "name": "p32768d1024",
                        "input_len": 32768,
                        "output_len": 1024,
                        "concurrency": 2,
                    }
                ]
            },
        },
        "paths": {
            "log_dir": str(tmp_path / "logs"),
            "torch_output_dir": str(tmp_path / "torch-raw"),
        },
    }


class SGLangProfileRunnerTest(unittest.TestCase):
    def test_get_scenarios_uses_current_scenario_type(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scenarios = get_scenarios(sample_config(Path(tmp)))

        self.assertEqual(
            scenarios,
            [
                {
                    "name": "p32768d1024",
                    "input_len": 32768,
                    "output_len": 1024,
                    "concurrency": 2,
                }
            ],
        )

    def test_resolve_run_paths_adds_sglang_report_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = resolve_run_paths(sample_config(tmp_path))

            self.assertEqual(paths.log_dir, tmp_path / "logs")
            self.assertEqual(paths.profile_dir, tmp_path / "torch-raw" / "report-sglang")

    def test_build_sglang_command_profiles_offline_throughput(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = sample_config(Path(tmp))
            paths = resolve_run_paths(config)
            scenario = get_scenarios(config)[0]

            cmd = build_sglang_command(scenario, config, paths.profile_dir, profile=True)

        self.assertEqual(cmd[1:3], ["-m", "sglang.bench_offline_throughput"])
        self.assertTrue(cmd[0].endswith("python") or cmd[0].endswith("python3"))
        self.assertIn("--model-path", cmd)
        self.assertIn("/models/DeepSeek-V4-Flash", cmd)
        self.assertIn("--dataset-name", cmd)
        self.assertIn("random", cmd)
        self.assertIn("--random-input-len", cmd)
        self.assertIn("32768", cmd)
        self.assertIn("--random-output-len", cmd)
        self.assertIn("1024", cmd)
        self.assertIn("--num-prompts", cmd)
        self.assertIn("2", cmd)
        self.assertIn("--tp-size", cmd)
        self.assertIn("8", cmd)
        self.assertIn("--trust-remote-code", cmd)
        self.assertIn("--profile", cmd)
        self.assertIn("--mem-frac", cmd)
        self.assertIn("0.82", cmd)
        self.assertIn("--attention-backend", cmd)
        self.assertIn("flashinfer", cmd)

    def test_build_sglang_command_without_profile_omits_profile_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = sample_config(Path(tmp))
            paths = resolve_run_paths(config)
            scenario = get_scenarios(config)[0]

            cmd = build_sglang_command(scenario, config, paths.profile_dir, profile=False)

        self.assertNotIn("--profile", cmd)


if __name__ == "__main__":
    unittest.main()
