import os
import json
import subprocess
import tempfile
import textwrap
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / "scripts" / "sglang-nsys-workflow.sh"


FAKE_YQ = r'''#!/usr/bin/env python3
import json
import re
import sys

args = sys.argv[1:]
if args and args[0] == "-r":
    args.pop(0)
expression, config_path = args
with open(config_path, encoding="utf-8") as handle:
    data = json.load(handle)

def resolve(path):
    value = data
    for name, index in re.findall(r"\.([A-Za-z0-9_]+)(?:\[([0-9]+)\])?", path):
        if not isinstance(value, dict) or name not in value:
            return None
        value = value[name]
        if index:
            if not isinstance(value, list) or int(index) >= len(value):
                return None
            value = value[int(index)]
    return value

if "| length" in expression:
    value = resolve(expression.split("|", 1)[0].strip())
    value = len(value) if isinstance(value, (list, dict, str)) else 0
else:
    value = None
    for part in expression.split(" // "):
        part = part.strip()
        if part.startswith("."):
            candidate = resolve(part)
        elif part.startswith(('"', "'")):
            candidate = json.loads(part.replace("'", '"'))
        elif part == "true":
            candidate = True
        elif part == "false":
            candidate = False
        elif part == "null":
            candidate = None
        else:
            try:
                candidate = int(part)
            except ValueError:
                candidate = part
        if candidate is not None:
            value = candidate
            break

if isinstance(value, bool):
    print(str(value).lower())
elif value is None:
    print("null")
elif isinstance(value, (dict, list)):
    print(json.dumps(value))
else:
    print(value)
'''


def make_config(model_name, model_path, tp, scenarios=None):
    return {
        "model": {
            "name": model_name,
            "path": model_path,
            "tokenizer_path": None,
            "tensor_parallel_size": tp,
        },
        "serve": {"trust_remote_code": True},
        "sglang": {
            "dtype": "bfloat16",
            "mem_fraction_static": 0.75,
            "context_length": 4096,
            "load_format": "auto",
            "extra_args": "--disable-cuda-graph --sampling-backend pytorch",
        },
        "benchmark": {
            "dataset_name": "random",
            "dataset_path": "/datasets/local.json",
            "scenarios": {
                "optimized": scenarios
                or [
                    {
                        "name": "p128d16_c1",
                        "input_len": 128,
                        "output_len": 16,
                        "concurrency": 1,
                    }
                ]
            },
        },
    }


class SGLangNsysWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.bin_dir = Path(self.temp_dir.name) / "bin"
        self.bin_dir.mkdir()
        yq = self.bin_dir / "yq"
        yq.write_text(FAKE_YQ, encoding="utf-8")
        yq.chmod(0o755)
        nsys = self.bin_dir / "nsys"
        nsys.write_text(
            "#!/usr/bin/env python3\n"
            "import pathlib,sys\n"
            "a=sys.argv[1:]\n"
            "if '--version' in a:\n"
            " print('NVIDIA Nsight Systems version 2025.3.1')\n"
            " raise SystemExit(0)\n"
            "p=a[a.index('--output')+1]\n"
            "pathlib.Path(p+'.nsys-rep').write_bytes(b'fake-report')\n"
            "print('fake nsys capture output')\n",
            encoding="utf-8",
        )
        nsys.chmod(0o755)
        self.config_paths = []

    def tearDown(self):
        for path in self.config_paths:
            path.unlink(missing_ok=True)
        self.temp_dir.cleanup()

    def write_config(self, config):
        suffix = f"NsysTest-{uuid.uuid4().hex}"
        path = ROOT / f"config.yaml.{suffix}"
        path.write_text(json.dumps(config), encoding="utf-8")
        self.config_paths.append(path)
        return suffix

    def run_workflow(self, suffix, *args):
        env = os.environ.copy()
        env["PATH"] = f"{self.bin_dir}{os.pathsep}{env['PATH']}"
        return subprocess.run(
            ["bash", str(WORKFLOW), "--model", suffix, *args],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_qwen_dry_run_has_required_nsys_flags_and_no_torch_profile(self):
        model_name = "Qwen3.6-35B-A3B-FP8-TP4-Test"
        suffix = self.write_config(
            make_config(model_name, "/models/Qwen3.6-35B-A3B-FP8", 4)
        )

        result = self.run_workflow(
            suffix,
            "--nsys",
            "--dry-run",
            "--nsys-output",
            "capture",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--trace=cuda,nvtx,osrt", result.stdout)
        self.assertIn("--sample=none", result.stdout)
        self.assertIn("--cpuctxsw=none", result.stdout)
        self.assertIn("--capture-range=cudaProfilerApi", result.stdout)
        self.assertIn("--capture-range-end=stop", result.stdout)
        self.assertIn("--trace-fork-before-exec=true", result.stdout)
        self.assertNotIn(" --profile ", result.stdout)
        self.assertNotIn("SGLANG_TORCH_PROFILER", result.stdout)
        self.assertIn(
            f"results/{model_name}/nsys/capture.nsys-rep",
            result.stdout,
        )
        self.assertIn("--random-input-len 128", result.stdout)
        self.assertIn("--random-output-len 16", result.stdout)
        self.assertIn("--tp-size 4", result.stdout)

    def test_deepseek_tp8_is_supported(self):
        model_name = "DeepSeek-V4-Flash-FP8-TP8-Test"
        suffix = self.write_config(
            make_config(model_name, "/models/DeepSeek-V4-Flash-FP8", 8)
        )

        result = self.run_workflow(suffix, "--nsys", "--dry-run")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--tp-size 8", result.stdout)
        self.assertIn(f"results/{model_name}/nsys/", result.stdout)
        self.assertIn("p128d16_c1", result.stdout)

    def test_deepseek_tp_mismatch_is_rejected(self):
        suffix = self.write_config(
            make_config(
                "DeepSeek-V4-Flash-FP8-TP8-Test",
                "/models/DeepSeek-V4-Flash-FP8",
                4,
            )
        )

        result = self.run_workflow(suffix, "--nsys", "--dry-run")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TP8", result.stderr)

    def test_nsys_flag_is_required(self):
        suffix = self.write_config(
            make_config(
                "Qwen3.6-35B-A3B-FP8-TP4-Test",
                "/models/Qwen3.6-35B-A3B-FP8",
                4,
            )
        )

        result = self.run_workflow(suffix, "--dry-run")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--nsys", result.stderr)

    def test_multiple_scenarios_append_names_to_explicit_prefix(self):
        scenarios = [
            {"name": "prefill", "input_len": 128, "output_len": 1, "concurrency": 1},
            {"name": "decode", "input_len": 32, "output_len": 16, "concurrency": 2},
        ]
        model_name = "Qwen3.6-35B-A3B-FP8-TP4-Multi"
        suffix = self.write_config(
            make_config(
                model_name,
                "/models/Qwen3.6-35B-A3B-FP8",
                4,
                scenarios,
            )
        )

        result = self.run_workflow(
            suffix,
            "--nsys",
            "--dry-run",
            "--nsys-output",
            "custom/report.nsys-rep",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("custom/report-prefill.nsys-rep", result.stdout)
        self.assertIn("custom/report-decode.nsys-rep", result.stdout)
        self.assertNotIn(".nsys-rep-prefill", result.stdout)

    def test_missing_scenario_group_is_rejected(self):
        suffix = self.write_config(
            make_config(
                "Qwen3.6-35B-A3B-FP8-TP4-Test",
                "/models/Qwen3.6-35B-A3B-FP8",
                4,
            )
        )

        result = self.run_workflow(
            suffix,
            "--nsys",
            "--dry-run",
            "--scenario",
            "full",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("full", result.stderr)

    def test_unsupported_model_is_rejected(self):
        suffix = self.write_config(make_config("OtherModel", "/models/other", 4))

        result = self.run_workflow(suffix, "--nsys", "--dry-run")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("不支持", result.stderr)

    def test_torch_profile_flag_in_extra_args_is_rejected(self):
        config = make_config(
            "Qwen3.6-35B-A3B-FP8-TP4-Test",
            "/models/Qwen3.6-35B-A3B-FP8",
            4,
        )
        config["sglang"]["extra_args"] = "--profile --disable-cuda-graph"
        suffix = self.write_config(config)

        result = self.run_workflow(suffix, "--nsys", "--dry-run")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--profile", result.stderr)
        self.assertIn("Torch Profiler", result.stderr)

    def test_help_does_not_require_dependencies_or_model(self):
        result = subprocess.run(
            ["bash", str(WORKFLOW), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--nsys-output", result.stdout)
        self.assertIn("--scenario", result.stdout)
        self.assertIn("--parse-top", result.stdout)
        self.assertIn("--analyze-dependencies", result.stdout)

    def test_parse_and_dependency_options_are_forwarded_in_dry_run(self):
        suffix = self.write_config(
            make_config(
                "Qwen3.6-35B-A3B-FP8-TP4-Test",
                "/models/Qwen3.6-35B-A3B-FP8",
                4,
            )
        )
        output_dir = Path(self.temp_dir.name) / "summary"
        result = self.run_workflow(
            suffix,
            "--nsys",
            "--dry-run",
            "--parse",
            "--parse-top",
            "7",
            "--parse-output-dir",
            str(output_dir),
            "--force-parse-export",
            "--analyze-dependencies",
            "--analyze-communication",
            "--dependency-trace",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--cuda-event-trace=true", result.stdout)
        self.assertIn("parse_nsys.py", result.stdout)
        self.assertIn("--top 7", result.stdout)
        self.assertIn("--force-export", result.stdout)
        self.assertIn("--analyze-dependencies", result.stdout)
        self.assertIn("--analyze-communication", result.stdout)
        self.assertIn(str(output_dir), result.stdout)

    def test_dependency_trace_is_off_by_default(self):
        suffix = self.write_config(
            make_config(
                "Qwen3.6-35B-A3B-FP8-TP4-Test",
                "/models/Qwen3.6-35B-A3B-FP8",
                4,
            )
        )
        result = self.run_workflow(suffix, "--nsys", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("--cuda-event-trace=true", result.stdout)

    def test_capture_writes_log_metadata_and_report_size(self):
        model_name = "Qwen3.6-35B-A3B-FP8-TP4-Test"
        suffix = self.write_config(
            make_config(model_name, "/models/Qwen3.6-35B-A3B-FP8", 4)
        )
        prefix = Path(self.temp_dir.name) / "capture"
        result = self.run_workflow(
            suffix,
            "--nsys",
            "--nsys-output",
            str(prefix),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        report = Path(str(prefix) + ".nsys-rep")
        metadata = Path(str(report) + ".metadata.json")
        log = Path(str(prefix) + ".nsys.log")
        self.assertTrue(report.is_file())
        self.assertTrue(metadata.is_file())
        self.assertTrue(log.is_file())
        self.assertIn("fake nsys capture output", log.read_text())
        value = json.loads(metadata.read_text())
        self.assertEqual(value["tp_size"], 4)
        self.assertEqual(value["capture_mode"], "full-offline")
        self.assertEqual(value["capture_scope"], "startup_and_full_process")
        self.assertEqual(value["profile_phase"], "full_process")
        self.assertFalse(value["steady_state_guaranteed"])
        self.assertIn("git_commit", value)
        self.assertIn("git_dirty", value)
        self.assertRegex(value["workflow_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(value["parser_sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("2025.3.1", value["nsys_version"])
        self.assertIn("report size", result.stdout.lower())

    def test_server_steps_prefill_dry_run_builds_server_and_client(self):
        suffix = self.write_config(
            make_config(
                "Qwen3.6-35B-A3B-FP8-TP4-Test",
                "/models/Qwen3.6-35B-A3B-FP8",
                4,
            )
        )
        result = self.run_workflow(
            suffix,
            "--nsys",
            "--dry-run",
            "--capture-mode",
            "server-steps",
            "--profile-phase",
            "prefill",
            "--profile-start-step",
            "0",
            "--profile-num-steps",
            "4",
            "--profile-warmup-prompts",
            "2",
            "--profile-concurrency",
            "3",
            "--profile-ready-timeout",
            "30",
            "--cuda-graph-trace",
            "node",
            "--layerwise-nvtx",
            "auto",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sglang_server_steps.py", result.stdout)
        self.assertIn("sglang.launch_server", result.stdout)
        self.assertIn("sglang.bench_serving", result.stdout)
        self.assertIn("--trace-fork-before-exec=true", result.stdout)
        self.assertIn("--capture-range=cudaProfilerApi", result.stdout)
        self.assertIn("--capture-range-end=stop", result.stdout)
        self.assertIn("--cuda-graph-trace=node", result.stdout)
        self.assertIn("--profile-phase prefill", result.stdout)
        self.assertIn("--profile-num-steps 4", result.stdout)
        self.assertIn("--profile-warmup-prompts 2", result.stdout)
        self.assertIn("--max-concurrency 3", result.stdout)

    def test_server_steps_decode_dry_run_is_log_driven(self):
        suffix = self.write_config(
            make_config(
                "Qwen3.6-35B-A3B-FP8-TP4-Test",
                "/models/Qwen3.6-35B-A3B-FP8",
                4,
            )
        )
        result = self.run_workflow(
            suffix,
            "--nsys",
            "--dry-run",
            "--capture-mode",
            "server-steps",
            "--profile-phase",
            "decode",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--profile-phase decode", result.stdout)
        self.assertIn("--decode-log-pattern", result.stdout)
        self.assertNotIn("sleep-before-profile", result.stdout)

    def test_new_profile_options_validate_enums_and_integers(self):
        suffix = self.write_config(
            make_config(
                "Qwen3.6-35B-A3B-FP8-TP4-Test",
                "/models/Qwen3.6-35B-A3B-FP8",
                4,
            )
        )
        cases = (
            (("--profile-phase", "wrong"), "--profile-phase"),
            (("--profile-num-steps", "0"), "--profile-num-steps"),
            (("--profile-start-step", "-1"), "--profile-start-step"),
            (("--cuda-graph-trace", "invalid"), "--cuda-graph-trace"),
            (("--layerwise-nvtx", "maybe"), "--layerwise-nvtx"),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                result = self.run_workflow(
                    suffix,
                    "--nsys",
                    "--dry-run",
                    "--capture-mode",
                    "server-steps",
                    *arguments,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)

    def test_full_offline_rejects_decode_label(self):
        suffix = self.write_config(
            make_config(
                "Qwen3.6-35B-A3B-FP8-TP4-Test",
                "/models/Qwen3.6-35B-A3B-FP8",
                4,
            )
        )
        result = self.run_workflow(
            suffix,
            "--nsys",
            "--dry-run",
            "--capture-mode",
            "full-offline",
            "--profile-phase",
            "decode",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("full-offline", result.stderr)
        self.assertIn("decode", result.stderr)

    def test_full_offline_accepts_startup_but_keeps_full_process_scope(self):
        suffix = self.write_config(
            make_config(
                "Qwen3.6-35B-A3B-FP8-TP4-Test",
                "/models/Qwen3.6-35B-A3B-FP8",
                4,
            )
        )
        result = self.run_workflow(
            suffix,
            "--nsys",
            "--dry-run",
            "--capture-mode",
            "full-offline",
            "--profile-phase",
            "startup",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sglang.bench_offline_throughput", result.stdout)

    def test_help_lists_server_step_options(self):
        result = subprocess.run(
            ["bash", str(WORKFLOW), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for option in (
            "--profile-phase",
            "--profile-start-step",
            "--profile-num-steps",
            "--profile-warmup-prompts",
            "--profile-concurrency",
            "--profile-ready-timeout",
            "--cuda-graph-trace",
            "--layerwise-nvtx",
        ):
            self.assertIn(option, result.stdout)

    def test_torch_profiler_cli_flag_has_clear_mutual_exclusion_error(self):
        suffix = self.write_config(
            make_config(
                "Qwen3.6-35B-A3B-FP8-TP4-Test",
                "/models/Qwen3.6-35B-A3B-FP8",
                4,
            )
        )
        result = self.run_workflow(suffix, "--nsys", "--profile")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Torch Profiler", result.stderr)
        self.assertIn("Nsight", result.stderr)


if __name__ == "__main__":
    unittest.main()
