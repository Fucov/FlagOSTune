from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.tools.sglang_collect_metadata import collect_metadata


class SGLangCollectMetadataTest(unittest.TestCase):
    def test_collect_metadata_wraps_values_with_sources_and_trace_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml.TestModel"
            config_path.write_text(
                """
model:
  name: TestModel
  path: /models/test
  tokenizer_path: /models/test-tokenizer
  tensor_parallel_size: 4
runtime:
  dtype: bfloat16
sglang:
  context_length: 4096
  extra_args: "--attention-backend triton --max-running-requests 1"
benchmark:
  num_runs: 2
  scenarios:
    optimized:
      - name: p128d16_c1
        input_len: 128
        output_len: 16
        concurrency: 1
""",
                encoding="utf-8",
            )
            trace_dir = root / "results" / "TestModel" / "sglang-torch-raw" / "report-sglang"
            trace_dir.mkdir(parents=True)
            trace_path = trace_dir / "worker-rank0.pt.trace.json"
            trace_path.write_text('{"traceEvents":[]}', encoding="utf-8")
            metadata_dir = root / "results" / "TestModel" / "sglang-run-metadata"
            report_dir = root / "reports" / "TestModel"

            metadata = collect_metadata(
                model_name="TestModel",
                config_path=config_path,
                output_dir=metadata_dir,
                report_dir=report_dir,
                trace_dir=trace_dir,
                selected_rank="0",
                phase="after_profile",
                workflow_command="./scripts/sglang-auto-workflow.sh --model TestModel --torch",
                processing_command="./scripts/sglang-auto-processing.sh --model TestModel --workflow torch",
                nvidia_smi_text="GPU 0 test",
            )

            self.assertEqual(metadata["model"]["model_name"]["value"], "TestModel")
            self.assertEqual(metadata["model"]["model_name"]["source"], str(config_path))
            self.assertEqual(metadata["benchmark"]["scenario_name"]["value"], "p128d16_c1")
            self.assertEqual(metadata["trace"]["selected_rank"]["value"], "0")
            self.assertEqual(metadata["trace"]["trace_files"]["value"][0]["path"], str(trace_path))
            self.assertTrue((report_dir / "run_metadata.json").exists())
            saved = json.loads((metadata_dir / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["trace"]["trace_files"]["value"][0]["size_bytes"], trace_path.stat().st_size)
            self.assertTrue((metadata_dir / "nvidia_smi_after_profile.txt").exists())

    def test_collect_metadata_records_profile_detail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml.TestModel"
            config_path.write_text(
                "model:\n  name: TestModel\nbenchmark:\n  scenarios:\n    optimized:\n      - name: smoke\n        input_len: 1\n        output_len: 1\n        concurrency: 1\n",
                encoding="utf-8",
            )
            old = os.environ.get("SGLANG_TORCH_PROFILER_DETAIL")
            os.environ["SGLANG_TORCH_PROFILER_DETAIL"] = "full_stack"
            try:
                metadata = collect_metadata(
                    model_name="TestModel",
                    config_path=config_path,
                    output_dir=root / "metadata",
                    report_dir=root / "report",
                    trace_dir=None,
                    selected_rank="0",
                    phase="after_profile",
                    workflow_command="test",
                    processing_command="test",
                    nvidia_smi_text="",
                )
            finally:
                if old is None:
                    os.environ.pop("SGLANG_TORCH_PROFILER_DETAIL", None)
                else:
                    os.environ["SGLANG_TORCH_PROFILER_DETAIL"] = old
        self.assertEqual(metadata["trace"]["profiler_config"]["detail"], "full_stack")


if __name__ == "__main__":
    unittest.main()
