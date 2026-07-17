from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.tools.nsys.sqlite_events import (
    inspect_schema,
    load_device_metadata,
    load_cuda_api_events,
    load_kernel_events,
    load_memory_events,
    load_nvtx_ranges,
    write_event_artifacts,
    write_kernel_summary_from_events,
)


class NsysSqliteEventsTest(unittest.TestCase):
    def make_database(self, root: Path) -> Path:
        path = root / "capture.sqlite"
        connection = sqlite3.connect(str(path))
        connection.executescript(
            """
            create table StringIds(id integer primary key, value text);
            insert into StringIds values(1, 'deep_gemm::sm90_fp8_gemm');
            insert into StringIds values(2, 'all_reduce_two_shot_kernel');
            insert into StringIds values(3, 'cudaLaunchKernel');
            insert into StringIds values(4, 'Decode batch');
            create table TARGET_INFO_GPU(deviceId integer, deviceName text);
            insert into TARGET_INFO_GPU values(0, 'GPU-A');
            insert into TARGET_INFO_GPU values(1, 'GPU-B');
            create table CUPTI_ACTIVITY_KIND_KERNEL(
                startNs integer, endNs integer, device integer, context integer,
                stream integer, processId integer, threadId integer,
                correlation integer, nameId integer
            );
            insert into CUPTI_ACTIVITY_KIND_KERNEL values(10, 30, 0, 2, 7, 100, 101, 55, 1);
            insert into CUPTI_ACTIVITY_KIND_KERNEL values(31, 51, 1, 3, 8, 200, 201, 56, 2);
            create table CUPTI_ACTIVITY_KIND_MEMCPY(
                start integer, end integer, deviceId integer, copyKind integer, bytes integer
            );
            insert into CUPTI_ACTIVITY_KIND_MEMCPY values(5, 9, 0, 1, 4096);
            create table CUPTI_ACTIVITY_KIND_RUNTIME(
                start integer, end integer, globalPid integer, globalTid integer,
                correlationId integer, nameId integer
            );
            insert into CUPTI_ACTIVITY_KIND_RUNTIME values(1, 5, 100, 101, 55, 3);
            create table NVTX_EVENTS(
                start integer, end integer, globalPid integer, globalTid integer,
                textId integer
            );
            insert into NVTX_EVENTS values(0, 40, 100, 101, 4);
            """
        )
        connection.commit()
        connection.close()
        return path

    def test_schema_introspection_and_variant_event_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = self.make_database(Path(tmp))
            connection = sqlite3.connect(str(database))
            inventory = inspect_schema(connection)
            connection.close()
            extraction = load_kernel_events(database)
            devices = load_device_metadata(database)
            memory = load_memory_events(database)
            api = load_cuda_api_events(database)
            nvtx = load_nvtx_ranges(database)

        self.assertIn("CUPTI_ACTIVITY_KIND_KERNEL", inventory.tables)
        self.assertIn("startNs", inventory.tables["CUPTI_ACTIVITY_KIND_KERNEL"])
        self.assertEqual(len(extraction.events), 2)
        first = extraction.events[0]
        self.assertEqual((first.start_ns, first.end_ns), (10, 30))
        self.assertEqual((first.device_id, first.context_id, first.stream_id), (0, 2, 7))
        self.assertEqual((first.process_id, first.thread_id, first.correlation_id), (100, 101, 55))
        self.assertEqual(first.name, "deep_gemm::sm90_fp8_gemm")
        self.assertEqual(first.category, "GEMM (unattributed)")
        self.assertEqual(extraction.events[1].category, "Custom AllReduce")
        self.assertEqual(devices, {0: "GPU-A", 1: "GPU-B"})
        self.assertEqual(memory[0].bytes, 4096)
        self.assertEqual(api[0].correlation_id, 55)
        self.assertEqual(api[0].name, "cudaLaunchKernel")
        self.assertEqual(nvtx[0].text, "Decode batch")
        self.assertEqual(first.nvtx_range, "Decode batch")

    def test_missing_kernel_table_is_partial_with_concrete_capability(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.sqlite"
            connection = sqlite3.connect(str(path))
            connection.execute("create table StringIds(id integer, value text)")
            connection.commit()
            connection.close()
            extraction = load_kernel_events(path)
        self.assertEqual(extraction.events, [])
        self.assertIn("kernel event table", " ".join(extraction.missing_capabilities))

    def test_event_artifacts_include_stream_timeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = self.make_database(root)
            extraction = load_kernel_events(database)
            output = root / "summary"
            write_event_artifacts(extraction, output)
            kernel_csv = (output / "kernel_events.csv").read_text()
            stream_csv = (output / "stream_timeline.csv").read_text()
            schema_json = (output / "sqlite_schema.json").read_text()
            api_exists = (output / "cuda_api_events.csv").is_file()
            nvtx_exists = (output / "nvtx_ranges.csv").is_file()
        self.assertIn("classification_confidence", kernel_csv)
        self.assertIn("stream_id", stream_csv)
        self.assertIn("CUPTI_ACTIVITY_KIND_KERNEL", schema_json)
        self.assertTrue(api_exists)
        self.assertTrue(nvtx_exists)

    def test_sqlite_events_can_generate_core_kernel_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            extraction = load_kernel_events(self.make_database(root))
            output = root / "cuda_gpu_kern_sum.csv"
            write_kernel_summary_from_events(extraction.events, output)
            text = output.read_text()
        for header in ("Total Time (ns)", "Instances", "Avg (ns)", "Name"):
            self.assertIn(header, text.splitlines()[0])
        self.assertIn("deep_gemm::sm90_fp8_gemm", text)


if __name__ == "__main__":
    unittest.main()
