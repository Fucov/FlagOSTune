from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.tools.nsys.sqlite_events import (
    CudaApiEvent,
    NvtxRange,
    _attribute_nvtx,
    inspect_schema,
    load_device_metadata,
    load_cuda_api_events,
    load_kernel_events,
    load_memory_events,
    load_nvtx_ranges,
    write_event_artifacts,
    write_kernel_summary_from_events,
)
from scripts.tools.nsys.models import KernelEvent


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
                start INTEGER NOT NULL, end INTEGER, eventType INTEGER NOT NULL,
                rangeId INTEGER, category INTEGER, color INTEGER, text TEXT,
                globalTid INTEGER, endGlobalTid INTEGER, textId INTEGER,
                domainId INTEGER, uint64Value INTEGER, int64Value INTEGER,
                doubleValue REAL, uint32Value INTEGER, int32Value INTEGER,
                floatValue REAL, jsonTextId INTEGER, jsonText TEXT, binaryData TEXT
            );
            insert into NVTX_EVENTS(start,end,eventType,rangeId,text,globalTid,textId,domainId)
                values(0, 40, 59, 1, NULL, 1677721701, 4, 0);
            insert into NVTX_EVENTS(start,end,eventType,rangeId,text,globalTid,domainId)
                values(-228624625930, NULL, 75, NULL, 'NCCL', 288244734765109, 1);
            insert into NVTX_EVENTS(start,end,eventType,rangeId,text,globalTid,domainId)
                values(-227470231076, NULL, 39, NULL, 'NCCL Progress 0', 288244734766439, 0);
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
            extraction = load_kernel_events(
                database, include_cuda_api=True, include_nvtx=True, attribute_nvtx=True
            )
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
        self.assertEqual(nvtx.ranges[0].text, "Decode batch")
        self.assertEqual(nvtx.stats.null_end_rows, 2)
        self.assertEqual(first.nvtx_range, "Decode batch")
        self.assertEqual(first.nvtx_attribution_source, "CUDA_API_CORRELATION")
        self.assertEqual(first.launch_api_name, "cudaLaunchKernel")

    def test_default_kernel_load_does_not_read_nvtx_or_cuda_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = self.make_database(Path(tmp))
            with mock.patch(
                "scripts.tools.nsys.sqlite_events._load_nvtx_ranges",
                side_effect=AssertionError("NVTX loader must stay lazy"),
            ), mock.patch(
                "scripts.tools.nsys.sqlite_events._load_cuda_api_events",
                side_effect=AssertionError("CUDA API loader must stay lazy"),
            ):
                extraction = load_kernel_events(database)
        self.assertEqual(len(extraction.events), 2)
        self.assertEqual(extraction.nvtx_load_status, "NOT_REQUESTED")

    def test_nvtx_nulls_reversed_negative_and_arbitrary_type_are_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            database = self.make_database(root)
            connection = sqlite3.connect(str(database))
            connection.executescript(
                """
                insert into NVTX_EVENTS(start,end,eventType,rangeId,text,globalTid,domainId)
                    values(100, 200, 999, 9, 'custom', 1001, 0);
                insert into NVTX_EVENTS(start,end,eventType,rangeId,text,globalTid,domainId)
                    values(300, 250, 59, 10, 'reversed', 1001, 0);
                insert into NVTX_EVENTS(start,end,eventType,rangeId,text,globalTid,domainId)
                    values(-20, -10, 59, 11, 'negative-valid', 1001, 0);
                """
            )
            connection.commit()
            connection.close()
            result = load_nvtx_ranges(database)

        texts = [row.text for row in result.ranges]
        self.assertIn("custom", texts)
        self.assertIn("negative-valid", texts)
        self.assertNotIn("NCCL", texts)
        self.assertNotIn("NCCL Progress 0", texts)
        self.assertNotIn("reversed", texts)
        self.assertEqual(result.stats.null_end_rows, 2)
        self.assertEqual(result.stats.skipped_by_event_type, {75: 1, 39: 1})
        self.assertEqual(result.stats.invalid_interval_rows, 1)
        self.assertEqual(result.stats.negative_start_rows, 3)
        self.assertEqual(result.stats.counts_by_event_type[999], 1)

    def test_nvtx_null_end_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = load_nvtx_ranges(self.make_database(Path(tmp)))
        self.assertEqual(result.stats.null_end_rows, 2)
        self.assertNotIn("NCCL", [row.text for row in result.ranges])

    def test_nvtx_null_end_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = load_nvtx_ranges(self.make_database(Path(tmp)))
        self.assertEqual(result.stats.valid_closed_ranges, 1)

    def test_nvtx_reversed_interval_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = self.make_database(Path(tmp))
            connection = sqlite3.connect(str(database))
            connection.execute(
                "insert into NVTX_EVENTS(start,end,eventType,text) values(20,10,59,'bad')"
            )
            connection.commit()
            connection.close()
            result = load_nvtx_ranges(database)
        self.assertEqual(result.stats.invalid_interval_rows, 1)
        self.assertNotIn("bad", [row.text for row in result.ranges])

    def test_nvtx_null_start_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nullable.sqlite"
            connection = sqlite3.connect(str(path))
            connection.execute(
                "create table NVTX_EVENTS(start integer, end integer, eventType integer, text text)"
            )
            connection.execute("insert into NVTX_EVENTS values(NULL, 20, 59, 'bad')")
            connection.execute("insert into NVTX_EVENTS values(10, 20, 59, 'good')")
            connection.commit()
            connection.close()
            result = load_nvtx_ranges(path)
        self.assertEqual([row.text for row in result.ranges], ["good"])
        self.assertEqual(result.stats.null_start_rows, 1)

    def test_real_null_end_counts_and_no_synthetic_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = self.make_database(Path(tmp))
            connection = sqlite3.connect(str(database))
            for event_type, text in ((75, "B"), (39, "C")):
                connection.execute(
                    "insert into NVTX_EVENTS(start,end,eventType,text) values(?,NULL,?,?)",
                    (-100 - event_type, event_type, text),
                )
            connection.execute(
                "insert into NVTX_EVENTS(start,end,eventType,text) values(100,200,59,'second')"
            )
            connection.commit()
            connection.close()
            result = load_nvtx_ranges(database)
        self.assertEqual(result.stats.valid_closed_ranges, 2)
        self.assertEqual(result.stats.null_end_rows, 4)
        self.assertEqual(result.stats.skipped_by_event_type[75], 2)
        self.assertEqual(result.stats.skipped_by_event_type[39], 2)
        self.assertTrue(all(row.end_ns is not None for row in result.ranges))

    def test_correlation_uses_launch_time_and_selects_innermost_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "correlation.sqlite"
            connection = sqlite3.connect(str(path))
            connection.executescript(
                """
                create table StringIds(id integer primary key, value text);
                insert into StringIds values(1, 'kernel');
                insert into StringIds values(2, 'cudaLaunchKernel');
                create table CUPTI_ACTIVITY_KIND_KERNEL(
                    start integer, end integer, deviceId integer, contextId integer,
                    streamId integer, correlationId integer, nameId integer
                );
                insert into CUPTI_ACTIVITY_KIND_KERNEL values(500,700,0,1,1,42,1);
                create table CUPTI_ACTIVITY_KIND_RUNTIME(
                    start integer, end integer, globalTid integer,
                    correlationId integer, nameId integer
                );
                insert into CUPTI_ACTIVITY_KIND_RUNTIME values(150,160,1677721701,42,2);
                create table NVTX_EVENTS(
                    start integer not null, end integer, eventType integer not null,
                    rangeId integer, text text, globalTid integer, endGlobalTid integer,
                    textId integer, domainId integer
                );
                insert into NVTX_EVENTS values(100,300,59,1,'layer.0',1677721701,1677721701,NULL,0);
                insert into NVTX_EVENTS values(120,220,59,2,'attention',1677721701,1677721701,NULL,0);
                insert into NVTX_EVENTS values(140,180,999,3,'qkv',1677721701,1677721701,NULL,0);
                """
            )
            connection.commit()
            connection.close()
            extraction = load_kernel_events(
                path, include_cuda_api=True, include_nvtx=True, attribute_nvtx=True
            )
        event = extraction.events[0]
        self.assertEqual(event.nvtx_range, "qkv")
        self.assertEqual(event.nvtx_attribution_source, "CUDA_API_CORRELATION")
        self.assertEqual(event.nvtx_attribution_confidence, "HIGH")
        self.assertEqual(event.launch_api_start_ns, 150)
        self.assertLess(extraction.attribution_candidate_checks, 10)

    def test_nvtx_enrichment_failure_preserves_kernel_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            database = self.make_database(Path(tmp))
            with mock.patch(
                "scripts.tools.nsys.sqlite_events._load_nvtx_ranges",
                side_effect=sqlite3.DatabaseError("broken NVTX table"),
            ):
                extraction = load_kernel_events(
                    database, include_cuda_api=True, include_nvtx=True, attribute_nvtx=True
                )
        self.assertEqual(len(extraction.events), 2)
        self.assertEqual(extraction.kernel_event_status, "PASS")
        self.assertEqual(extraction.nvtx_load_status, "FAILED")
        self.assertEqual(extraction.overall_status, "PARTIAL")

    def test_large_interval_index_does_not_scan_every_range_per_kernel(self):
        range_count = 10_000
        event_count = 200
        ranges = [
            NvtxRange(index, index * 20, index * 20 + 10, 1, 7, str(index), "NVTX_EVENTS")
            for index in range(range_count)
        ]
        apis = [
            CudaApiEvent(index, index * 20 + 2, index * 20 + 3, 1, 7, index,
                         "cudaLaunchKernel", "CUPTI_ACTIVITY_KIND_RUNTIME")
            for index in range(event_count)
        ]
        events = [
            KernelEvent(index, 0, 1, 1, 1_000_000 + index * 10,
                        1_000_005 + index * 10, "kernel", correlation_id=index)
            for index in range(event_count)
        ]

        attributed, counters = _attribute_nvtx(events, apis, ranges)

        self.assertEqual(len(attributed), event_count)
        self.assertEqual(counters.queries, event_count)
        self.assertLess(counters.candidate_checks, range_count)

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
            metadata_exists = (output / "event_extraction_metadata.json").is_file()
        self.assertIn("classification_confidence", kernel_csv)
        self.assertIn("stream_id", stream_csv)
        self.assertIn("CUPTI_ACTIVITY_KIND_KERNEL", schema_json)
        self.assertIn('"nvtx_load_status": "NOT_REQUESTED"', schema_json)
        self.assertTrue(api_exists)
        self.assertTrue(nvtx_exists)
        self.assertTrue(metadata_exists)

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
