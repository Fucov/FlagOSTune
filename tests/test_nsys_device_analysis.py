import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.tools.nsys.analyze_devices import analyze_devices, write_device_summary


class NsysDeviceAnalysisTest(unittest.TestCase):
    def make_database(self, path, devices):
        connection = sqlite3.connect(str(path))
        connection.execute(
            "create table CUPTI_ACTIVITY_KIND_KERNEL "
            "(deviceId integer, globalPid integer, start integer, end integer, name text)"
        )
        connection.execute("create table TARGET_INFO_GPU (id integer, name text)")
        for device in range(devices):
            connection.execute("insert into TARGET_INFO_GPU values (?, ?)", (device, f"GPU-{device}"))
            connection.execute(
                "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, ?, ?)",
                (device, 100 + device, 0, 100 + device, "gemm_kernel"),
            )
            connection.execute(
                "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, ?, ?)",
                (device, 100 + device, 200, 250, "ncclAllReduce"),
            )
        connection.commit()
        connection.close()

    def test_tp4_device_counts_and_compute_communication_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.sqlite"
            self.make_database(path, 4)
            summaries, warnings, integrity = analyze_devices(path, expected_tp=4)

        self.assertEqual(len(summaries), 4)
        self.assertEqual(summaries[0].kernel_events, 2)
        self.assertEqual(summaries[0].communication_time_ns, 50)
        self.assertEqual(summaries[0].compute_time_ns, 100)
        self.assertEqual(summaries[0].gpu_name, "GPU-0")
        self.assertTrue(integrity)
        self.assertEqual(warnings, [])

    def test_tp_mismatch_is_integrity_failure_with_trace_fork_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.sqlite"
            self.make_database(path, 4)
            summaries, warnings, integrity = analyze_devices(path, expected_tp=8)

        self.assertFalse(integrity)
        self.assertEqual(len(summaries), 4)
        self.assertIn("--trace-fork-before-exec=true", warnings[0].message)

    def test_missing_optional_gpu_table_is_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.sqlite"
            connection = sqlite3.connect(str(path))
            connection.execute(
                "create table CUPTI_ACTIVITY_KIND_KERNEL "
                "(device integer, pid integer, startNs integer, endNs integer, kernelName text)"
            )
            connection.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (0, 1, 0, 10, 'gemm')")
            connection.commit()
            connection.close()
            summaries, warnings, integrity = analyze_devices(path)
            write_device_summary(summaries, Path(tmp))
            self.assertTrue((Path(tmp) / "device_summary.csv").exists())

        self.assertIsNone(summaries[0].gpu_name)
        self.assertTrue(integrity)


if __name__ == "__main__":
    unittest.main()
