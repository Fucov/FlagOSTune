import tempfile
import unittest
from pathlib import Path

from scripts.tools.nsys.normalize_stats import load_kernel_summary
from scripts.tools.nsys.utils import find_column, normalize_header, parse_duration_ns


class NsysNormalizeStatsTest(unittest.TestCase):
    def test_header_lookup_tolerates_punctuation_and_case(self):
        headers = ["Time:Percent", "TOTAL TIME (ns)", "Kernel Name"]
        self.assertEqual(normalize_header(" Time (%) "), "time")
        self.assertEqual(find_column(headers, ("total time ns",)), "TOTAL TIME (ns)")

    def test_duration_units_are_converted_to_nanoseconds(self):
        self.assertEqual(parse_duration_ns("1.5", "Total Time (ms)"), 1_500_000)
        self.assertEqual(parse_duration_ns("2", "Avg (us)"), 2_000)
        self.assertEqual(parse_duration_ns("3", "Duration (ns)"), 3)

    def test_loads_aliases_and_recomputes_full_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kernels.csv"
            path.write_text(
                "Time:Percent,Total Time (us),Instances,Avg (us),Kernel Name\n"
                "1,600,3,200,kernel_a\n"
                "99,400,2,200,kernel_b\n",
                encoding="utf-8",
            )
            rows = load_kernel_summary(path)

        self.assertEqual([row.name for row in rows], ["kernel_a", "kernel_b"])
        self.assertEqual(rows[0].total_ns, 600_000)
        self.assertAlmostEqual(rows[0].time_percentage, 60.0)
        self.assertEqual(rows[0].instances, 3)

    def test_missing_optional_columns_are_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kernels.csv"
            path.write_text(
                "Total Time (ns),Calls,Name\n100,1,kernel\n",
                encoding="utf-8",
            )
            row = load_kernel_summary(path)[0]

        self.assertIsNone(row.avg_ns)
        self.assertIsNone(row.median_ns)

    def test_missing_required_columns_fail_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("Name,Calls\nkernel,1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "required column"):
                load_kernel_summary(path)


if __name__ == "__main__":
    unittest.main()
