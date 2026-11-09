from __future__ import annotations

import csv
import io
import json
import unittest

from llm_bench.bench import ModelSummary, RunResult
from llm_bench.report import render, render_csv, render_json, render_markdown, render_table


SUMMARY = ModelSummary(
    model="good",
    runs=2,
    passed=2,
    errors=0,
    pass_rate=1.0,
    avg_latency_ms=12.5,
    prompt_tokens=200,
    completion_tokens=40,
    cost_usd=0.00016,
)

RESULTS = [
    RunResult("good", "code-fix", "修复加法函数", 1, True, "返回值已改成加法", 10.0, 100, 20, 0.00008),
    RunResult("good", "extract", "抽取发票字段", 1, False, "缺字段 total | 换行\n说明", 15.0, 100, 20, 0.00008),
]


class TableTest(unittest.TestCase):
    def test_contains_summary_values(self) -> None:
        text = render_table([SUMMARY])
        self.assertIn("good", text)
        self.assertIn("2/2", text)
        self.assertIn("100%", text)
        self.assertIn("12 ms", text)
        self.assertIn("$0.000160", text)

    def test_unknown_cost_is_not_zero(self) -> None:
        unknown = ModelSummary("x", 1, 0, 1, 0.0, 0.0, 0, 0, None)
        self.assertIn("n/a", render_table([unknown]))


class MarkdownTest(unittest.TestCase):
    def test_has_summary_and_detail_sections(self) -> None:
        text = render_markdown([SUMMARY], RESULTS)
        self.assertIn("## 汇总", text)
        self.assertIn("## 明细", text)
        self.assertIn("| good | code-fix | 1 | pass |", text)
        self.assertIn("| good | extract | 1 | fail |", text)

    def test_escapes_pipes_and_newlines(self) -> None:
        text = render_markdown([SUMMARY], RESULTS)
        self.assertIn(r"缺字段 total \| 换行 说明", text)


class CsvTest(unittest.TestCase):
    def test_round_trips_through_csv_reader(self) -> None:
        rows = list(csv.reader(io.StringIO(render_csv(RESULTS))))
        self.assertEqual(rows[0][0], "model")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1][1], "code-fix")
        self.assertEqual(rows[1][3], "true")
        self.assertEqual(rows[1][7], "0.000080")
        self.assertEqual(rows[2][3], "false")


class JsonTest(unittest.TestCase):
    def test_structure_and_values(self) -> None:
        payload = json.loads(render_json([SUMMARY], RESULTS))
        self.assertEqual(payload["summary"][0]["model"], "good")
        self.assertEqual(payload["summary"][0]["cost_usd"], 0.00016)
        self.assertEqual(payload["results"][1]["passed"], False)
        self.assertIsNone(payload["results"][0]["error"])


class DispatchTest(unittest.TestCase):
    def test_each_format(self) -> None:
        self.assertIn("good", render("table", [SUMMARY], RESULTS))
        self.assertIn("# llm-bench 报告", render("markdown", [SUMMARY], RESULTS))
        self.assertIn("model,task", render("csv", [SUMMARY], RESULTS))
        self.assertEqual(json.loads(render("json", [SUMMARY], RESULTS))["summary"][0]["runs"], 2)

    def test_unknown_format(self) -> None:
        with self.assertRaises(ValueError):
            render("xml", [SUMMARY], RESULTS)


if __name__ == "__main__":
    unittest.main()
