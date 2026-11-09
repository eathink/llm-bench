from __future__ import annotations

import unittest

from llm_bench.tasks import (
    TASKS,
    check_code_fix,
    check_extract,
    check_summary,
    get_tasks,
)

VALID_SUMMARY = (
    "长上下文推理受延迟、成本和缓存约束。"
    "前缀缓存可省重复 prefill，结果缓存适合重复问题。"
    "应按分位延迟评估体验并压缩提示词。"
)


class GetTasksTest(unittest.TestCase):
    def test_default_is_all_builtin_tasks(self) -> None:
        self.assertEqual([task.id for task in get_tasks()], ["code-fix", "extract", "summarize"])

    def test_filters_in_requested_order(self) -> None:
        self.assertEqual([task.id for task in get_tasks(["summarize", "code-fix"])], ["summarize", "code-fix"])

    def test_unknown_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "未知任务: nope"):
            get_tasks(["nope"])

    def test_task_ids_are_unique(self) -> None:
        self.assertEqual(len({task.id for task in TASKS}), len(TASKS))


class CodeFixTest(unittest.TestCase):
    def test_accepts_fenced_fix(self) -> None:
        result = check_code_fix("```python\ndef add(a, b):\n    return a + b\n```")
        self.assertTrue(result.passed)

    def test_accepts_commuted_operands(self) -> None:
        self.assertTrue(check_code_fix("```python\ndef add(a, b):\n    return b + a\n```").passed)

    def test_rejects_unchanged_bug(self) -> None:
        result = check_code_fix("```python\ndef add(a, b):\n    return a - b\n```")
        self.assertFalse(result.passed)
        self.assertIn("a - b", result.detail)

    def test_rejects_answer_without_fix(self) -> None:
        self.assertFalse(check_code_fix("把减号改成加号").passed)


class ExtractTest(unittest.TestCase):
    def test_accepts_json_inside_explanation(self) -> None:
        text = '结果如下：\n```json\n{"invoice_no": "INV-2026-0917", "total": 1280.5, "currency": "CNY"}\n```'
        self.assertTrue(check_extract(text).passed)

    def test_rejects_wrong_integer_total(self) -> None:
        text = '{"invoice_no": "INV-2026-0917", "total": 1280, "currency": "CNY"}'
        self.assertFalse(check_extract(text).passed)

    def test_reports_missing_and_wrong_fields(self) -> None:
        result = check_extract('{"invoice_no": "WRONG", "currency": "USD"}')
        self.assertFalse(result.passed)
        self.assertIn("total", result.detail)
        self.assertIn("invoice_no", result.detail)
        self.assertIn("currency", result.detail)

    def test_rejects_non_json(self) -> None:
        self.assertFalse(check_extract("发票号是 INV-2026-0917").passed)


class SummaryTest(unittest.TestCase):
    def test_accepts_valid_summary(self) -> None:
        result = check_summary(VALID_SUMMARY)
        self.assertTrue(result.passed, result.detail)

    def test_rejects_missing_keyword(self) -> None:
        result = check_summary("长上下文推理受延迟和成本约束。应压缩提示词。")
        self.assertFalse(result.passed)
        self.assertIn("缓存", result.detail)

    def test_rejects_too_long(self) -> None:
        result = check_summary(VALID_SUMMARY + "补充" * 40)
        self.assertFalse(result.passed)
        self.assertIn("超过 100 字", result.detail)

    def test_rejects_too_many_sentences(self) -> None:
        text = "延迟。成本。缓存。都重要。"
        result = check_summary(text)
        self.assertFalse(result.passed)
        self.assertIn("4 句", result.detail)

    def test_rejects_too_short(self) -> None:
        result = check_summary("延迟、成本、缓存。")
        self.assertFalse(result.passed)
        self.assertIn("太短", result.detail)


if __name__ == "__main__":
    unittest.main()
