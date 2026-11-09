from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fake_server import FakeOpenAI
from llm_bench.bench import run_bench, summarize
from llm_bench.config import ModelConfig, Price
from llm_bench.tasks import get_tasks
from test_tasks import VALID_SUMMARY


def responder(payload: dict[str, object]):
    model = payload["model"]
    if model == "bad":
        return 500, {"error": "boom"}
    prompt = payload["messages"][0]["content"]  # type: ignore[index]
    if "add(a, b)" in prompt:
        content = "```python\ndef add(a, b):\n    return a + b\n```"
    elif "发票" in prompt:
        content = '{"invoice_no": "INV-2026-0917", "total": 1280.5, "currency": "CNY"}'
    else:
        content = VALID_SUMMARY
    return 200, {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


class RunBenchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = FakeOpenAI(responder)
        self.addCleanup(self.server.close)
        self.models = (
            ModelConfig("good", self.server.base_url, "good", "GOOD_KEY", Price(0.5, 1.5)),
            ModelConfig("no-price", self.server.base_url, "no-price", "GOOD_KEY"),
            ModelConfig("down", self.server.base_url, "bad", "GOOD_KEY", Price(0.5, 1.5)),
        )
        self.environment = {"GOOD_KEY": "test-key"}

    def run_all(self, **overrides: object):
        options: dict[str, object] = {
            "models": self.models,
            "tasks": get_tasks(),
            "repeats": 2,
            "environment": self.environment,
            "timeout": 5.0,
            "temperature": 0.0,
            "max_tokens": 32,
        }
        options.update(overrides)
        return run_bench(**options)  # type: ignore[arg-type]

    def test_runs_every_model_task_and_repeat(self) -> None:
        results = self.run_all()

        self.assertEqual(len(results), 3 * 3 * 2)
        good = [row for row in results if row.model == "good"]
        self.assertTrue(all(row.passed and row.error is None for row in good))
        down = [row for row in results if row.model == "down"]
        self.assertTrue(all(not row.passed and row.error for row in down))
        self.assertTrue(all("HTTP 500" in row.error for row in down))

    def test_computes_cost_only_with_price_and_tokens(self) -> None:
        results = self.run_all()
        good = next(row for row in results if row.model == "good")
        no_price = next(row for row in results if row.model == "no-price")

        self.assertAlmostEqual(good.cost_usd, 0.00008, places=10)
        self.assertIsNone(no_price.cost_usd)

    def test_saves_raw_responses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = self.run_all(
                models=self.models[:1],
                tasks=get_tasks(["code-fix"]),
                repeats=1,
                save_dir=Path(tmp),
            )
            files = sorted(path.name for path in Path(tmp).iterdir())

        self.assertEqual(len(results), 1)
        self.assertEqual(files, ["good__code-fix__1.txt"])

    def test_single_request_failure_does_not_stop_the_run(self) -> None:
        results = self.run_all(models=self.models[2:], tasks=get_tasks(["code-fix"]), repeats=1)
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].error)


class SummarizeTest(unittest.TestCase):
    def test_aggregates_and_marks_unknown_cost(self) -> None:
        server = FakeOpenAI(responder)
        self.addCleanup(server.close)
        models = (
            ModelConfig("good", server.base_url, "good", "GOOD_KEY", Price(0.5, 1.5)),
            ModelConfig("no-price", server.base_url, "no-price", "GOOD_KEY"),
            ModelConfig("down", server.base_url, "bad", "GOOD_KEY", Price(0.5, 1.5)),
        )
        results = run_bench(
            models=models,
            tasks=get_tasks(["code-fix"]),
            repeats=2,
            environment={"GOOD_KEY": "test-key"},
            timeout=5.0,
            temperature=0.0,
            max_tokens=32,
        )

        summaries = {summary.model: summary for summary in summarize(results, models)}
        self.assertEqual(summaries["good"].runs, 2)
        self.assertEqual(summaries["good"].passed, 2)
        self.assertEqual(summaries["good"].pass_rate, 1.0)
        self.assertAlmostEqual(summaries["good"].cost_usd, 0.00016, places=10)
        self.assertIsNone(summaries["no-price"].cost_usd)
        self.assertEqual(summaries["down"].errors, 2)
        self.assertEqual(summaries["down"].pass_rate, 0.0)

    def test_model_without_results(self) -> None:
        model = ModelConfig("empty", "http://127.0.0.1:1/v1", "empty", "KEY")
        summary = summarize([], [model])[0]
        self.assertEqual(summary.runs, 0)
        self.assertEqual(summary.pass_rate, 0.0)
        self.assertIsNone(summary.cost_usd)


if __name__ == "__main__":
    unittest.main()
