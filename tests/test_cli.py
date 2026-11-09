from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from fake_server import FakeOpenAI
from llm_bench.cli import main
from test_tasks import VALID_SUMMARY


def responder(payload: dict[str, object]):
    prompt = payload["messages"][0]["content"]  # type: ignore[index]
    if "add(a, b)" in prompt:
        content = "```python\ndef add(a, b):\n    return a + b\n```"
    elif "发票" in prompt:
        content = '{"invoice_no": "INV-2026-0917", "total": 1280.5, "currency": "CNY"}'
    else:
        content = VALID_SUMMARY
    return 200, {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def broken_responder(payload: dict[str, object]):
    return 200, {"choices": [{"message": {"content": "不知道"}}], "usage": {}}


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)
        self.server = FakeOpenAI(responder)
        self.addCleanup(self.server.close)
        self.config = self.write_config()

    def write_config(self, base_url: str | None = None) -> Path:
        payload = {
            "models": [
                {
                    "name": "demo",
                    "base_url": base_url or self.server.base_url,
                    "model": "demo-1",
                    "api_key_env": "DEMO_KEY",
                }
            ],
            "max_tokens": 32,
        }
        path = self.directory / "bench.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def run_cli(self, argv: list[str]):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_version(self) -> None:
        with self.assertRaises(SystemExit) as caught, redirect_stdout(io.StringIO()):
            main(["--version"])
        self.assertEqual(caught.exception.code, 0)

    def test_lists_tasks(self) -> None:
        code, out, _ = self.run_cli(["tasks"])
        self.assertEqual(code, 0)
        for task_id in ("code-fix", "extract", "summarize"):
            self.assertIn(task_id, out)

    def test_dry_run_skips_network_and_keys(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            code, out, _ = self.run_cli(["run", "--config", str(self.config), "--dry-run"])

        self.assertEqual(code, 0)
        self.assertIn("未设置", out)
        self.assertIn("不会发起网络请求", out)
        self.assertEqual(self.server.requests, [])

    def test_missing_key_is_an_error(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            code, _, err = self.run_cli(["run", "--config", str(self.config)])
        self.assertEqual(code, 2)
        self.assertIn("DEMO_KEY", err)

    def test_unknown_model(self) -> None:
        code, _, err = self.run_cli(
            ["run", "--config", str(self.config), "--models", "nope", "--dry-run"]
        )
        self.assertEqual(code, 2)
        self.assertIn("未知模型", err)

    def test_unknown_task(self) -> None:
        code, _, err = self.run_cli(
            ["run", "--config", str(self.config), "--tasks", "nope", "--dry-run"]
        )
        self.assertEqual(code, 2)
        self.assertIn("未知任务", err)

    def test_end_to_end_all_tasks_pass(self) -> None:
        with patch.dict(os.environ, {"DEMO_KEY": "test-key"}, clear=True):
            code, out, _ = self.run_cli(
                ["run", "--config", str(self.config), "--format", "markdown"]
            )

        self.assertEqual(code, 0, out)
        self.assertIn("# llm-bench 报告", out)
        self.assertIn("3/3", out)
        self.assertEqual(len(self.server.requests), 3)
        self.assertEqual(self.server.requests[0]["authorization"], "Bearer test-key")

    def test_failure_sets_exit_code_and_exit_zero_overrides(self) -> None:
        self.server.close()
        self.server = FakeOpenAI(broken_responder)
        self.addCleanup(self.server.close)
        self.config = self.write_config()

        with patch.dict(os.environ, {"DEMO_KEY": "test-key"}, clear=True):
            code, out, _ = self.run_cli(["run", "--config", str(self.config)])
            relaxed, _, _ = self.run_cli(["run", "--config", str(self.config), "--exit-zero"])

        self.assertEqual(code, 1)
        self.assertIn("0/3", out)
        self.assertEqual(relaxed, 0)

    def test_writes_output_file_and_saves_responses(self) -> None:
        report = self.directory / "out" / "report.md"
        responses = self.directory / "responses"
        with patch.dict(os.environ, {"DEMO_KEY": "test-key"}, clear=True):
            code, out, err = self.run_cli(
                [
                    "run",
                    "--config",
                    str(self.config),
                    "--tasks",
                    "code-fix",
                    "--format",
                    "markdown",
                    "--output",
                    str(report),
                    "--save-responses",
                    str(responses),
                ]
            )

        self.assertEqual(code, 0, err)
        self.assertEqual(out, "")
        self.assertIn("已写入", err)
        self.assertIn("# llm-bench 报告", report.read_text(encoding="utf-8"))
        self.assertEqual([path.name for path in responses.iterdir()], ["demo__code-fix__1.txt"])

    def test_missing_config_file(self) -> None:
        code, _, err = self.run_cli(["run", "--config", str(self.directory / "nope.json")])
        self.assertEqual(code, 2)
        self.assertIn("读不到配置文件", err)

    def test_repeat_must_be_positive(self) -> None:
        code, _, err = self.run_cli(
            ["run", "--config", str(self.config), "--repeat", "0", "--dry-run"]
        )
        self.assertEqual(code, 2)
        self.assertIn("--repeat", err)


if __name__ == "__main__":
    unittest.main()
