from __future__ import annotations

import time
import unittest

from fake_server import FakeOpenAI
from llm_bench.client import ApiError, chat, endpoint


def _chat(base_url: str, **overrides: object):
    options: dict[str, object] = {
        "base_url": base_url,
        "api_key": "test-key",
        "model": "demo-1",
        "prompt": "hello",
        "timeout": 5.0,
        "temperature": 0.0,
        "max_tokens": 32,
    }
    options.update(overrides)
    return chat(**options)  # type: ignore[arg-type]


class EndpointTest(unittest.TestCase):
    def test_strips_trailing_slash(self) -> None:
        self.assertEqual(
            endpoint("https://api.example.com/v1/"),
            "https://api.example.com/v1/chat/completions",
        )


class ChatTest(unittest.TestCase):
    def test_successful_completion(self) -> None:
        with FakeOpenAI() as server:
            completion = _chat(server.base_url)

        self.assertEqual(completion.content, "ok")
        self.assertEqual(completion.prompt_tokens, 10)
        self.assertEqual(completion.completion_tokens, 5)
        self.assertGreaterEqual(completion.latency_ms, 0)
        request = server.requests[0]
        self.assertEqual(request["path"], "/v1/chat/completions")
        self.assertEqual(request["authorization"], "Bearer test-key")
        payload = request["payload"]
        self.assertEqual(payload["model"], "demo-1")
        self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])
        self.assertEqual(payload["max_tokens"], 32)

    def test_omits_max_tokens_when_none(self) -> None:
        with FakeOpenAI() as server:
            _chat(server.base_url, max_tokens=None)

        self.assertNotIn("max_tokens", server.requests[0]["payload"])

    def test_usage_missing_gives_none(self) -> None:
        with FakeOpenAI(lambda payload: (200, {"choices": [{"message": {"content": "ok"}}]})) as server:
            completion = _chat(server.base_url)

        self.assertIsNone(completion.prompt_tokens)
        self.assertIsNone(completion.completion_tokens)

    def test_http_error_includes_status_and_body(self) -> None:
        with FakeOpenAI(lambda payload: (429, {"error": "slow down"})) as server:
            with self.assertRaisesRegex(ApiError, "HTTP 429: .*slow down"):
                _chat(server.base_url)

    def test_invalid_json_response(self) -> None:
        with FakeOpenAI(lambda payload: (200, b"not json")) as server:
            with self.assertRaisesRegex(ApiError, "不是合法 JSON"):
                _chat(server.base_url)

    def test_missing_choices(self) -> None:
        with FakeOpenAI(lambda payload: (200, {"choices": []})) as server:
            with self.assertRaisesRegex(ApiError, "choices"):
                _chat(server.base_url)

    def test_non_string_content(self) -> None:
        with FakeOpenAI(lambda payload: (200, {"choices": [{"message": {"content": None}}]})) as server:
            with self.assertRaisesRegex(ApiError, "不是字符串"):
                _chat(server.base_url)

    def test_timeout_raises_api_error(self) -> None:
        def slow(payload: dict[str, object]):
            time.sleep(0.4)
            return 200, {"choices": [{"message": {"content": "late"}}]}

        with FakeOpenAI(slow) as server:
            with self.assertRaisesRegex(ApiError, "超时|请求失败"):
                _chat(server.base_url, timeout=0.05)


if __name__ == "__main__":
    unittest.main()
