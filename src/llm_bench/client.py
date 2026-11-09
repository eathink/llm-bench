"""最小 OpenAI 兼容客户端（仅标准库）。"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

_ERROR_BODY_LIMIT = 300


class ApiError(RuntimeError):
    """请求失败或响应结构不符合 OpenAI Chat Completions 约定。"""

    def __init__(self, message: str, latency_ms: float = 0.0) -> None:
        super().__init__(message)
        self.latency_ms = latency_ms


@dataclass(frozen=True)
class Completion:
    content: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float


def endpoint(base_url: str) -> str:
    """把 base_url 规范化为 chat/completions 地址。"""
    return base_url.rstrip("/") + "/chat/completions"


def _token_count(usage: dict[str, object], key: str) -> int | None:
    value = usage.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: float,
    temperature: float,
    max_tokens: int | None,
) -> Completion:
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    request = urllib.request.Request(
        endpoint(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        elapsed = (time.perf_counter() - started) * 1000
        detail = exc.read(_ERROR_BODY_LIMIT).decode("utf-8", "replace").strip()
        raise ApiError(f"HTTP {exc.code}: {detail}", elapsed) from None
    except urllib.error.URLError as exc:
        elapsed = (time.perf_counter() - started) * 1000
        raise ApiError(f"请求失败: {exc.reason}", elapsed) from None
    except TimeoutError:
        elapsed = (time.perf_counter() - started) * 1000
        raise ApiError(f"请求超时（{timeout:g}s）", elapsed) from None
    latency_ms = (time.perf_counter() - started) * 1000

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise ApiError("响应不是合法 JSON", latency_ms) from None
    if not isinstance(data, dict):
        raise ApiError("响应根节点不是 JSON 对象", latency_ms)

    try:
        content = data["choices"][0]["message"]["content"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError):
        raise ApiError("响应缺少 choices[0].message.content", latency_ms) from None
    if not isinstance(content, str):
        raise ApiError("choices[0].message.content 不是字符串", latency_ms)

    usage = data.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    return Completion(
        content=content,
        prompt_tokens=_token_count(usage, "prompt_tokens"),
        completion_tokens=_token_count(usage, "completion_tokens"),
        latency_ms=latency_ms,
    )
