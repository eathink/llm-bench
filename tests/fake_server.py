"""测试用的假 OpenAI 兼容服务。"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

Responder = Callable[[dict[str, object]], "tuple[int, dict[str, object] | bytes]"]


def default_responder(payload: dict[str, object]) -> tuple[int, dict[str, object] | bytes]:
    return (
        200,
        {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


class FakeOpenAI:
    def __init__(self, responder: Responder = default_responder) -> None:
        self.responder = responder
        self.requests: list[dict[str, object]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server 约定
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append(
                    {
                        "path": self.path,
                        "payload": payload,
                        "authorization": self.headers.get("Authorization"),
                    }
                )
                status, data = outer.responder(payload)
                raw = data if isinstance(data, bytes) else json.dumps(data).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except OSError:
                    pass  # 客户端可能已经超时断开

            def log_message(self, *args: object) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def __enter__(self) -> "FakeOpenAI":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
