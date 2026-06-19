from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from agentic_qa_lab.domain import AgentAction, FailureCategory
from agentic_qa_lab.environments import APIEnvironment


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib hook name
        body = json.dumps({"path": self.path})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length).decode("utf-8")
        if self.path == "/fail":
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"bad request")
            return
        body = json.dumps({"path": self.path, "payload": payload})
        self.send_response(201)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003 - stdlib signature
        return


def _serve() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, f"http://{host}:{port}"


def test_api_environment_get_request_round_trip() -> None:
    server, base_url = _serve()
    try:
        env = APIEnvironment(base_url)
        env.open(base_url)
        assert env.execute(AgentAction.type_text("/hello", selector="#path")).success
        result = env.execute(AgentAction.click("#send"))

        assert result.success
        observation = env.observe()
        assert '"status": 200' in (observation.visible_text or "")
        assert "/hello" in (observation.visible_text or "")
    finally:
        server.shutdown()
        server.server_close()


def test_api_environment_post_request_with_body() -> None:
    server, base_url = _serve()
    try:
        env = APIEnvironment(base_url)
        env.open(base_url)
        env.execute(AgentAction.type_text("POST", selector="#method"))
        env.execute(AgentAction.type_text("/submit", selector="#path"))
        env.execute(AgentAction.type_text('{"name":"alice"}', selector="#body"))

        result = env.execute(AgentAction.click("#send"))

        assert result.success
        visible = env.observe().visible_text or ""
        assert '"status": 201' in visible
        assert "/submit" in visible
        assert "alice" in visible
    finally:
        server.shutdown()
        server.server_close()


def test_api_environment_http_error_is_captured() -> None:
    server, base_url = _serve()
    try:
        env = APIEnvironment(base_url)
        env.open(base_url)
        env.execute(AgentAction.type_text("POST", selector="#method"))
        env.execute(AgentAction.type_text("/fail", selector="#path"))

        result = env.execute(AgentAction.click("#send"))

        assert not result.success
        assert result.failure_category is FailureCategory.UNKNOWN
        assert "HTTP 400" in (result.error or "")
    finally:
        server.shutdown()
        server.server_close()


def test_api_environment_rejects_invalid_builder_selector() -> None:
    env = APIEnvironment("https://example.com")

    result = env.execute(AgentAction.type_text("x", selector="#nope"))

    assert not result.success
    assert result.failure_category is FailureCategory.INVALID_ACTION
