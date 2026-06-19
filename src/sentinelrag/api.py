from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import AppConfig, load_or_create_api_token
from .rag import ask_question


def build_api_handler(config: AppConfig) -> type[BaseHTTPRequestHandler]:
    token = load_or_create_api_token(config)

    class SentinelRAGHandler(BaseHTTPRequestHandler):
        server_version = "SentinelRAG/0.1"

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/ask":
                self._send_json(404, {"error": "not_found"})
                return
            if self.headers.get("Authorization") != f"Bearer {token}":
                self._send_json(401, {"error": "unauthorized"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid_json"})
                return
                
            question = str(payload.get("question", "")).strip()
            collection = payload.get("collection")
            top_k = payload.get("top_k")
            if not question:
                self._send_json(400, {"error": "question_required"})
                return
                
            result = ask_question(question, config, collection=collection, top_k=top_k)
            self._send_json(200, result)

        def log_message(self, format: str, *args) -> None:
            return

        def _send_json(self, status_code: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return SentinelRAGHandler


def run_api_server(config: AppConfig) -> None:
    server = ThreadingHTTPServer((config.api.host, config.api.port), build_api_handler(config))
    try:
        server.serve_forever()
    finally:
        server.server_close()
