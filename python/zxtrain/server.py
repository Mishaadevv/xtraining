"""A real local inference server.

Serves the loaded model over HTTP with an OpenAI-compatible surface, so other
Zeqou components (or any client) can talk to a model trained here. Requests are
logged to the job directory, and every response is produced by the same backend
the Playground uses — there is no separate code path that could drift.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from .errors import ZxError
from .util import append_jsonl, now_iso

DOCS_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>ZeqouXTraining local API</title>
<style>body{font:14px system-ui;margin:2rem;max-width:70ch}code{background:#eee;padding:1px 4px}</style></head>
<body>
<h1>ZeqouXTraining local API</h1>
<p>This server is bound to the loopback interface and serves the model loaded by the engine.</p>
<h2>Endpoints</h2>
<ul>
  <li><code>GET /health</code> — process and model status</li>
  <li><code>GET /v1/models</code> — OpenAI-compatible model list</li>
  <li><code>POST /v1/chat/completions</code> — chat (supports <code>stream: true</code> as SSE)</li>
  <li><code>POST /v1/completions</code> — raw completion</li>
</ul>
<h2>Example</h2>
<pre>curl http://HOST:PORT/v1/chat/completions -H "Content-Type: application/json" \\
  -d '{"messages":[{"role":"user","content":"hello"}],"max_tokens":64}'</pre>
</body></html>
"""


class InferenceServer:
    def __init__(self, model_path: str, backend: Any, host: str, port: int,
                 log_file: Path | None, emit: Callable[[dict[str, Any]], None],
                 concurrency: int = 2):
        self.model_path = model_path
        self.backend = backend
        self.host = host
        self.port = port
        self.log_file = log_file
        self.emit = emit
        self.concurrency = max(1, concurrency)
        self.started_at = time.time()
        self.requests = 0
        self.errors = 0
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._state = {"running": True}
        self.loopback = host in ("127.0.0.1", "localhost", "::1")

    # -- lifecycle --------------------------------------------------------- #
    def start(self) -> dict[str, Any]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
                return  # the engine logs requests itself

            def _cors(self) -> None:
                # Only ever added for loopback binds: the app's own request
                # inspector, and local tools, then need no extra proxy. Nothing
                # is exposed beyond 127.0.0.1 unless the host is changed.
                if server.loopback:
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                    self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self._cors()
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self) -> None:  # noqa: N802 - stdlib name
                self.send_response(204)
                self._cors()
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _read_body(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    payload = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    raise ZxError(code="bad_request", message="Request body is not valid JSON.")
                return payload if isinstance(payload, dict) else {}

            def do_GET(self) -> None:  # noqa: N802 - stdlib name
                if self.path.startswith("/health"):
                    self._send(200, {
                        "status": "ok",
                        "model": server.model_path,
                        "backend": server.backend.info.id,
                        "uptime_seconds": round(time.time() - server.started_at, 1),
                        "requests": server.requests,
                        "errors": server.errors,
                    })
                elif self.path.startswith("/v1/models"):
                    self._send(200, {
                        "object": "list",
                        "data": [{
                            "id": Path(server.model_path).name,
                            "object": "model",
                            "created": int(server.started_at),
                            "owned_by": "zeqou",
                            "root": server.model_path,
                        }],
                    })
                elif self.path.startswith("/docs"):
                    body = DOCS_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self._cors()
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._send(404, {"error": {"message": f"Unknown path {self.path}"}})

            def do_POST(self) -> None:  # noqa: N802 - stdlib name
                try:
                    payload = self._read_body()
                except ZxError as exc:
                    self._send(400, {"error": exc.to_dict()})
                    return
                try:
                    if self.path.startswith("/v1/chat/completions"):
                        request = {
                            "messages": payload.get("messages") or [],
                            "max_tokens": payload.get("max_tokens", 256),
                            "temperature": payload.get("temperature", 0.8),
                            "top_p": payload.get("top_p", 0.95),
                            "top_k": payload.get("top_k", 0),
                            "repetition_penalty": payload.get("repetition_penalty", 1.0),
                            "seed": payload.get("seed"),
                            "stop": payload.get("stop"),
                        }
                        stream = bool(payload.get("stream"))
                    elif self.path.startswith("/v1/completions"):
                        request = {
                            "prompt": payload.get("prompt", ""),
                            "max_tokens": payload.get("max_tokens", 256),
                            "temperature": payload.get("temperature", 0.8),
                            "top_p": payload.get("top_p", 0.95),
                            "seed": payload.get("seed"),
                        }
                        stream = bool(payload.get("stream"))
                    else:
                        self._send(404, {"error": {"message": f"Unknown path {self.path}"}})
                        return

                    if stream:
                        self._stream(request)
                        return
                    result = server.backend.generate(server.model_path, request, lambda _event: None)
                    server._record(self.path, request, result)
                    self._send(200, {
                        "id": f"chatcmpl-{int(time.time() * 1000)}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": Path(server.model_path).name,
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": result.get("text", "")},
                            "finish_reason": result.get("finish_reason", "stop"),
                        }],
                        "usage": {
                            "prompt_tokens": result.get("prompt_tokens"),
                            "completion_tokens": result.get("completion_tokens"),
                            "total_tokens": (result.get("prompt_tokens") or 0) + (result.get("completion_tokens") or 0),
                        },
                        "zxtrain": {
                            "backend": result.get("backend"),
                            "latency_seconds": result.get("latency_seconds"),
                            "tokens_per_second": result.get("tokens_per_second"),
                        },
                    })
                except ZxError as exc:
                    server.errors += 1
                    server._log({"type": "error", "path": self.path, "error": exc.to_dict()})
                    self._send(500, {"error": exc.to_dict()})
                except BaseException as exc:  # noqa: BLE001 - the server must answer
                    from .errors import from_exception

                    server.errors += 1
                    wrapped = from_exception(exc)
                    server._log({"type": "error", "path": self.path, "error": wrapped.to_dict()})
                    self._send(500, {"error": wrapped.to_dict()})

            def _stream(self, request: dict[str, Any]) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self._cors()
                self.end_headers()
                chunks: list[str] = []

                def on_event(event: dict[str, Any]) -> None:
                    if event.get("type") == "token":
                        chunks.append(str(event.get("text", "")))
                        data = {
                            "id": f"chatcmpl-{int(time.time() * 1000)}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": Path(server.model_path).name,
                            "choices": [{"index": 0, "delta": {"content": str(event.get("text", ""))}, "finish_reason": None}],
                        }
                        try:
                            self.wfile.write(f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8"))
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            pass

                try:
                    result = server.backend.generate(server.model_path, request, on_event)
                except ZxError as exc:
                    self.wfile.write(f"data: {json.dumps({'error': exc.to_dict()})}\n\n".encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    server.errors += 1
                    return
                server._record(self.path, request, result)
                self.wfile.write(b"data: [DONE]\n\n")
                try:
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError as exc:
            raise ZxError(
                code="port_in_use",
                message=f"Cannot bind {self.host}:{self.port} — {exc}",
                hint="Another process is using that port. Pick a different port in the Deploy panel.",
            ) from exc
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        info = {
            "state": "running",
            "host": self.host,
            "port": self.port,
            "model": self.model_path,
            "backend": self.backend.info.id,
            "endpoints": ["/health", "/v1/models", "/v1/chat/completions", "/v1/completions", "/docs"],
            "started_at": now_iso(),
        }
        self._log({"type": "server_started", **info})
        self.emit({"type": "log", "level": "info",
                   "message": f"Local API listening on http://{self.host}:{self.port} (model {Path(self.model_path).name})"})
        self.emit({"type": "metrics", "phase": "serve", "server": info})
        return info

    def _record(self, path: str, request: dict[str, Any], result: dict[str, Any]) -> None:
        with self._lock:
            self.requests += 1
        self._log({
            "type": "request",
            "path": path,
            "at": now_iso(),
            "request": {key: value for key, value in request.items() if key != "messages"},
            "prompt_tokens": result.get("prompt_tokens"),
            "completion_tokens": result.get("completion_tokens"),
            "latency_seconds": result.get("latency_seconds"),
            "tokens_per_second": result.get("tokens_per_second"),
            "preview": (result.get("text") or "")[:400],
        })
        self.emit({
            "type": "metrics",
            "phase": "serve",
            "requests": self.requests,
            "errors": self.errors,
            "last_latency_seconds": result.get("latency_seconds"),
            "last_tokens_per_second": result.get("tokens_per_second"),
        })

    def _log(self, payload: dict[str, Any]) -> None:
        if self.log_file:
            try:
                append_jsonl(self.log_file, payload)
            except OSError:
                pass

    def stop(self) -> None:
        self._state["running"] = False
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        self._log({"type": "server_stopped", "at": now_iso()})

    def serve_until(self, should_stop: Callable[[], bool], poll: float = 0.5) -> dict[str, Any]:
        info = self.start()
        while not should_stop():
            time.sleep(poll)
        self.stop()
        return {**info, "state": "stopped", "requests": self.requests, "errors": self.errors}
