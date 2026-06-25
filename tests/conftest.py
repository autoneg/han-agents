"""Shared test fixtures.

The HAN LLM finalists talk to an Ollama backend in three different ways:

* raw ``urllib`` POSTs to ``http://localhost:11434/api/generate`` (e.g.
  ``team_22147``, ``team_22270``),
* ``litellm`` with the ``ollama`` provider (``/api/generate``) or the
  ``ollama_chat`` provider (``/api/chat``), used by the ``negmas_llm``
  components and ``team_21099``.

The single point that intercepts *all* of them is the Ollama HTTP
endpoint on port 11434 -- not ``litellm`` (which the raw-urllib agents
bypass). So instead of pulling a real model, we stand up a tiny
Ollama-compatible HTTP server that returns a fixed, parseable reply.
This lets the LLM agents actually exercise their model-call code paths
locally and in CI with no model download and fully deterministically.

If a real Ollama is already listening on 11434 (the developer is running
one), we defer to it and never start the fake -- so the same tests can
run against a real, small model by simply having ``ollama serve`` up.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_BASE = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# A reply that every finalist can parse without raising. Most agents use
# the LLM only for the free-text/persuasion layer (the numeric offer is
# computed classically), so a benign message is enough; the extra keys
# cover agents that look for a decision or an explicit offer. ``outcome``
# is null on purpose -- agents fall back to their classical offer.
_FAKE_CONTENT = json.dumps(
    {
        "message": "Let us work towards a fair agreement that benefits us both.",
        "text": "Let us work towards a fair agreement that benefits us both.",
        "decision": "reject",
        "outcome": None,
        "offer": None,
    }
)


def _backend_reachable(base: str, timeout: float = 0.5) -> bool:
    try:
        urllib.request.urlopen(f"{base}/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


class _FakeOllamaHandler(BaseHTTPRequestHandler):
    """Minimal Ollama wire-protocol shim (non-streaming)."""

    def log_message(self, format, *args):  # noqa: A002 - silence logging
        pass

    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (http.server API)
        # /api/tags is the readiness probe used by both the agents and the
        # test's _run_llm_enabled() check.
        self._send({"models": [{"name": "fake:tiny", "model": "fake:tiny"}]})

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            self.rfile.read(length)  # drain the request body
        model = "fake:tiny"
        # Common metric fields litellm expects when building its response.
        metrics = {
            "model": model,
            "created_at": "2024-01-01T00:00:00Z",
            "done": True,
            "done_reason": "stop",
            "total_duration": 1,
            "load_duration": 1,
            "prompt_eval_count": 1,
            "prompt_eval_duration": 1,
            "eval_count": 1,
            "eval_duration": 1,
        }
        if self.path.endswith("/api/chat"):
            # /api/chat (litellm ollama_chat provider)
            self._send(
                {**metrics, "message": {"role": "assistant", "content": _FAKE_CONTENT}}
            )
        else:
            # /api/generate (raw-urllib agents and litellm ollama provider)
            self._send({**metrics, "response": _FAKE_CONTENT})


@pytest.fixture(scope="session")
def ollama_backend():
    """Guarantee an Ollama-compatible backend on :11434 for LLM tests.

    Yields ``"real"`` if one is already running, otherwise starts the fake
    server, points the relevant env vars at it, and yields ``"fake"``.
    """
    if _backend_reachable(OLLAMA_BASE):
        yield "real"
        return

    server = ThreadingHTTPServer((OLLAMA_HOST, OLLAMA_PORT), _FakeOllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    # Point every client style at the fake. setdefault so an explicit
    # developer override always wins.
    os.environ.setdefault("OLLAMA_HOST", OLLAMA_BASE)
    os.environ.setdefault("OLLAMA_API_BASE", OLLAMA_BASE)
    os.environ.setdefault("OLLAMA_MODEL", "fake:tiny")
    os.environ.setdefault("NEGMAS_LLM_OLLAMA_DEFAULT_MODEL", "fake:tiny")
    # Enable the LLM tests' opt-in gate.
    os.environ.setdefault("HANAGENTS_RUN_LLM", "1")

    try:
        yield "fake"
    finally:
        server.shutdown()
        server.server_close()
