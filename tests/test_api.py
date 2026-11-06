import hashlib
import re

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient

from app import config, embed as embed_mod, ingest, llm, rag

DIM = 256


def fake_embed(texts):
    """Deterministic bag-of-words embedder so tests need no model download."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z]+", t.lower()):
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1
    n = np.linalg.norm(out, axis=1, keepdims=True)
    n[n == 0] = 1
    return out / n


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(embed_mod, "embed", fake_embed)
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path)
    monkeypatch.setattr(rag, "_index", None)
    ingest.build_index(
        [
            {"text": "Maternity leave: employees receive twelve weeks of paid maternity leave.", "source": "handbook.pdf"},
            {"text": "Dress code: business casual is expected in the office.", "source": "handbook.pdf"},
        ],
        tmp_path,
    )
    from app.main import app

    return TestClient(app)


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_uses_context_and_returns_sources(client, monkeypatch):
    seen = {}

    def fake_chat(system, user):
        seen["user"] = user
        return "Twelve weeks."

    monkeypatch.setattr(llm, "chat", fake_chat)
    r = client.post("/ask", json={"question": "How long is maternity leave?"})
    assert r.status_code == 200
    assert r.json() == {"answer": "Twelve weeks.", "sources": ["handbook.pdf"]}
    assert "twelve weeks" in seen["user"].lower()


def test_irrelevant_question_skips_llm(client, monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a: pytest.fail("LLM should not be called"))
    r = client.post("/ask", json={"question": "zzzz qqqq xxxx"})
    assert r.json()["answer"] == "NOT FOUND IN CONTEXT"


def test_empty_and_too_long_question(client):
    assert client.post("/ask", json={"question": "   "}).json() == {"answer": "Enter Non-Null Question"}
    assert client.post("/ask", json={"question": "a" * 501}).status_code == 400


def test_missing_key_is_503(client, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    r = client.post("/ask", json={"question": "maternity leave"})
    assert r.status_code == 503


def test_rate_limit_maps_to_429(client, monkeypatch):
    def boom(*a):
        resp = requests.Response()
        resp.status_code = 429
        raise requests.HTTPError(response=resp)

    monkeypatch.setattr(llm, "chat", boom)
    assert client.post("/ask", json={"question": "maternity leave"}).status_code == 429
