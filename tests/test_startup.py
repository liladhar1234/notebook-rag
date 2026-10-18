"""Runs the REAL lifespan in main.py (wiring) with fake models, then a chat round-trip."""
from __future__ import annotations

from fastapi.testclient import TestClient

from notebook_rag.api import main
from tests.conftest import FakeDense, FakeLLM, FakeReranker, FakeSparse


def test_lifespan_wires_chats_planner_and_serves_a_conversation(monkeypatch, tmp_path):
    from notebook_rag.config import get_settings

    # Real lifespan, but on temp folders: never your real data/ (and no clash with a running app).
    cfg = get_settings()
    monkeypatch.setattr(cfg, "qdrant_path", tmp_path / "qdrant")
    monkeypatch.setattr(cfg, "registry_db_path", tmp_path / "reg.db")
    monkeypatch.setattr(cfg, "upload_dir", tmp_path / "uploads")

    llm = FakeLLM()
    monkeypatch.setattr(main, "DenseEmbedder", lambda name: FakeDense())
    monkeypatch.setattr(main, "SparseEmbedder", lambda name: FakeSparse())
    monkeypatch.setattr(main, "CrossEncoderReranker", lambda model_name: FakeReranker())
    monkeypatch.setattr(main, "build_llm_client", lambda **kw: llm)
    monkeypatch.setattr(main, "RagasEvaluator", lambda **kw: object())

    with TestClient(main.create_app()) as client:  # `with` runs startup and shutdown
        assert client.post("/chats").status_code == 401  # login required
        reg = client.post("/auth/register", json={"email": "startup@example.com", "password": "password123"})
        assert reg.status_code == 201
        chat = client.post("/chats").json()["chat_id"]
        up = client.post(
            "/upload",
            files={"file": ("n.txt", b"Kafka and Spark pipelines for streaming.", "text/plain")},
            data={"chat_id": chat},
        )
        assert up.status_code == 202

        answer = client.post("/query", json={"question": "kafka spark streaming", "chat_id": chat}).json()
        assert answer["gate_passed"] and answer["sources"][0]["doc_name"] == "n.txt"
        assert "multi_query" in answer["retrieval_mode"]  # planner ran (variants enabled by default)

        msgs = client.get(f"/chats/{chat}/messages").json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert client.delete(f"/chats/{chat}").json()["documents_deleted"] == 1
