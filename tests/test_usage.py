"""Daily question limits: per user, for the whole server, refunds and the usage endpoint."""
from __future__ import annotations

import io

import pytest

from notebook_rag.config import get_settings
from notebook_rag.registry.usage import UsageRepository

pytestmark = pytest.mark.asyncio


async def new_chat_with_doc(client):
    chat = (await client.post("/chats")).json()["chat_id"]
    resp = await client.post(
        "/upload", files={"file": ("a.txt", io.BytesIO(b"Project codename is Bluebird."), "text/plain")},
        data={"chat_id": chat},
    )
    assert resp.status_code == 202
    return chat


async def ask(client, chat, q="codename Bluebird"):
    return await client.post("/query", json={"question": q, "chat_id": chat})


async def test_per_user_limit_blocks_the_next_question(client, other_client, monkeypatch):
    monkeypatch.setattr(get_settings(), "daily_questions_per_user", 2)
    mine, theirs = await new_chat_with_doc(client), await new_chat_with_doc(other_client)

    assert (await ask(client, mine)).status_code == 200
    assert (await ask(client, mine)).status_code == 200
    blocked = await ask(client, mine)
    assert blocked.status_code == 429 and "2" in blocked.json()["detail"]

    assert (await ask(other_client, theirs)).status_code == 200      # another account is unaffected
    usage = (await client.get("/auth/usage")).json()
    assert usage == {"used": 2, "limit": 2, "remaining": 0}


async def test_global_limit_stops_everyone(client, other_client, monkeypatch):
    monkeypatch.setattr(get_settings(), "daily_questions_global", 2)
    mine, theirs = await new_chat_with_doc(client), await new_chat_with_doc(other_client)

    assert (await ask(client, mine)).status_code == 200
    assert (await ask(other_client, theirs)).status_code == 200
    for c, chat in ((client, mine), (other_client, theirs)):
        resp = await ask(c, chat)
        assert resp.status_code == 429 and "everyone" in resp.json()["detail"]


async def test_zero_means_unlimited(client):
    chat = await new_chat_with_doc(client)
    for _ in range(5):
        assert (await ask(client, chat)).status_code == 200
    assert (await client.get("/auth/usage")).json() == {"used": 5, "limit": 0, "remaining": None}


async def test_failed_question_is_refunded(client, wired, monkeypatch):
    monkeypatch.setattr(get_settings(), "daily_questions_per_user", 2)
    chat = await new_chat_with_doc(client)

    def boom(**kwargs):
        raise RuntimeError("LLM is down")

    monkeypatch.setattr(wired.state.generation_service, "answer", boom)
    assert (await ask(client, chat)).status_code == 500
    assert (await client.get("/auth/usage")).json()["used"] == 0     # not charged


async def test_evaluate_costs_several_questions(client, wired, monkeypatch):
    monkeypatch.setattr(get_settings(), "daily_questions_per_user", 4)
    monkeypatch.setattr(get_settings(), "evaluate_cost", 3)

    class FakeEvaluator:
        async def evaluate(self, **kwargs):
            return {"faithfulness": 1.0}

    wired.state.ragas_evaluator = FakeEvaluator()
    body = {"question": "q", "answer": "a", "contexts": ["c"]}
    assert (await client.post("/evaluate", json=body)).status_code == 200
    assert (await client.get("/auth/usage")).json()["used"] == 3
    assert (await client.post("/evaluate", json=body)).status_code == 429     # 3 + 3 > 4


async def test_usage_requires_login(anon_client):
    assert (await anon_client.get("/auth/usage")).status_code == 401


async def test_new_day_starts_fresh(wired):
    repo: UsageRepository = wired.state.usage_repository
    assert (await repo.consume("u1", 1, 2, 0, day="2026-01-01")).allowed
    assert (await repo.consume("u1", 1, 2, 0, day="2026-01-01")).allowed
    denied = await repo.consume("u1", 1, 2, 0, day="2026-01-01")
    assert not denied.allowed and denied.reason == "user"
    assert await repo.used_today("u1", day="2026-01-01") == 2       # a denied request is not recorded
    assert (await repo.consume("u1", 1, 2, 0, day="2026-01-02")).allowed
