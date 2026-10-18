"""Accounts, login sessions, and per-user isolation of chats and documents."""
from __future__ import annotations

import asyncio
import io

import pytest

from notebook_rag.api.security import LoginLimiter, hash_password, verify_password
from tests.conftest import PASSWORD, register

pytestmark = pytest.mark.asyncio


async def upload_ready(client, chat_id, name="a.txt", content="Secret project codename is Bluebird."):
    resp = await client.post(
        "/upload", files={"file": (name, io.BytesIO(content.encode()), "text/plain")}, data={"chat_id": chat_id}
    )
    assert resp.status_code == 202, resp.text
    doc_id = resp.json()["doc_id"]
    for _ in range(200):
        body = (await client.get(f"/documents/{doc_id}")).json()
        if body["status"] != "processing":
            return doc_id
        await asyncio.sleep(0.05)
    raise AssertionError("indexing timed out")


# ── password hashing (pure functions) ────────────────────────────────────────

def test_password_hash_roundtrip_and_salt():
    h1, h2 = hash_password("hunter2hunter2"), hash_password("hunter2hunter2")
    assert h1 != h2                                   # random salt
    assert verify_password("hunter2hunter2", h1)
    assert not verify_password("hunter2hunter3", h1)
    assert not verify_password("x", "garbage")        # malformed hash never logs in


def test_login_limiter_blocks_then_resets():
    lim = LoginLimiter(max_failures=3, window_s=60)
    for _ in range(3):
        assert not lim.blocked("k")
        lim.record_failure("k")
    assert lim.blocked("k") and not lim.blocked("other")
    lim.reset("k")
    assert not lim.blocked("k")


# ── register / login / logout ────────────────────────────────────────────────

async def test_data_routes_require_login(anon_client):
    for method, url in [("get", "/chats"), ("post", "/chats"), ("get", "/documents?chat_id=x"),
                        ("get", "/chats/x/messages"), ("delete", "/chats/x"), ("get", "/auth/me")]:
        resp = await getattr(anon_client, method)(url)
        assert resp.status_code == 401, (method, url)
    assert (await anon_client.post("/query", json={"question": "hi", "chat_id": "x"})).status_code == 401
    assert (await anon_client.post("/upload", data={"chat_id": "x"},
                                   files={"file": ("a.txt", b"hello", "text/plain")})).status_code == 401
    assert (await anon_client.get("/health")).status_code == 200   # health stays public


async def test_register_sets_cookie_and_me_works(anon_client):
    user = await register(anon_client, "  New@Example.COM ")
    assert user["email"] == "new@example.com"                       # normalised
    set_cookie = (await anon_client.get("/auth/me")).status_code
    assert set_cookie == 200


async def test_register_validation_and_duplicates(anon_client):
    bad_email = await anon_client.post("/auth/register", json={"email": "nope", "password": PASSWORD})
    short_pw = await anon_client.post("/auth/register", json={"email": "a@b.co", "password": "short"})
    assert bad_email.status_code == 422 and short_pw.status_code == 422
    await register(anon_client, "dup@example.com")
    again = await anon_client.post("/auth/register", json={"email": "DUP@example.com", "password": PASSWORD})
    assert again.status_code == 409


async def test_login_wrong_password_unknown_user_and_success(client, wired):
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=wired), base_url="http://test") as ac:
        wrong = await ac.post("/auth/login", json={"email": "tester@example.com", "password": "wrong-password"})
        unknown = await ac.post("/auth/login", json={"email": "ghost@example.com", "password": PASSWORD})
        assert wrong.status_code == unknown.status_code == 401
        assert wrong.json()["detail"] == unknown.json()["detail"]   # no hint which part was wrong
        ok = await ac.post("/auth/login", json={"email": "Tester@Example.com", "password": PASSWORD})
        assert ok.status_code == 200
        assert (await ac.get("/auth/me")).json()["email"] == "tester@example.com"


async def test_login_is_rate_limited(client, wired):
    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=wired), base_url="http://test") as ac:
        codes = [
            (await ac.post("/auth/login", json={"email": "tester@example.com", "password": f"bad-{i}-pass"})).status_code
            for i in range(6)
        ]
    assert codes[:5] == [401] * 5 and codes[5] == 429


async def test_logout_ends_the_session(client):
    assert (await client.get("/chats")).status_code == 200
    assert (await client.post("/auth/logout")).status_code == 204
    assert (await client.get("/chats")).status_code == 401


async def test_signup_can_be_closed(anon_client, monkeypatch):
    from notebook_rag.config import get_settings

    monkeypatch.setattr(get_settings(), "allow_signup", False)
    resp = await anon_client.post("/auth/register", json={"email": "x@example.com", "password": PASSWORD})
    assert resp.status_code == 403


async def test_expired_session_is_rejected(client, wired):
    from sqlalchemy import update

    from notebook_rag.registry.users import SessionRecord

    users = wired.state.user_repository
    async with users._sf() as session:
        await session.execute(update(SessionRecord).values(expires_at="2000-01-01T00:00:00+00:00"))
        await session.commit()
    assert (await client.get("/auth/me")).status_code == 401


# ── isolation between users ──────────────────────────────────────────────────

async def test_users_only_see_their_own_chats(client, other_client):
    mine = (await client.post("/chats")).json()["chat_id"]
    theirs = (await other_client.post("/chats")).json()["chat_id"]
    assert {c["chat_id"] for c in (await client.get("/chats")).json()["chats"]} == {mine}
    assert {c["chat_id"] for c in (await other_client.get("/chats")).json()["chats"]} == {theirs}


async def test_other_users_chat_is_untouchable(client, other_client):
    mine = (await client.post("/chats")).json()["chat_id"]
    doc_id = await upload_ready(client, mine)
    await client.post("/query", json={"question": "codename Bluebird", "chat_id": mine})

    # Every way of reaching someone else's chat answers 404, same as a chat that doesn't exist.
    assert (await other_client.get(f"/chats/{mine}/messages")).status_code == 404
    assert (await other_client.delete(f"/chats/{mine}")).status_code == 404
    assert (await other_client.get("/documents", params={"chat_id": mine})).status_code == 404
    assert (await other_client.get(f"/documents/{doc_id}")).status_code == 404
    assert (await other_client.delete(f"/documents/{doc_id}")).status_code == 404
    assert (await other_client.post("/query", json={"question": "codename", "chat_id": mine})).status_code == 404
    up = await other_client.post("/upload", data={"chat_id": mine},
                                 files={"file": ("x.txt", b"injected", "text/plain")})
    assert up.status_code == 404
    ev = await other_client.post("/evaluate", json={"question": "codename", "chat_id": mine})
    assert ev.status_code == 404

    # ...and nothing of mine was changed by those attempts.
    docs = (await client.get("/documents", params={"chat_id": mine})).json()["documents"]
    assert [d["doc_id"] for d in docs] == [doc_id]
    assert len((await client.get(f"/chats/{mine}/messages")).json()["messages"]) == 2


async def test_same_question_in_another_users_chat_finds_nothing(client, other_client):
    mine = (await client.post("/chats")).json()["chat_id"]
    await upload_ready(client, mine)
    theirs = (await other_client.post("/chats")).json()["chat_id"]
    resp = await other_client.post("/query", json={"question": "codename Bluebird", "chat_id": theirs})
    assert resp.status_code == 200
    assert resp.json()["gate_passed"] is False and resp.json()["sources"] == []


# ── data from before accounts existed ────────────────────────────────────────

async def test_first_account_adopts_old_chats_later_ones_do_not(anon_client, wired):
    chats = wired.state.chat_repository
    await chats.ensure_legacy("default-session", title="default-session")   # owner NULL, as after the upgrade

    first = await register(anon_client, "first@example.com")
    listed = (await anon_client.get("/chats")).json()["chats"]
    assert [c["chat_id"] for c in listed] == ["default-session"]
    assert (await chats.get("default-session")).user_id == first["user_id"]

    from httpx import ASGITransport, AsyncClient

    async with AsyncClient(transport=ASGITransport(app=wired), base_url="http://test") as ac:
        await register(ac, "second@example.com")
        assert (await ac.get("/chats")).json()["chats"] == []
        assert (await ac.get("/chats/default-session/messages")).status_code == 404
