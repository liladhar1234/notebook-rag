"""Chats, history, follow-ups and cascading delete, through the HTTP API."""
from __future__ import annotations

import asyncio
import io
import json

import pytest

from notebook_rag.config import get_settings

pytestmark = pytest.mark.asyncio


async def upload_ready(client, chat_id, name, content: str):
    resp = await client.post(
        "/upload", files={"file": (name, io.BytesIO(content.encode()), "text/plain")}, data={"chat_id": chat_id}
    )
    assert resp.status_code == 202, resp.text
    doc_id = resp.json()["doc_id"]
    for _ in range(200):
        body = (await client.get(f"/documents/{doc_id}")).json()
        if body["status"] != "processing":
            assert body["status"] == "ready", body
            return doc_id
        await asyncio.sleep(0.05)
    raise AssertionError("indexing timed out")


async def ask(client, chat_id, question):
    resp = await client.post("/query", json={"question": question, "chat_id": chat_id})
    assert resp.status_code == 200, resp.text
    return resp.json()


async def new_chat(client) -> str:
    resp = await client.post("/chats")
    assert resp.status_code == 201
    return resp.json()["chat_id"]


async def test_create_list_and_unknown_chat(client):
    a, b = await new_chat(client), await new_chat(client)
    assert a != b
    listed = (await client.get("/chats")).json()["chats"]
    assert {c["chat_id"] for c in listed} == {a, b}
    assert all(c["title"] == "New chat" for c in listed)
    assert (await client.get("/chats/nope/messages")).status_code == 404
    assert (await client.delete("/chats/nope")).status_code == 404


async def test_new_chat_starts_from_scratch(client):
    old, fresh = await new_chat(client), await new_chat(client)
    await upload_ready(client, old, "a.txt", "Secret project codename is Bluebird.")
    await ask(client, old, "codename Bluebird")

    assert (await client.get("/documents", params={"chat_id": fresh})).json()["documents"] == []
    assert (await client.get(f"/chats/{fresh}/messages")).json()["messages"] == []
    answer = await ask(client, fresh, "codename Bluebird")
    assert answer["gate_passed"] is False and answer["sources"] == []


async def test_exchange_is_persisted_with_titled_chat_and_filenames(client):
    chat = await new_chat(client)
    await upload_ready(client, chat, "resume.txt", "Skills include Kafka Spark and Hadoop pipelines.")
    answer = await ask(client, chat, "Which skills include Kafka Spark?")

    assert answer["gate_passed"] and "[" not in answer["answer"]  # inline citations stripped
    assert answer["sources"][0]["doc_name"] == "resume.txt"

    msgs = (await client.get(f"/chats/{chat}/messages")).json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["sources"][0]["doc_name"] == "resume.txt" and msgs[1]["gate_passed"] is True
    title = next(c["title"] for c in (await client.get("/chats")).json()["chats"] if c["chat_id"] == chat)
    assert title == "Which skills include Kafka Spark?"


async def test_follow_up_is_rewritten_and_answered_from_the_right_document(client):
    chat = await new_chat(client)
    await upload_ready(client, chat, "alpha.txt", "Alpha plan price is 25 dollars per month.")
    await upload_ready(client, chat, "beta.txt", "Beta plan price is 40 dollars per month.")
    llm = client._transport.app.state.llm

    first = await ask(client, chat, "What is the Beta plan price?")
    assert first["sources"][0]["doc_name"] == "beta.txt"

    llm.plan_reply = json.dumps({"standalone_question": "What is the Alpha plan price?", "search_queries": []})
    follow = await ask(client, chat, "and the other one?")

    assert follow["standalone_question"] == "What is the Alpha plan price?"
    assert "rewrite" in follow["retrieval_mode"]
    assert follow["sources"][0]["doc_name"] == "alpha.txt"
    # the planner saw the previous turn; the answering prompt got the conversation + resolved question
    assert "What is the Beta plan price?" in llm.plan_prompts[-1]
    assert "<conversation>" in llm.answers[-1]
    assert "(Understood as: What is the Alpha plan price?)" in llm.answers[-1]


async def test_planner_failure_still_answers(client):
    chat = await new_chat(client)
    await upload_ready(client, chat, "a.txt", "Kafka is a streaming platform.")
    client._transport.app.state.llm.plan_reply = RuntimeError("quota exceeded")
    answer = await ask(client, chat, "What is Kafka streaming?")
    assert answer["gate_passed"] and answer["standalone_question"] == "What is Kafka streaming?"


async def test_only_recent_history_reaches_the_planner(client):
    chat = await new_chat(client)
    await upload_ready(client, chat, "a.txt", "Kafka is a streaming platform.")
    llm = client._transport.app.state.llm
    for i in range(5):
        await ask(client, chat, f"question number {i} about kafka")
    window = get_settings().history_messages  # messages, i.e. window // 2 past turns
    last_prompt = llm.plan_prompts[-1]
    assert last_prompt.count("User: question number") == window // 2
    assert "question number 0 about kafka" not in last_prompt


async def test_reranker_scores_the_whole_pool_and_gate_blocks_off_topic(client):
    chat = await new_chat(client)
    # ~250-word paragraphs: two never fit in one chunk, so this yields 15 separate chunks.
    paragraphs = [f"alpha{i} common term " + "filler " * 250 for i in range(15)]
    await upload_ready(client, chat, "a.txt", "\n\n".join(paragraphs))
    app = client._transport.app
    result = await ask(client, chat, "common term alpha3")
    assert max(app.state.reranker.seen) > get_settings().retrieval_top_k  # pool, not just top_k
    assert 0 < len(result["sources"]) <= get_settings().retrieval_top_k

    calls_before = len(app.state.llm.answers)
    off = await ask(client, chat, "recipe for chocolate cake")
    assert off["gate_passed"] is False
    assert len(app.state.llm.answers) == calls_before  # gate: no generation call


async def test_delete_chat_removes_documents_vectors_files_and_history(client):
    keep, doomed = await new_chat(client), await new_chat(client)
    kept_doc = await upload_ready(client, keep, "keep.txt", "Kafka stays in the other chat.")
    doc_id = await upload_ready(client, doomed, "gone.txt", "Kafka is deleted with its chat.")
    await ask(client, doomed, "Kafka chat?")
    saved = get_settings().upload_dir / doc_id
    assert saved.exists()

    resp = await client.delete(f"/chats/{doomed}")
    assert resp.status_code == 200 and resp.json()["documents_deleted"] == 1
    assert not saved.exists()
    assert (await client.get(f"/chats/{doomed}/messages")).status_code == 404
    assert (await client.get("/documents", params={"chat_id": doomed})).json()["documents"] == []
    # vectors are gone: re-using the id finds nothing
    assert (await ask(client, doomed, "Kafka deleted chat"))["gate_passed"] is False
    # the other chat is untouched
    assert (await ask(client, keep, "Kafka other chat"))["sources"][0]["doc_id"] == kept_doc


async def test_client_chosen_chat_id_appears_in_the_list(client):
    await upload_ready(client, "my-own-id", "a.txt", "some text about kafka")
    assert "my-own-id" in {c["chat_id"] for c in (await client.get("/chats")).json()["chats"]}
