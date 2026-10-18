"""Chats and their message history (SQLite).

Tables:
  chats(chat_id PK, user_id, title, created_at, updated_at)
  messages(id PK, chat_id, role, content, sources_json, gate_passed, retrieval_mode, created_at)

A chat is the unit of isolation: its documents, vectors and messages never mix
with another chat's, and every chat belongs to one user (user_id). Chats created
before accounts existed have user_id NULL until the first account adopts them.
Deleting a chat is cascaded by the route (see api/routes/chats.py).
"""
from __future__ import annotations

import datetime
import json
import uuid

from sqlalchemy import String, Text, delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from notebook_rag.registry.db import Base

DEFAULT_TITLE = "New chat"
_TITLE_MAX = 60


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _title_from(question: str) -> str:
    text = " ".join(question.split())
    return text if len(text) <= _TITLE_MAX else text[: _TITLE_MAX - 1].rstrip() + "…"


class ChatRecord(Base):
    __tablename__ = "chats"

    chat_id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True, default=None, index=True)
    title: Mapped[str] = mapped_column(String, default=DEFAULT_TITLE)
    created_at: Mapped[str] = mapped_column(String, default=_now)
    updated_at: Mapped[str] = mapped_column(String, default=_now, index=True)


class MessageRecord(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    chat_id: Mapped[str] = mapped_column(String, index=True)
    role: Mapped[str] = mapped_column(String)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    sources_json: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    gate_passed: Mapped[bool | None] = mapped_column(nullable=True, default=None)
    retrieval_mode: Mapped[str | None] = mapped_column(String, nullable=True, default=None)
    created_at: Mapped[str] = mapped_column(String, default=_now)

    @property
    def sources(self) -> list[dict]:
        return json.loads(self.sources_json) if self.sources_json else []


class ChatRepository:
    """Async CRUD for chats and their messages."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def create(
        self, user_id: str | None, chat_id: str | None = None, title: str = DEFAULT_TITLE
    ) -> ChatRecord:
        """Create a chat for `user_id` (server-generated id by default). Idempotent for an existing id."""
        chat_id = chat_id or uuid.uuid4().hex[:16]
        async with self._sf() as session:
            existing = await session.get(ChatRecord, chat_id)
            if existing:
                return existing
            record = ChatRecord(chat_id=chat_id, user_id=user_id, title=title)
            session.add(record)
            await session.commit()
            return record

    async def ensure(self, user_id: str, chat_id: str, title: str = DEFAULT_TITLE) -> ChatRecord | None:
        """The user's chat with this id, created first if the id is new.

        Returns None when the id belongs to someone else: callers answer 404, so
        another user's chat can neither be read nor taken over.
        """
        chat = await self.create(user_id, chat_id, title)
        return chat if chat.user_id == user_id else None

    async def ensure_legacy(self, chat_id: str, title: str = DEFAULT_TITLE) -> None:
        """Startup only: give a pre-accounts chat (documents without a chat row) a row, owner NULL."""
        await self.create(None, chat_id, title)

    async def get(self, chat_id: str) -> ChatRecord | None:
        async with self._sf() as session:
            return await session.get(ChatRecord, chat_id)

    async def get_owned(self, user_id: str, chat_id: str) -> ChatRecord | None:
        """The chat only if it exists AND belongs to `user_id`."""
        chat = await self.get(chat_id)
        return chat if chat is not None and chat.user_id == user_id else None

    async def list(self, user_id: str) -> list[ChatRecord]:
        """The user's chats, most recently active first."""
        async with self._sf() as session:
            result = await session.execute(
                select(ChatRecord).where(ChatRecord.user_id == user_id).order_by(ChatRecord.updated_at.desc())
            )
            return list(result.scalars().all())

    async def add_exchange(
        self,
        chat_id: str,
        question: str,
        answer: str,
        sources: list[dict],
        gate_passed: bool,
        retrieval_mode: str,
    ) -> None:
        """Store one question/answer pair; the first question becomes the chat's title."""
        async with self._sf() as session:
            chat = await session.get(ChatRecord, chat_id)
            if chat is None:
                chat = ChatRecord(chat_id=chat_id)
                session.add(chat)
            if chat.title == DEFAULT_TITLE:
                chat.title = _title_from(question)
            chat.updated_at = _now()
            session.add(MessageRecord(chat_id=chat_id, role="user", content=question))
            session.add(
                MessageRecord(
                    chat_id=chat_id,
                    role="assistant",
                    content=answer,
                    sources_json=json.dumps(sources),
                    gate_passed=gate_passed,
                    retrieval_mode=retrieval_mode,
                )
            )
            await session.commit()

    async def recent_messages(self, chat_id: str, limit: int) -> list[MessageRecord]:
        """The last `limit` messages, oldest first."""
        if limit <= 0:
            return []
        async with self._sf() as session:
            result = await session.execute(
                select(MessageRecord)
                .where(MessageRecord.chat_id == chat_id)
                .order_by(MessageRecord.id.desc())
                .limit(limit)
            )
            return list(reversed(result.scalars().all()))

    async def messages(self, chat_id: str) -> list[MessageRecord]:
        async with self._sf() as session:
            result = await session.execute(
                select(MessageRecord).where(MessageRecord.chat_id == chat_id).order_by(MessageRecord.id)
            )
            return list(result.scalars().all())

    async def delete(self, chat_id: str) -> bool:
        async with self._sf() as session:
            await session.execute(delete(MessageRecord).where(MessageRecord.chat_id == chat_id))
            result = await session.execute(delete(ChatRecord).where(ChatRecord.chat_id == chat_id))
            await session.commit()
            return (result.rowcount or 0) > 0
