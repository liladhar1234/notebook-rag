"""Accounts and login sessions (SQLite).

Tables:
  users(user_id PK, email UNIQUE, password_hash, created_at)
  sessions(token_hash PK, user_id, created_at, expires_at)

Only a SHA-256 hash of the session token is stored, so a leaked database
cannot be used to log in. Passwords are hashed in api/security.py.
"""
from __future__ import annotations

import datetime
import hashlib
import uuid

from sqlalchemy import String, delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from notebook_rag.registry.chats import ChatRecord
from notebook_rag.registry.db import Base


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _iso(dt: datetime.datetime) -> str:
    return dt.isoformat()


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class UserRecord(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String)
    created_at: Mapped[str] = mapped_column(String, default=lambda: _iso(_now()))


class SessionRecord(Base):
    __tablename__ = "sessions"

    token_hash: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str] = mapped_column(String, index=True)
    created_at: Mapped[str] = mapped_column(String, default=lambda: _iso(_now()))
    expires_at: Mapped[str] = mapped_column(String)


class EmailTaken(Exception):
    pass


class UserRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def create(self, email: str, password_hash: str) -> UserRecord:
        """Create an account. The very first account also adopts all chats that
        existed before accounts did (chats.user_id IS NULL)."""
        async with self._sf() as session:
            is_first = (await session.execute(select(func.count()).select_from(UserRecord))).scalar_one() == 0
            user = UserRecord(user_id=uuid.uuid4().hex, email=email, password_hash=password_hash)
            session.add(user)
            try:
                await session.flush()
            except IntegrityError as exc:
                await session.rollback()
                raise EmailTaken(email) from exc
            if is_first:
                await session.execute(
                    update(ChatRecord).where(ChatRecord.user_id.is_(None)).values(user_id=user.user_id)
                )
            await session.commit()
            return user

    async def by_email(self, email: str) -> UserRecord | None:
        async with self._sf() as session:
            result = await session.execute(select(UserRecord).where(UserRecord.email == email))
            return result.scalar_one_or_none()

    async def get(self, user_id: str) -> UserRecord | None:
        async with self._sf() as session:
            return await session.get(UserRecord, user_id)

    async def create_session(self, user_id: str, token: str, days: int) -> None:
        async with self._sf() as session:
            session.add(
                SessionRecord(
                    token_hash=hash_token(token),
                    user_id=user_id,
                    expires_at=_iso(_now() + datetime.timedelta(days=days)),
                )
            )
            await session.commit()

    async def user_for_token(self, token: str) -> UserRecord | None:
        """The user behind a valid, unexpired session token, else None."""
        async with self._sf() as session:
            rec = await session.get(SessionRecord, hash_token(token))
            if rec is None:
                return None
            if datetime.datetime.fromisoformat(rec.expires_at) < _now():
                await session.delete(rec)
                await session.commit()
                return None
            return await session.get(UserRecord, rec.user_id)

    async def delete_session(self, token: str) -> None:
        async with self._sf() as session:
            await session.execute(delete(SessionRecord).where(SessionRecord.token_hash == hash_token(token)))
            await session.commit()
