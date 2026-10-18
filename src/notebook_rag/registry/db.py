"""SQLAlchemy async engine and session factory for the document registry."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


def make_engine(db_path: Path):
    url = f"sqlite+aiosqlite:///{db_path}"
    return create_async_engine(url, echo=False, future=True)


def make_session_factory(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def migrate(conn) -> None:
    """Add columns introduced after the first release to an existing SQLite file.

    `create_all` only creates missing tables, it never alters existing ones.
    """
    from sqlalchemy import text

    rows = (await conn.execute(text("PRAGMA table_info(documents)"))).fetchall()
    have = {r[1] for r in rows}
    if "status" not in have:
        await conn.execute(
            text("ALTER TABLE documents ADD COLUMN status VARCHAR DEFAULT 'ready'")
        )
    if "error" not in have:
        await conn.execute(text("ALTER TABLE documents ADD COLUMN error VARCHAR"))

    chat_cols = {r[1] for r in (await conn.execute(text("PRAGMA table_info(chats)"))).fetchall()}
    # (a brand-new database already has the column from create_all; chat_cols is empty only
    # if the chats table does not exist at all, which create_all prevents)
    if chat_cols and "user_id" not in chat_cols:
        await conn.execute(text("ALTER TABLE chats ADD COLUMN user_id VARCHAR"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS ix_chats_user_id ON chats (user_id)"))
