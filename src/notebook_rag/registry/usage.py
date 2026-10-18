"""Daily question quota per user, plus one shared cap for the whole server.

Why: every question spends requests from ONE Gemini key. Without limits, a few
visitors could use up the free-tier quota and the app would stop answering for
everyone. Days are counted in UTC (reset at 00:00 UTC = 05:30 IST).

Table: usage(user_id, day, count) - one row per user per day.
"""
from __future__ import annotations

import datetime
from typing import NamedTuple

from sqlalchemy import String, case, func, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from notebook_rag.registry.db import Base


def today_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).date().isoformat()


class UsageRecord(Base):
    __tablename__ = "usage"

    user_id: Mapped[str] = mapped_column(String, primary_key=True)
    day: Mapped[str] = mapped_column(String, primary_key=True)
    count: Mapped[int] = mapped_column(default=0)


class Quota(NamedTuple):
    allowed: bool
    reason: str | None   # None | "user" | "global"
    day: str
    used: int            # this user's count today (after this request if allowed)
    limit: int           # per-user limit, 0 = unlimited


def limit_message(q: Quota) -> str:
    if q.reason == "global":
        return "This app has reached its daily question limit for everyone. Please try again tomorrow."
    return (
        f"You've used all {q.limit} of today's questions. "
        "The limit resets at midnight UTC (05:30 IST)."
    )


class UsageRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def consume(
        self, user_id: str, cost: int, per_user: int, global_limit: int, day: str | None = None
    ) -> Quota:
        """Reserve `cost` questions. A limit of 0 means unlimited. Nothing is written when denied."""
        day = day or today_utc()
        async with self._sf() as session:
            await session.execute(
                sqlite_insert(UsageRecord).values(user_id=user_id, day=day, count=0).on_conflict_do_nothing()
            )
            used = (
                await session.execute(
                    select(UsageRecord.count).where(UsageRecord.user_id == user_id, UsageRecord.day == day)
                )
            ).scalar_one()
            total = (
                await session.execute(
                    select(func.coalesce(func.sum(UsageRecord.count), 0)).where(UsageRecord.day == day)
                )
            ).scalar_one()
            if per_user and used + cost > per_user:
                return Quota(False, "user", day, used, per_user)
            if global_limit and total + cost > global_limit:
                return Quota(False, "global", day, used, per_user)
            await session.execute(
                update(UsageRecord)
                .where(UsageRecord.user_id == user_id, UsageRecord.day == day)
                .values(count=UsageRecord.count + cost)  # done in SQL: safe with concurrent requests
            )
            await session.commit()
            return Quota(True, None, day, used + cost, per_user)

    async def refund(self, user_id: str, cost: int, day: str) -> None:
        """Give the questions back when the request failed on our side."""
        async with self._sf() as session:
            await session.execute(
                update(UsageRecord)
                .where(UsageRecord.user_id == user_id, UsageRecord.day == day)
                .values(count=case((UsageRecord.count > cost, UsageRecord.count - cost), else_=0))
            )
            await session.commit()

    async def used_today(self, user_id: str, day: str | None = None) -> int:
        day = day or today_utc()
        async with self._sf() as session:
            row = await session.get(UsageRecord, (user_id, day))
            return row.count if row else 0
