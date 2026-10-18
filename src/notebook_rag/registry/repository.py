"""Document metadata CRUD operations against SQLite.

Tables:
  documents(doc_id PK, chat_id, filename, file_type, chunk_count, created_at,
                 status, error)
"""
from __future__ import annotations

import datetime

from sqlalchemy import String, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from notebook_rag.registry.db import Base


STATUS_PROCESSING = "processing"
STATUS_READY = "ready"
STATUS_FAILED = "failed"


class DocumentRecord(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String, primary_key=True)
    chat_id: Mapped[str] = mapped_column(String, index=True)
    filename: Mapped[str] = mapped_column(String)
    file_type: Mapped[str] = mapped_column(String)
    chunk_count: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[str] = mapped_column(
        String,
        default=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )
    status: Mapped[str] = mapped_column(String, default=STATUS_READY)
    error: Mapped[str | None] = mapped_column(String, nullable=True, default=None)


class DocumentRepository:
    """Async CRUD repository for document metadata."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def upsert(
        self,
        doc_id: str,
        chat_id: str,
        filename: str,
        file_type: str,
        chunk_count: int,
        status: str = STATUS_READY,
        error: str | None = None,
    ) -> DocumentRecord:
        async with self._sf() as session:
            existing = await session.get(DocumentRecord, doc_id)
            if existing:
                existing.chunk_count = chunk_count
                existing.filename = filename
                existing.status = status
                existing.error = error
                await session.commit()
                return existing
            record = DocumentRecord(
                doc_id=doc_id,
                chat_id=chat_id,
                filename=filename,
                file_type=file_type,
                chunk_count=chunk_count,
                status=status,
                error=error,
            )
            session.add(record)
            await session.commit()
            return record

    async def list_by_chat(self, chat_id: str) -> list[DocumentRecord]:
        async with self._sf() as session:
            result = await session.execute(
                select(DocumentRecord).where(DocumentRecord.chat_id == chat_id)
            )
            return list(result.scalars().all())

    async def count_by_chat(self, chat_id: str) -> int:
        """Documents in a chat that count toward the per-chat limit (not failed)."""
        async with self._sf() as session:
            result = await session.execute(
                select(func.count())
                .select_from(DocumentRecord)
                .where(
                    DocumentRecord.chat_id == chat_id,
                    DocumentRecord.status != STATUS_FAILED,
                )
            )
            return int(result.scalar_one())

    async def set_status(
        self,
        doc_id: str,
        status: str,
        chunk_count: int | None = None,
        error: str | None = None,
    ) -> None:
        values: dict = {"status": status, "error": error}
        if chunk_count is not None:
            values["chunk_count"] = chunk_count
        async with self._sf() as session:
            await session.execute(
                update(DocumentRecord)
                .where(DocumentRecord.doc_id == doc_id)
                .values(**values)
            )
            await session.commit()

    async def fail_stale(self) -> int:
        """Mark documents left 'processing' by a crash/restart as failed."""
        async with self._sf() as session:
            result = await session.execute(
                update(DocumentRecord)
                .where(DocumentRecord.status == STATUS_PROCESSING)
                .values(status=STATUS_FAILED, error="Interrupted by a server restart. Please upload again.")
            )
            await session.commit()
            return result.rowcount or 0

    async def list_all(self) -> list[DocumentRecord]:
        async with self._sf() as session:
            result = await session.execute(select(DocumentRecord))
            return list(result.scalars().all())

    async def get(self, doc_id: str) -> DocumentRecord | None:
        async with self._sf() as session:
            return await session.get(DocumentRecord, doc_id)

    async def delete(self, doc_id: str) -> bool:
        async with self._sf() as session:
            result = await session.execute(
                delete(DocumentRecord).where(DocumentRecord.doc_id == doc_id)
            )
            await session.commit()
            return result.rowcount > 0
