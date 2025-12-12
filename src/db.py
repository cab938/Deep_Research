"""Async database setup for task persistence."""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from sqlalchemy import Column, DateTime, String, Text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

DEFAULT_DB_URL = "sqlite+aiosqlite:///data/thinkdepthai/thinkdepthai.sqlite3"
DB_URL = os.getenv("THINKDEPTH_DB_URL", DEFAULT_DB_URL)


class Base(DeclarativeBase):
    pass


class TaskRecordDB(Base):
    __tablename__ = "tasks"

    task_id = Column(String, primary_key=True)
    status = Column(String, nullable=False)
    request_json = Column(Text, nullable=False)
    result_json = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    pending_action_json = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker | None = None


def _ensure_db_dir(url: str) -> None:
    if url.startswith("sqlite"):
        path_part = url.split("///", 1)[-1]
        db_path = Path(path_part.split("?")[0])
        db_path.parent.mkdir(parents=True, exist_ok=True)


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _ensure_db_dir(DB_URL)
        _engine = create_async_engine(DB_URL, future=True)
    return _engine


def get_sessionmaker() -> async_sessionmaker:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _sessionmaker


@asynccontextmanager
async def session_scope():
    session_factory = get_sessionmaker()
    async with session_factory() as session:
        yield session


async def init_db() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # Enable WAL for better write concurrency on SQLite
        if DB_URL.startswith("sqlite"):
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
            await conn.exec_driver_sql("PRAGMA busy_timeout=5000;")


def json_dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def utcnow() -> datetime:
    return datetime.utcnow()
