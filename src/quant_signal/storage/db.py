"""Database engine and session helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from quant_signal.core.config import get_settings
from quant_signal.storage.base import Base


def _build_engine(database_url: str) -> Engine:
    """Create a SQLAlchemy engine for the given database URL."""

    engine_kwargs: dict[str, object] = {"pool_pre_ping": True}
    if database_url.startswith("sqlite"):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(database_url, **engine_kwargs)


@cache
def _get_engine(database_url: str) -> Engine:
    """Return a cached engine keyed by database URL."""

    return _build_engine(database_url)


def clear_engine_cache() -> None:
    """Clear the engine cache for tests and environment changes."""

    _get_engine.cache_clear()


def get_engine(database_url: str | None = None) -> Engine:
    """Return an engine for the configured database."""

    resolved_url = database_url or get_settings().database_url
    return _get_engine(resolved_url)


def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    """Return a configured SQLAlchemy session factory."""

    return sessionmaker(bind=get_engine(database_url), autoflush=False, expire_on_commit=False)


@contextmanager
def session_scope(database_url: str | None = None) -> Iterator[Session]:
    """Yield a transactional database session."""

    session = get_session_factory(database_url)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database_connection(database_url: str | None = None) -> bool:
    """Return True when the database connection can execute a simple query."""

    with get_engine(database_url).connect() as connection:
        connection.execute(text("SELECT 1"))
    return True


def create_all_tables(database_url: str | None = None) -> None:
    """Create all tables defined in metadata."""

    Base.metadata.create_all(bind=get_engine(database_url))
