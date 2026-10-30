"""PostgreSQL connection and ZyWise persistence models."""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def load_local_env() -> None:
    """Load simple KEY=VALUE settings from the project .env file."""
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        # The project .env file is the source of truth for this local app.
        os.environ[key.strip()] = value.strip().strip('"').strip("'")


load_local_env()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@/zywise?host=/tmp",
)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    file_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    status: Mapped[str] = mapped_column(String(40), default="uploaded", nullable=False)


class ProcessingRun(Base):
    __tablename__ = "processing_runs"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    file_id: Mapped[int] = mapped_column(ForeignKey("uploaded_files.file_id"), nullable=False)
    rows_before: Mapped[int] = mapped_column(Integer, nullable=False)
    rows_after: Mapped[int] = mapped_column(Integer, nullable=False)
    issue_summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class Report(Base):
    __tablename__ = "reports"

    report_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("processing_runs.run_id"), nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    recommendations_json: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


def refresh_connection_settings() -> None:
    """Reload .env settings and discard any engine created with old settings."""
    global DATABASE_URL, _engine, _session_factory
    load_local_env()
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@/zywise?host=/tmp",
    )
    _engine = None
    _session_factory = None


def init_db() -> None:
    """Create ZyWise tables if they do not already exist."""
    Base.metadata.create_all(get_engine())


def get_session_factory():
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_user(name: str, email: str, password_hash: str) -> dict[str, Any] | None:
    """Create a user and return a safe session representation."""
    with session_scope() as session:
        existing = session.scalar(select(User).where(User.email == email))
        if existing is not None:
            return None
        user = User(name=name, email=email, password_hash=password_hash)
        session.add(user)
        session.flush()
        return {"user_id": user.user_id, "name": user.name, "email": user.email}


def authenticate_user(email: str, password_hash: str) -> dict[str, Any] | None:
    with session_scope() as session:
        user = session.scalar(select(User).where(User.email == email))
        if user is None or user.password_hash != password_hash:
            return None
        return {"user_id": user.user_id, "name": user.name, "email": user.email}
