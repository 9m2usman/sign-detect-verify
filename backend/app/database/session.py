"""
Database Session Management and Engine Initialization.
Supports PostgreSQL for production and local SQLite for lightweight testing.
"""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from backend.app.core.config import settings

# SQLite needs connect_args for multithreading in FastAPI
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency provider for FastAPI route handlers."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
