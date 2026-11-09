"""
Database Session Management and Engine Initialization.
Supports PostgreSQL for production and local SQLite for lightweight testing.
"""

import os
import tempfile
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from backend.app.core.config import settings

def is_dir_writable(dir_path: str) -> bool:
    try:
        test_file = os.path.join(dir_path, f".test_write_{os.getpid()}")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        return True
    except Exception:
        return False

db_url = settings.DATABASE_URL
if db_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False, "timeout": 15}
    if ":///" in db_url:
        path = db_url.split(":///", 1)[1]
        parent_dir = os.path.dirname(os.path.abspath(path))
        if not is_dir_writable(parent_dir):
            temp_db = os.path.join(tempfile.gettempdir(), "signature_dev.db").replace("\\", "/")
            db_url = f"sqlite:///{temp_db}"
else:
    connect_args = {}

engine = create_engine(
    db_url,
    connect_args=connect_args,
    pool_pre_ping=True
)

if db_url.startswith("sqlite"):
    from sqlalchemy import event
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency provider for FastAPI route handlers."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
