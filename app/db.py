"""SQLite access layer. One file DB, WAL mode; schema applied once per process."""
import sqlite3
import threading
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent / "data"
DB_PATH = DATA_DIR / "aesthetically.db"
SCHEMA_PATH = APP_DIR / "schema.sql"

_schema_lock = threading.Lock()
_schema_applied: set[str] = set()


def get_conn(db_path: Path | None = None) -> sqlite3.Connection:
    db_path = db_path or DB_PATH  # late-bound so tests can monkeypatch DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=15000")
    key = str(db_path.resolve())
    if key not in _schema_applied:
        with _schema_lock:
            if key not in _schema_applied:
                conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
                _schema_applied.add(key)
    return conn
