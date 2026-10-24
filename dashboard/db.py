"""SQLite persistence (stdlib only). Uploaded files stay on disk; the DB stores metadata."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT NOT NULL UNIQUE COLLATE NOCASE,
  email TEXT,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'user' CHECK (role IN ('user','admin')),
  active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL,
  last_login TEXT
);
CREATE TABLE IF NOT EXISTS datasets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  owner_id INTEGER NOT NULL REFERENCES users(id),
  name TEXT NOT NULL,
  original_filename TEXT NOT NULL,
  stored_path TEXT,
  size_bytes INTEGER NOT NULL,
  n_rows INTEGER, n_cols INTEGER,
  status TEXT NOT NULL,            -- rejected | pending_confirmation | ready | deleted
  validation_status TEXT NOT NULL, -- passed | warning | failed
  validation_json TEXT NOT NULL,
  uploaded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL REFERENCES users(id),
  dataset_id INTEGER NOT NULL,     -- 0 = original project dataset
  dataset_name TEXT NOT NULL,
  run_id TEXT NOT NULL,
  model_variant TEXT NOT NULL,
  horizon INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  user_id INTEGER, username TEXT,
  action TEXT NOT NULL,
  detail TEXT, outcome TEXT NOT NULL, ip TEXT
);
CREATE TABLE IF NOT EXISTS evaluations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL, user_id INTEGER, run_id TEXT NOT NULL, result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_user ON predictions(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    conn = sqlite3.connect(config.db_path(), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def audit(action: str, outcome: str = "ok", user: dict | None = None, detail: str = "",
          ip: str | None = None, username: str | None = None) -> None:
    """Append an audit record. Callers must never pass passwords or tokens in `detail`."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO audit_log (ts, user_id, username, action, detail, outcome, ip) VALUES (?,?,?,?,?,?,?)",
            (now(), user["id"] if user else None, (user["username"] if user else username), action,
             detail[:500], outcome, ip))


def rows(sql: str, params: tuple = ()) -> list[dict]:
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def one(sql: str, params: tuple = ()) -> dict | None:
    with connect() as conn:
        r = conn.execute(sql, params).fetchone()
        return dict(r) if r else None


def execute(sql: str, params: tuple = ()) -> int:
    with connect() as conn:
        return conn.execute(sql, params).lastrowid or 0


def loads(text: str | None):
    return json.loads(text) if text else None
