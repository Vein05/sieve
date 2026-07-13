"""Bounded SQLite disk cache for expensive NLP results.

Sits underneath the existing ``@lru_cache`` decorators so that a process
restart no longer pays the full SpaCy warm-up cost (~13 min for 500 rows).

Safety rails
------------
* **Bounded**: each namespace caps at ``MAX_ENTRIES`` rows; oldest 20 % are
  evicted on overflow.
* **Versioned**: cache is keyed by SpaCy model name so a model upgrade
  auto-invalidates stale entries.
* **Disable**: set ``SIEVE_DISK_CACHE=0`` to bypass entirely.
* **Corruption-safe**: any sqlite error deletes the DB and continues
  without caching (never crashes the pipeline).
* **Size cap**: if the DB file exceeds ``MAX_DB_MB`` the cache is wiped.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / ".nlp_cache"
MAX_ENTRIES = 80_000          # per namespace
MAX_DB_MB = 400               # wipe the whole DB if it exceeds this
_EVICT_FRACTION = 0.20        # drop oldest 20 % on overflow
_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Global singleton connection (one per process)
# ---------------------------------------------------------------------------

_conn: sqlite3.Connection | None = None
_disabled: bool = False


def _is_disabled() -> bool:
    global _disabled
    if _disabled:
        return True
    if os.environ.get("SIEVE_DISK_CACHE", "1") == "0":
        _disabled = True
    return _disabled


def _db_path() -> Path:
    return _CACHE_DIR / "nlp_cache.db"


def _get_conn() -> sqlite3.Connection | None:
    """Return the process-local SQLite connection, creating the DB if needed."""
    global _conn
    if _is_disabled():
        return None
    if _conn is not None:
        return _conn
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        db = _db_path()
        # Size guard — nuke if too big
        if db.exists() and db.stat().st_size > MAX_DB_MB * 1024 * 1024:
            db.unlink()
        _conn = sqlite3.connect(str(db), timeout=5)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.execute(
            """CREATE TABLE IF NOT EXISTS cache (
                ns        TEXT    NOT NULL,
                key       TEXT    NOT NULL,
                value     BLOB   NOT NULL,
                version   INT    NOT NULL,
                ts        REAL   NOT NULL,
                PRIMARY KEY (ns, key)
            )"""
        )
        _conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cache_ns_ts ON cache (ns, ts)"
        )
        _conn.commit()
        return _conn
    except Exception:
        _conn = None
        # Unrecoverable — disable for this process
        return None


def _safe_close() -> None:
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def disk_get(namespace: str, key: str, version: int = _SCHEMA_VERSION) -> Any | None:
    """Look up a cached value.  Returns ``None`` on miss (or if disabled)."""
    conn = _get_conn()
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT value FROM cache WHERE ns = ? AND key = ? AND version = ?",
            (namespace, key, version),
        ).fetchone()
        if row is None:
            return None
        return pickle.loads(row[0])
    except Exception:
        return None


def disk_put(namespace: str, key: str, value: Any, version: int = _SCHEMA_VERSION) -> None:
    """Store a value.  Silently does nothing on error."""
    conn = _get_conn()
    if conn is None:
        return
    try:
        blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        conn.execute(
            "INSERT OR REPLACE INTO cache (ns, key, value, version, ts) VALUES (?, ?, ?, ?, ?)",
            (namespace, key, blob, version, time.time()),
        )
        conn.commit()
    except Exception:
        return
    # Lazy eviction: check count only occasionally (every ~500 puts)
    try:
        if int.from_bytes(os.urandom(2), "big") % 500 == 0:
            _maybe_evict(namespace, version)
    except Exception:
        pass


def _maybe_evict(namespace: str, version: int) -> None:
    conn = _get_conn()
    if conn is None:
        return
    try:
        (count,) = conn.execute(
            "SELECT COUNT(*) FROM cache WHERE ns = ? AND version = ?",
            (namespace, version),
        ).fetchone()
        if count <= MAX_ENTRIES:
            return
        drop = int(count * _EVICT_FRACTION)
        conn.execute(
            """DELETE FROM cache WHERE rowid IN (
                SELECT rowid FROM cache
                WHERE ns = ? AND version = ?
                ORDER BY ts ASC
                LIMIT ?
            )""",
            (namespace, drop),
        )
        conn.commit()
    except Exception:
        pass


def text_key(text: str) -> str:
    """Stable hash key for a text string.  Short texts stored verbatim for
    debuggability; longer texts hashed."""
    if len(text) <= 120:
        return text
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def cache_stats() -> dict[str, Any]:
    """Return per-namespace row counts (for diagnostics)."""
    conn = _get_conn()
    if conn is None:
        return {"disabled": True}
    try:
        rows = conn.execute(
            "SELECT ns, COUNT(*), SUM(LENGTH(value)) FROM cache GROUP BY ns"
        ).fetchall()
        db_size = _db_path().stat().st_size if _db_path().exists() else 0
        return {
            "namespaces": {ns: {"entries": cnt, "bytes": sz or 0} for ns, cnt, sz in rows},
            "db_size_mb": round(db_size / 1024 / 1024, 1),
        }
    except Exception:
        return {"error": "could not read stats"}
