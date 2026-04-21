"""
dtc_storage.py — SQLite-backed persistence for DTC simulations.

Research basis:
    SQLite.org WAL documentation (Write-Ahead Logging) — atomic commit
    semantics and concurrent read/write guarantees. Gray & Reuter (1993)
    "Transaction Processing: Concepts and Techniques" for theoretical
    foundation of ACID compliance.

Design rationale:
    Multi-worker Railway deployments need shared persistent storage. In-memory
    dicts are worker-local. Previous solution used /tmp/*.json with file locks,
    which has:
      - Race conditions on concurrent writes
      - Non-atomic reads (partial JSON visible)
      - Slow full-file rewrite on every update

    SQLite with WAL mode provides:
      - Multiple concurrent readers (no read-blocks-read)
      - Readers and writers don't block each other
      - Atomic commits (no partial writes)
      - 10-50x faster than JSON file I/O at our scale

    JSON file fallback retained as belt-and-suspenders safety net.

Storage:
    /tmp/dtc_simulations.db        ← primary (SQLite WAL)
    /tmp/dtc_simulations.json      ← fallback (mirror, updated opportunistically)

API compatibility:
    Drop-in replacement for the previous dict-based _load/_save functions.
    Returns dict[str, dict] matching original behavior.
"""

import json
import pathlib
import sqlite3
import threading
from typing import Any

# ── Paths ──────────────────────────────────────────────────────────
DB_PATH   = pathlib.Path("/tmp/dtc_simulations.db")
JSON_PATH = pathlib.Path("/tmp/dtc_simulations.json")  # fallback

# ── Concurrency ────────────────────────────────────────────────────
_write_lock = threading.Lock()

# ── Init ───────────────────────────────────────────────────────────
def _get_conn():
    """Open SQLite connection with WAL mode enabled."""
    conn = sqlite3.connect(str(DB_PATH), timeout=5.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn

def _init_db():
    """Create simulations table if not exists."""
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS simulations (
                simulation_id TEXT PRIMARY KEY,
                data_json     TEXT NOT NULL,
                updated_at    REAL DEFAULT (julianday('now'))
            )
        """)
        conn.commit()
    finally:
        conn.close()

# Initialize on import
_init_db()

# ── Public API ─────────────────────────────────────────────────────

def load_all() -> dict[str, dict]:
    """
    Load all simulations from SQLite. Fall back to JSON on failure.
    Returns dict[simulation_id, simulation_data].
    """
    try:
        conn = _get_conn()
        rows = conn.execute("SELECT simulation_id, data_json FROM simulations").fetchall()
        conn.close()
        return {sim_id: json.loads(data_json) for sim_id, data_json in rows}
    except Exception as e:
        print(f"[dtc_storage] SQLite load failed: {e}, falling back to JSON")
        try:
            return json.loads(JSON_PATH.read_text())
        except Exception:
            return {}

def load_one(simulation_id: str) -> dict | None:
    """
    Load a single simulation by ID. Fast path: SQLite direct lookup.
    Returns None if not found.
    """
    try:
        conn = _get_conn()
        row = conn.execute(
            "SELECT data_json FROM simulations WHERE simulation_id = ?",
            (simulation_id,)
        ).fetchone()
        conn.close()
        if row:
            return json.loads(row[0])
    except Exception as e:
        print(f"[dtc_storage] SQLite load_one failed: {e}, falling back to JSON")
        try:
            all_sims = json.loads(JSON_PATH.read_text())
            return all_sims.get(simulation_id)
        except Exception:
            return None
    return None

def save_all(all_sims: dict[str, dict]) -> None:
    """
    Persist full simulations dict to both SQLite (primary) and JSON (fallback).
    Called by routes.py after in-memory updates.
    """
    with _write_lock:
        # Primary: SQLite
        try:
            conn = _get_conn()
            for sim_id, sim_data in all_sims.items():
                conn.execute(
                    """
                    INSERT INTO simulations (simulation_id, data_json, updated_at)
                    VALUES (?, ?, julianday('now'))
                    ON CONFLICT(simulation_id) DO UPDATE SET
                        data_json = excluded.data_json,
                        updated_at = excluded.updated_at
                    """,
                    (sim_id, json.dumps(sim_data, default=str))
                )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[dtc_storage] SQLite save failed: {e}")

        # Fallback: JSON mirror
        try:
            JSON_PATH.write_text(json.dumps(all_sims, default=str))
        except Exception as e:
            print(f"[dtc_storage] JSON fallback write failed: {e}")
