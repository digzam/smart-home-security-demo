"""
database.py — SQLite layer for Smart Home Security Demo
=========================================================
Handles:  init, create visit, read visits, delete all (records + images)
"""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime

DB_PATH   = "security.db"
log       = logging.getLogger(__name__)


# ── Internal helper ──────────────────────────────────────────────────────────
def _conn():
    """Open a SQLite connection with row-factory set."""
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


# ── Public API ───────────────────────────────────────────────────────────────

def init_db():
    """Create the visits table (and indexes) if they don't exist yet."""
    with _conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS visits (
                visit_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         DATETIME NOT NULL,
                image_path        TEXT     NOT NULL DEFAULT '',
                human_detected    INTEGER  NOT NULL DEFAULT 1,
                human_count       INTEGER  NOT NULL DEFAULT 1,
                processing_status TEXT     NOT NULL DEFAULT 'pending',

                -- Reserved for future face-recognition module (unused now)
                face_detected     INTEGER,
                recognized_person TEXT,
                face_confidence   REAL
            )
        """)
        # Useful indexes for dashboard queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_visits_timestamp "
            "ON visits(timestamp DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_visits_status "
            "ON visits(processing_status)"
        )
        conn.commit()
    log.info(f"Database ready → {DB_PATH}")


def create_visit(timestamp: datetime, image_path: str, human_count: int) -> int:
    """
    Insert one visit record and return the new visit_id.

    Args:
        timestamp:   datetime of the visit event
        image_path:  path to the saved JPEG (may be '' initially)
        human_count: number of people detected in the frame
    """
    with _conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO visits (timestamp, image_path, human_detected, human_count, processing_status)
            VALUES (?, ?, 1, ?, 'pending')
            """,
            (timestamp.isoformat(sep=" ", timespec="seconds"), image_path, human_count),
        )
        conn.commit()
        return cur.lastrowid


def update_image_path(visit_id: int, image_path: str):
    """Update image_path after the JPEG has been written to disk."""
    with _conn() as conn:
        conn.execute(
            "UPDATE visits SET image_path = ? WHERE visit_id = ?",
            (image_path, visit_id),
        )
        conn.commit()


def get_visits() -> list[dict]:
    """Return all visits, newest first."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM visits ORDER BY visit_id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_visit(visit_id: int) -> dict | None:
    """Return a single visit by ID, or None if not found."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT * FROM visits WHERE visit_id = ?", (visit_id,)
        ).fetchone()
    return dict(row) if row else None


def delete_all_visits() -> int:
    """
    Delete every visit record AND the matching image file from disk.

    Returns:
        Number of records deleted.
    """
    with _conn() as conn:
        rows  = conn.execute("SELECT image_path FROM visits").fetchall()
        count = len(rows)

        for row in rows:
            img = Path(row["image_path"])
            if img.exists():
                img.unlink()
                log.info(f"  Deleted image → {img}")

        conn.execute("DELETE FROM visits")
        conn.commit()

    log.info(f"Cleared {count} visit record(s) from {DB_PATH}.")
    return count
