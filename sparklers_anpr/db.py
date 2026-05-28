"""Sparkler ANPR Studio — SQLite store.

Single file: ~/sparklers_anpr/runs/sparkler.db.  One row per `(session, track_id)`
detection.  Crops live as JPEG files under runs/crops/ and are referenced
by relative path.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,           -- uuid hex
    started_at  REAL NOT NULL,
    ended_at    REAL,
    source_type TEXT NOT NULL,              -- csi | usb | image | video | rtsp
    source_uri  TEXT,                       -- /dev/video0, file path, rtsp:// …
    backend     TEXT,                       -- fast | easyocr | fast+vlm | …
    detector    TEXT,                       -- engine path
    n_plates    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS plates (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id),
    track_id     INTEGER,                   -- ByteTrack id within the session
    plate_text   TEXT NOT NULL,             -- OCR (fast / easyocr) cleaned text
    vlm_text     TEXT NOT NULL DEFAULT '',  -- VLM cleaned text (empty if no VLM ran)
    raw_text     TEXT,
    first_seen   REAL NOT NULL,
    last_seen    REAL NOT NULL,
    n_frames     INTEGER NOT NULL DEFAULT 1,
    det_conf     REAL,
    ocr_conf     REAL,
    valid        INTEGER NOT NULL DEFAULT 0,   -- LEGACY (kept for old rows; new rows compute `verified` on the fly)
    vlm_verified INTEGER NOT NULL DEFAULT 0,   -- LEGACY
    crop_path    TEXT,
    frame_path   TEXT
);

CREATE INDEX IF NOT EXISTS idx_plates_text     ON plates(plate_text);
CREATE INDEX IF NOT EXISTS idx_plates_lastseen ON plates(last_seen DESC);
CREATE INDEX IF NOT EXISTS idx_plates_session  ON plates(session_id);
"""


@dataclass
class Session:
    id: str
    started_at: float
    source_type: str
    source_uri: str = ""
    backend: str = "fast"
    detector: str = ""


@dataclass
class Plate:
    session_id: str
    track_id: Optional[int]
    plate_text: str                # OCR cleaned text
    vlm_text: str = ""             # VLM cleaned text (empty until VLM ran)
    raw_text: str = ""
    det_conf: float = 0.0
    ocr_conf: float = 0.0
    crop_path: str = ""
    frame_path: str = ""

    @property
    def verified(self) -> bool:
        """A plate is 'verified' iff OCR and VLM agree on the same non-empty text."""
        return bool(self.vlm_text) and self.plate_text == self.vlm_text


class Store:
    def __init__(self, db_path: str | Path):
        self.path = str(db_path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, isolation_level=None,
                                     check_same_thread=False)
        self._conn.executescript(SCHEMA)
        # safe migration for older DBs that pre-date the vlm_text column
        try:
            self._conn.execute("ALTER TABLE plates ADD COLUMN vlm_text TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass
        self._conn.row_factory = sqlite3.Row

    # ----- sessions ----------------------------------------------------------

    def new_session(self, source_type: str, source_uri: str = "",
                    backend: str = "fast", detector: str = "") -> Session:
        sid = uuid.uuid4().hex[:12]
        now = time.time()
        self._conn.execute(
            "INSERT INTO sessions (id, started_at, source_type, source_uri, "
            "backend, detector) VALUES (?,?,?,?,?,?)",
            (sid, now, source_type, source_uri, backend, detector),
        )
        return Session(id=sid, started_at=now, source_type=source_type,
                       source_uri=source_uri, backend=backend, detector=detector)

    def end_session(self, sid: str) -> None:
        self._conn.execute("UPDATE sessions SET ended_at=? WHERE id=?",
                           (time.time(), sid))

    def list_sessions(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT s.*, "
            "(SELECT COUNT(*) FROM plates WHERE session_id=s.id) AS plate_count "
            "FROM sessions s ORDER BY s.started_at DESC LIMIT ?", (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ----- plates ------------------------------------------------------------

    def upsert_plate(self, p: Plate) -> int:
        """Upsert by (session_id, track_id).  Returns plate row id."""
        now = time.time()
        verified_flag = int(p.verified)   # convenience for legacy `valid` column readers
        row = self._conn.execute(
            "SELECT id, n_frames FROM plates WHERE session_id=? AND track_id=?",
            (p.session_id, p.track_id),
        ).fetchone()
        if row is None:
            cur = self._conn.execute(
                "INSERT INTO plates (session_id, track_id, plate_text, vlm_text, raw_text, "
                "first_seen, last_seen, n_frames, det_conf, ocr_conf, valid, vlm_verified, "
                "crop_path, frame_path) "
                "VALUES (?,?,?,?,?,?,?,1,?,?,?,?,?,?)",
                (p.session_id, p.track_id, p.plate_text, p.vlm_text, p.raw_text,
                 now, now, p.det_conf, p.ocr_conf,
                 verified_flag, int(bool(p.vlm_text)),
                 p.crop_path, p.frame_path),
            )
            self._conn.execute(
                "UPDATE sessions SET n_plates=n_plates+1 WHERE id=?",
                (p.session_id,),
            )
            return cur.lastrowid
        # update existing track row.  Both plate_text and vlm_text only
        # overwrite when the new value is non-empty (so the off-frame VLM
        # sync can update vlm_text without erasing the OCR text and vice versa).
        self._conn.execute(
            "UPDATE plates SET "
            "plate_text=COALESCE(NULLIF(?, ''), plate_text), "
            "vlm_text=COALESCE(NULLIF(?, ''), vlm_text), "
            "raw_text=COALESCE(NULLIF(?, ''), raw_text), "
            "last_seen=?, n_frames=n_frames+1, "
            "det_conf=MAX(det_conf, ?), ocr_conf=MAX(ocr_conf, ?), "
            "valid=MAX(valid, ?), vlm_verified=MAX(vlm_verified, ?), "
            "crop_path=COALESCE(NULLIF(?, ''), crop_path), "
            "frame_path=COALESCE(NULLIF(?, ''), frame_path) "
            "WHERE id=?",
            (p.plate_text, p.vlm_text, p.raw_text, now,
             p.det_conf, p.ocr_conf,
             verified_flag, int(bool(p.vlm_text)),
             p.crop_path, p.frame_path,
             row["id"]),
        )
        return row["id"]

    def recent(self, limit: int = 50, search: str = "",
               verified_only: bool = False) -> list[dict]:
        sql = ("SELECT *, (vlm_text != '' AND vlm_text = plate_text) AS verified "
               "FROM plates WHERE 1=1 ")
        params: list = []
        if search:
            sql += "AND (plate_text LIKE ? OR vlm_text LIKE ?) "
            params.extend([f"%{search.upper()}%", f"%{search.upper()}%"])
        if verified_only:
            sql += "AND vlm_text != '' AND vlm_text = plate_text "
        sql += "ORDER BY last_seen DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def by_text(self, plate_text: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT *, (vlm_text != '' AND vlm_text = plate_text) AS verified "
            "FROM plates WHERE plate_text=? OR vlm_text=? "
            "ORDER BY last_seen DESC",
            (plate_text.upper(), plate_text.upper()),
        ).fetchall()
        return [dict(r) for r in rows]

    def distinct_plates(self, limit: int = 100) -> list[dict]:
        rows = self._conn.execute(
            "SELECT plate_text, COUNT(*) AS sightings, "
            "       MAX(last_seen) AS last_seen, "
            "       MAX(vlm_text != '' AND vlm_text = plate_text) AS verified "
            "FROM plates WHERE plate_text != '' "
            "GROUP BY plate_text ORDER BY last_seen DESC LIMIT ?", (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ----- deletion ----------------------------------------------------------

    def delete_all(self) -> int:
        """Wipe all plates and sessions.  Returns count deleted."""
        n = self._conn.execute("SELECT COUNT(*) AS c FROM plates").fetchone()["c"]
        self._conn.execute("DELETE FROM plates")
        self._conn.execute("DELETE FROM sessions")
        return n

    def delete_by_id(self, plate_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM plates WHERE id=?", (plate_id,))
        return cur.rowcount > 0

    def delete_by_text(self, plate_text: str) -> int:
        """Delete every sighting of one plate.  Returns count deleted."""
        cur = self._conn.execute("DELETE FROM plates WHERE plate_text=?",
                                  (plate_text.upper(),))
        return cur.rowcount

    def get(self, plate_id: int) -> Optional[dict]:
        row = self._conn.execute("SELECT * FROM plates WHERE id=?",
                                  (plate_id,)).fetchone()
        return dict(row) if row else None

    def stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) AS c FROM plates").fetchone()["c"]
        unique = self._conn.execute("SELECT COUNT(DISTINCT plate_text) AS c FROM plates").fetchone()["c"]
        sessions = self._conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]
        last_hour = self._conn.execute(
            "SELECT COUNT(*) AS c FROM plates WHERE last_seen > ?",
            (time.time() - 3600,),
        ).fetchone()["c"]
        return {
            "total_detections": total,
            "unique_plates": unique,
            "sessions": sessions,
            "last_hour": last_hour,
        }
