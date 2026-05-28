"""Tiny SQLite store for finalized plate detections.

One row per `track_id` (the same plate across many frames collapses to one
record).  Use `Store.upsert(...)` from the runner whenever a track's voted
text is updated; the schema's PRIMARY KEY on (source, track_id) handles dedup.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS plates (
    source       TEXT    NOT NULL,
    track_id     INTEGER NOT NULL,
    plate_text   TEXT    NOT NULL,
    raw_text     TEXT,
    first_seen   REAL    NOT NULL,
    last_seen    REAL    NOT NULL,
    n_frames     INTEGER NOT NULL DEFAULT 1,
    det_conf     REAL,
    ocr_conf     REAL,
    valid        INTEGER NOT NULL DEFAULT 0,
    crop_path    TEXT,
    PRIMARY KEY (source, track_id)
);
CREATE INDEX IF NOT EXISTS plates_text_idx ON plates(plate_text);
CREATE INDEX IF NOT EXISTS plates_seen_idx ON plates(last_seen);
"""


@dataclass
class Row:
    source: str
    track_id: int
    plate_text: str
    raw_text: str = ""
    det_conf: float = 0.0
    ocr_conf: float = 0.0
    valid: bool = False
    crop_path: str = ""


class Store:
    def __init__(self, db_path: str | Path):
        self.path = str(db_path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None)  # autocommit
        self.conn.executescript(SCHEMA)

    def upsert(self, r: Row) -> None:
        now = time.time()
        cur = self.conn.execute(
            "SELECT n_frames FROM plates WHERE source=? AND track_id=?",
            (r.source, r.track_id),
        )
        existing = cur.fetchone()
        if existing is None:
            self.conn.execute(
                "INSERT INTO plates (source, track_id, plate_text, raw_text, "
                "first_seen, last_seen, n_frames, det_conf, ocr_conf, valid, crop_path) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (r.source, r.track_id, r.plate_text, r.raw_text,
                 now, now, 1, r.det_conf, r.ocr_conf, int(r.valid), r.crop_path),
            )
        else:
            self.conn.execute(
                "UPDATE plates SET plate_text=?, raw_text=?, last_seen=?, "
                "n_frames=n_frames+1, det_conf=MAX(det_conf,?), ocr_conf=MAX(ocr_conf,?), "
                "valid=MAX(valid,?), crop_path=COALESCE(NULLIF(?, ''), crop_path) "
                "WHERE source=? AND track_id=?",
                (r.plate_text, r.raw_text, now, r.det_conf, r.ocr_conf,
                 int(r.valid), r.crop_path, r.source, r.track_id),
            )

    def recent(self, limit: int = 20):
        return self.conn.execute(
            "SELECT plate_text, source, n_frames, det_conf, ocr_conf, valid, "
            "datetime(first_seen,'unixepoch','localtime') AS first, "
            "datetime(last_seen, 'unixepoch','localtime') AS last "
            "FROM plates ORDER BY last_seen DESC LIMIT ?",
            (limit,),
        ).fetchall()
