"""Durable SQLite event history.

Every accepted event is committed here before ingestion is acknowledged. The
``event_id`` column is UNIQUE, so deduplication holds across restarts. Times
are stored as integer UTC microseconds for exact range queries.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union

from .schemas import AudioSegment, VideoObservation
from .timeutil import Event, event_available, event_kind, event_start, from_us, to_us

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    ingest_seq     INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id       TEXT    NOT NULL UNIQUE,
    session_id     TEXT    NOT NULL,
    kind           TEXT    NOT NULL CHECK (kind IN ('audio', 'video')),
    start_us       INTEGER NOT NULL,
    available_us   INTEGER NOT NULL,
    ingested_at_us INTEGER NOT NULL,
    schema_version TEXT    NOT NULL,
    payload_sha256 TEXT    NOT NULL,
    payload_json   TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_session_start
    ON events (session_id, start_us, available_us, event_id);
CREATE INDEX IF NOT EXISTS idx_events_session_available
    ON events (session_id, available_us, event_id);
"""

_COLUMNS = "ingest_seq, event_id, session_id, kind, ingested_at_us, payload_sha256, payload_json"


class EventConflictError(Exception):
    """An event_id was re-sent with a different payload (or kind)."""

    def __init__(self, event_id: str) -> None:
        super().__init__(
            f"event_id {event_id!r} already exists with a different payload; "
            "event ids must be unique and retries must resend identical data"
        )
        self.event_id = event_id


@dataclass(frozen=True)
class StoredEvent:
    ingest_seq: int
    kind: Literal["audio", "video"]
    event: Event
    ingested_at: datetime
    payload_sha256: str


def canonical_payload(event: Event) -> tuple[str, str]:
    """Return (canonical JSON, sha256) for an event. Used for dedup and provenance."""
    payload = json.dumps(
        event.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


class EventStore:
    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def insert(self, event: Event, ingested_at: datetime) -> Literal["inserted", "duplicate"]:
        """Commit an event. Returns "duplicate" for an identical retry.

        Raises EventConflictError if the event_id exists with different content.
        """
        payload, digest = canonical_payload(event)
        row = (
            event.event_id,
            event.session_id,
            event_kind(event),
            to_us(event_start(event)),
            to_us(event_available(event)),
            to_us(ingested_at),
            event.schema_version,
            digest,
            payload,
        )
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO events (event_id, session_id, kind, start_us, available_us, "
                    " ingested_at_us, schema_version, payload_sha256, payload_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    row,
                )
                return "inserted"
            except sqlite3.IntegrityError:
                existing = self._conn.execute(
                    "SELECT payload_sha256 FROM events WHERE event_id = ?", (event.event_id,)
                ).fetchone()
        if existing is not None and existing[0] == digest:
            return "duplicate"
        raise EventConflictError(event.event_id)

    def max_available_us(self, session_id: str) -> Optional[int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(available_us) FROM events WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row[0]

    def query_available(
        self, session_id: str, from_us: Optional[int], to_us_inclusive: Optional[int]
    ) -> list[StoredEvent]:
        """Events with from_us <= available_us <= to_us_inclusive (bounds optional)."""
        sql = f"SELECT {_COLUMNS} FROM events WHERE session_id = ?"
        params: list[object] = [session_id]
        if from_us is not None:
            sql += " AND available_us >= ?"
            params.append(from_us)
        if to_us_inclusive is not None:
            sql += " AND available_us <= ?"
            params.append(to_us_inclusive)
        return self._fetch(sql + " ORDER BY available_us, event_id", params)

    def query_range(
        self, session_id: str, start_us: Optional[int], end_us: Optional[int]
    ) -> list[StoredEvent]:
        """Chronological events with start_us <= event start < end_us (half-open)."""
        sql = f"SELECT {_COLUMNS} FROM events WHERE session_id = ?"
        params: list[object] = [session_id]
        if start_us is not None:
            sql += " AND start_us >= ?"
            params.append(start_us)
        if end_us is not None:
            sql += " AND start_us < ?"
            params.append(end_us)
        return self._fetch(sql + " ORDER BY start_us, available_us, kind, event_id", params)

    def get(self, event_id: str) -> Optional[StoredEvent]:
        rows = self._fetch(f"SELECT {_COLUMNS} FROM events WHERE event_id = ?", [event_id])
        return rows[0] if rows else None

    def query_after_seq(self, session_id: str, after_seq: int, limit: Optional[int] = None) -> list[StoredEvent]:
        """Events with ingest_seq > after_seq, in ingestion order (consumer replay)."""
        sql = f"SELECT {_COLUMNS} FROM events WHERE session_id = ? AND ingest_seq > ? ORDER BY ingest_seq"
        params: list[object] = [session_id, after_seq]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return self._fetch(sql, params)

    def count(self, session_id: Optional[str] = None) -> int:
        with self._lock:
            if session_id is None:
                return self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            return self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE session_id = ?", (session_id,)
            ).fetchone()[0]

    def _fetch(self, sql: str, params: list[object]) -> list[StoredEvent]:
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out = []
        for seq, _event_id, _session, kind, ingested_us, digest, payload in rows:
            model = AudioSegment if kind == "audio" else VideoObservation
            out.append(
                StoredEvent(
                    ingest_seq=seq,
                    kind=kind,
                    event=model.model_validate_json(payload),
                    ingested_at=from_us(ingested_us),
                    payload_sha256=digest,
                )
            )
        return out
