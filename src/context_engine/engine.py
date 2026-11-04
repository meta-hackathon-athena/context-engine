"""ContextEngine: the library entry point (usable without FastAPI)."""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import IO, Any, Optional, Union

from .alignment import align, audio_sort_key, video_sort_key
from .buffer import BufferManager
from .schemas import (
    AudioSegment,
    ContextSnapshot,
    DreamerEventRecord,
    IngestResult,
    ObservedContext,
    SnapshotTruncation,
    VideoObservation,
)
from .store import EventStore, StoredEvent
from .timeutil import Clock, Event, event_available, event_start, require_aware, to_us, utc_now

# Only event ids and counts are logged; transcript text and descriptions never are.
logger = logging.getLogger("context_engine")


@dataclass(frozen=True)
class EngineConfig:
    default_window_seconds: float = 300.0
    # How far back (from the session's newest event) working memory retains
    # events. Windows reaching further back are served from SQLite.
    buffer_retention_seconds: float = 900.0
    max_buffer_events_per_session: int = 5000
    max_buffered_sessions: int = 64
    # See alignment.py. 0.5 s absorbs frame-sampling jitter at ~1-2 fps.
    alignment_tolerance_seconds: float = 0.5
    # Safety cap on observed records per snapshot (newest kept). None = no cap.
    max_snapshot_events: Optional[int] = 2000

    def __post_init__(self) -> None:
        if self.default_window_seconds <= 0 or self.buffer_retention_seconds <= 0:
            raise ValueError("window and retention must be > 0")
        if self.max_buffer_events_per_session < 1 or self.max_buffered_sessions < 1:
            raise ValueError("buffer bounds must be >= 1")
        if self.alignment_tolerance_seconds < 0:
            raise ValueError("alignment_tolerance_seconds must be >= 0")


class ContextEngine:
    """Ingest observed events, persist them, and build rolling context snapshots.

    Thread-safe: a single lock serializes ingestion and snapshot reads, which is
    plenty for one device's event rate.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = ":memory:",
        config: Optional[EngineConfig] = None,
        clock: Clock = utc_now,
    ) -> None:
        self.config = config or EngineConfig()
        self.clock = clock
        self.store = EventStore(db_path)
        self._lock = threading.RLock()
        self._buffers = BufferManager(
            retention_us=int(self.config.buffer_retention_seconds * 1_000_000),
            max_events_per_session=self.config.max_buffer_events_per_session,
            max_sessions=self.config.max_buffered_sessions,
            hydrate=self._hydrate,
        )

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "ContextEngine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- ingestion ---------------------------------------------------------

    def ingest_audio(self, segment: Union[AudioSegment, Mapping[str, Any]]) -> IngestResult:
        if not isinstance(segment, AudioSegment):
            segment = AudioSegment.model_validate(segment)
        return self._ingest(segment, "audio")

    def ingest_video(
        self, observation: Union[VideoObservation, Mapping[str, Any]]
    ) -> IngestResult:
        if not isinstance(observation, VideoObservation):
            observation = VideoObservation.model_validate(observation)
        return self._ingest(observation, "video")

    def _ingest(self, event: Event, kind: str) -> IngestResult:
        with self._lock:
            buffer = self._buffers.get(event.session_id)
            # Raises EventConflictError on id reuse with different content;
            # nothing is buffered in that case.
            outcome = self.store.insert(event, ingested_at=self.clock())
            buffered = buffer.add(event)
        logger.debug("ingest %s %s %s buffered=%s", kind, event.event_id, outcome, buffered)
        return IngestResult(
            status="accepted" if outcome == "inserted" else "duplicate",
            event_id=event.event_id,
            session_id=event.session_id,
            kind=kind,  # type: ignore[arg-type]
            buffered=buffered,
        )

    def _hydrate(self, session_id: str, retention_us: int) -> tuple[Optional[int], list[Event]]:
        high_watermark = self.store.max_available_us(session_id)
        if high_watermark is None:
            return None, []
        rows = self.store.query_available(session_id, high_watermark - retention_us, None)
        return high_watermark, [row.event for row in rows]

    # -- context -----------------------------------------------------------

    def get_context(
        self,
        session_id: str,
        as_of: Optional[datetime] = None,
        window_seconds: Optional[float] = None,
        max_events: Optional[int] = None,
    ) -> ContextSnapshot:
        """Snapshot of events with window_start <= available_time <= as_of.

        available_time is audio end_time / video timestamp, so audio still in
        progress at ``as_of`` and anything later are excluded. Audio segments
        are always returned whole, even if they started before window_start.
        """
        as_of = require_aware(as_of, "as_of") if as_of is not None else self.clock()
        window = self.config.default_window_seconds if window_seconds is None else window_seconds
        if window <= 0:
            raise ValueError("window_seconds must be > 0")
        cap = self.config.max_snapshot_events if max_events is None else max_events
        if cap is not None and cap < 1:
            raise ValueError("max_events must be >= 1")
        window_start = as_of - timedelta(seconds=window)
        lo, hi = to_us(window_start), to_us(as_of)

        with self._lock:
            buffer = self._buffers.get(session_id)
            if buffer.covers(lo):
                events = buffer.range(lo, hi)
                served_from = "buffer"
            else:
                events = [row.event for row in self.store.query_available(session_id, lo, hi)]
                served_from = "store"

        # Cap keeps the newest records by availability time.
        events.sort(key=lambda e: (event_available(e), e.event_id))
        omitted: list[Event] = []
        if cap is not None and len(events) > cap:
            omitted, events = events[: len(events) - cap], events[len(events) - cap :]

        audio = sorted((e for e in events if isinstance(e, AudioSegment)), key=audio_sort_key)
        video = sorted((e for e in events if isinstance(e, VideoObservation)), key=video_sort_key)
        chronological = sorted(events, key=lambda e: (event_start(e), event_available(e), e.event_id))
        return ContextSnapshot(
            session_id=session_id,
            as_of=as_of,
            window_start=window_start,
            window_end=as_of,
            window_seconds=window,
            observed=ObservedContext(transcript=audio, video=video),
            derived=align(audio, video, self.config.alignment_tolerance_seconds),
            source_event_ids=[e.event_id for e in chronological],
            truncation=SnapshotTruncation(
                truncated=bool(omitted),
                max_events=cap,
                omitted_audio_event_ids=[e.event_id for e in omitted if isinstance(e, AudioSegment)],
                omitted_video_event_ids=[
                    e.event_id for e in omitted if isinstance(e, VideoObservation)
                ],
            ),
            served_from=served_from,  # type: ignore[arg-type]
        )

    def buffer_stats(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            buffer = self._buffers.get(session_id)
            return {
                "session_id": session_id,
                "buffered_events": len(buffer),
                "evicted_events": buffer.evicted_count,
                "persisted_events": self.store.count(session_id),
            }

    # -- dreamer export ----------------------------------------------------

    def export_events(
        self,
        session_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[DreamerEventRecord]:
        """Persisted events with start_time <= event_time < end_time, chronological.

        event_time is audio start_time / video timestamp. Either bound may be
        omitted. The range is half-open so consecutive exports never overlap.
        """
        start_us = to_us(require_aware(start_time, "start_time")) if start_time else None
        end_us = to_us(require_aware(end_time, "end_time")) if end_time else None
        if start_us is not None and end_us is not None and end_us < start_us:
            raise ValueError("end_time must be >= start_time")
        return [_to_record(row) for row in self.store.query_range(session_id, start_us, end_us)]

    def get_event_record(self, event_id: str) -> Optional[DreamerEventRecord]:
        """One persisted event by id (provenance lookup), or None."""
        row = self.store.get(event_id)
        return _to_record(row) if row is not None else None

    def records_after(
        self, session_id: str, after_seq: int = 0, limit: Optional[int] = None
    ) -> list[DreamerEventRecord]:
        """Persisted events with ingest_seq > after_seq, in ingestion order.

        For downstream consumers that must replay or recover side effects after
        a restart (resume from the last ingest_seq they fully processed).
        """
        return [_to_record(row) for row in self.store.query_after_seq(session_id, after_seq, limit)]

    def export_jsonl(
        self,
        session_id: str,
        destination: Union[str, Path, IO[str]],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> int:
        records = self.export_events(session_id, start_time, end_time)
        write_jsonl(records, destination)
        return len(records)


def _to_record(row: StoredEvent) -> DreamerEventRecord:
    event = row.event
    return DreamerEventRecord(
        kind=row.kind,
        event_id=event.event_id,
        session_id=event.session_id,
        event_time=event_start(event),
        event_end_time=event.end_time if isinstance(event, AudioSegment) else None,
        ingest_seq=row.ingest_seq,
        ingested_at=row.ingested_at,
        payload_sha256=row.payload_sha256,
        event=event.model_dump(mode="json"),
    )


def iter_jsonl_lines(records: Iterable[DreamerEventRecord]) -> Iterable[str]:
    for record in records:
        yield json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"


def write_jsonl(
    records: Iterable[DreamerEventRecord], destination: Union[str, Path, IO[str]]
) -> None:
    if isinstance(destination, (str, Path)):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        with open(destination, "w", encoding="utf-8") as fp:
            fp.writelines(iter_jsonl_lines(records))
    else:
        destination.writelines(iter_jsonl_lines(records))
