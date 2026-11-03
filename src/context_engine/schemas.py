"""Versioned data contracts for the context engine.

Two families of models live here:

* Observed input events (``AudioSegment``, ``VideoObservation``) as produced by
  perception. These are stored and returned unchanged.
* Derived output (``ContextSnapshot``, ``Association``, ``DreamerEventRecord``)
  computed by this component. Derived data always references source events by
  ``event_id`` and never rewrites them.

Timestamp policy: every timestamp must be an ISO-8601 string (or ``datetime``)
with an explicit UTC offset. Naive timestamps and bare epoch numbers are
rejected. Offsets other than UTC are accepted and converted to UTC.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

AUDIO_SCHEMA_VERSION = "vixy.audio_segment.v1"
VIDEO_SCHEMA_VERSION = "vixy.video_observation.v1"
SNAPSHOT_SCHEMA_VERSION = "vixy.context_snapshot.v1"
PROMPT_SCHEMA_VERSION = "vixy.context_prompt.v1"
EXPORT_SCHEMA_VERSION = "vixy.dreamer_event.v1"


def _reject_numeric(value: Any) -> Any:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        raise ValueError(
            "numeric epoch timestamps are not accepted; send ISO-8601 with an "
            "explicit offset (e.g. 2026-03-14T18:30:02Z) or use an adapter"
        )
    return value


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


UtcDatetime = Annotated[
    AwareDatetime, BeforeValidator(_reject_numeric), AfterValidator(_to_utc)
]
EventId = Annotated[str, Field(min_length=1, max_length=256)]
SessionId = Annotated[str, Field(min_length=1, max_length=256)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
SourceMetadata = Optional[dict[str, JsonValue]]


class _Model(BaseModel):
    # extra="forbid" makes upstream schema drift fail loudly instead of being
    # silently dropped. Use adapters.py to map foreign shapes.
    model_config = ConfigDict(extra="forbid", frozen=True)


def _require_text(value: str) -> str:
    if not value.strip():
        raise ValueError("must contain non-whitespace text")
    return value


class Location(_Model):
    """Optional device location. Never required; never fabricated."""

    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    accuracy_m: Optional[float] = Field(default=None, ge=0.0)


class Detection(_Model):
    label: str = Field(min_length=1)
    confidence: Confidence


class AudioSegment(_Model):
    """A finalized transcript segment. Observed data."""

    schema_version: Literal["vixy.audio_segment.v1"] = AUDIO_SCHEMA_VERSION
    event_id: EventId
    session_id: SessionId
    start_time: UtcDatetime
    end_time: UtcDatetime
    text: str
    speaker_id: Optional[str] = None
    confidence: Optional[Confidence] = None
    is_final: bool = True
    location: Optional[Location] = None
    source: SourceMetadata = None

    _check_text = field_validator("text")(_require_text)

    @field_validator("is_final")
    @classmethod
    def _only_final(cls, value: bool) -> bool:
        if not value:
            raise ValueError(
                "partial (non-final) transcript segments are not supported in v1; "
                "send only finalized segments"
            )
        return value

    @model_validator(mode="after")
    def _check_interval(self) -> "AudioSegment":
        if self.end_time < self.start_time:
            raise ValueError("end_time must be >= start_time")
        return self


class VideoObservation(_Model):
    """A processed visual observation (scene description). Observed data."""

    schema_version: Literal["vixy.video_observation.v1"] = VIDEO_SCHEMA_VERSION
    event_id: EventId
    session_id: SessionId
    timestamp: UtcDatetime
    description: str
    detections: list[Detection] = Field(default_factory=list)
    frame_ref: Optional[str] = None
    location: Optional[Location] = None
    source: SourceMetadata = None

    _check_description = field_validator("description")(_require_text)


EventKind = Literal["audio", "video"]


class IngestResult(_Model):
    status: Literal["accepted", "duplicate"]
    event_id: str
    session_id: str
    kind: EventKind
    # False when the event is older than the in-memory retention horizon. It is
    # still persisted and served via the SQLite fallback path.
    buffered: bool


# ---------------------------------------------------------------------------
# Derived output
# ---------------------------------------------------------------------------


class Association(_Model):
    """Derived: a video observation that falls within an audio segment's interval."""

    audio_event_id: str
    video_event_id: str
    # video.timestamp - audio.start_time, in seconds.
    offset_seconds: float
    # True if the frame lies inside [start_time, end_time] without tolerance.
    within_interval: bool


class ObservedContext(_Model):
    transcript: list[AudioSegment]
    video: list[VideoObservation]


class DerivedContext(_Model):
    alignment_tolerance_seconds: float
    associations: list[Association]
    unmatched_audio_event_ids: list[str]
    unmatched_video_event_ids: list[str]


class SnapshotTruncation(_Model):
    truncated: bool
    max_events: Optional[int]
    omitted_audio_event_ids: list[str]
    omitted_video_event_ids: list[str]


class ContextSnapshot(_Model):
    schema_version: Literal["vixy.context_snapshot.v1"] = SNAPSHOT_SCHEMA_VERSION
    session_id: str
    as_of: datetime
    window_start: datetime
    window_end: datetime
    window_seconds: float
    observed: ObservedContext
    derived: DerivedContext
    # All included event ids in chronological order (audio by start_time,
    # video by timestamp).
    source_event_ids: list[str]
    truncation: SnapshotTruncation
    # "buffer" (in-memory) or "store" (SQLite fallback); purely diagnostic.
    served_from: Literal["buffer", "store"]


class RenderedPrompt(_Model):
    schema_version: Literal["vixy.context_prompt.v1"] = PROMPT_SCHEMA_VERSION
    text: str
    max_chars: int
    char_count: int
    truncated: bool
    included_event_ids: list[str]
    omitted_event_ids: list[str]


class DreamerEventRecord(_Model):
    """One line of the dreamer JSONL export."""

    export_schema_version: Literal["vixy.dreamer_event.v1"] = EXPORT_SCHEMA_VERSION
    kind: EventKind
    event_id: str
    session_id: str
    # audio: start_time; video: timestamp. Export ranges and ordering use this.
    event_time: datetime
    # audio: end_time; video: null.
    event_end_time: Optional[datetime]
    ingest_seq: int
    ingested_at: datetime
    payload_sha256: str
    event: dict[str, Any]
