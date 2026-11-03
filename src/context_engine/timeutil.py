"""Time helpers and a controllable clock. All internal times are UTC."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable, Union

from .schemas import AudioSegment, VideoObservation

Event = Union[AudioSegment, VideoObservation]
Clock = Callable[[], datetime]

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MICRO = timedelta(microseconds=1)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def require_aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def to_us(value: datetime) -> int:
    """Exact integer microseconds since the Unix epoch (no float rounding)."""
    return (value - _EPOCH) // _MICRO


def from_us(value: int) -> datetime:
    return _EPOCH + timedelta(microseconds=value)


def event_kind(event: Event) -> str:
    return "audio" if isinstance(event, AudioSegment) else "video"


def event_start(event: Event) -> datetime:
    """When the event began: audio start_time, video timestamp."""
    return event.start_time if isinstance(event, AudioSegment) else event.timestamp


def event_available(event: Event) -> datetime:
    """When the event became fully observable: audio end_time, video timestamp.

    A finalized transcript segment cannot exist before its speech ends, so the
    rolling window and future-event exclusion key on this time.
    """
    return event.end_time if isinstance(event, AudioSegment) else event.timestamp


class SimulatedClock:
    """Manually advanced clock for deterministic replay and tests."""

    def __init__(self, start: datetime) -> None:
        self._now = require_aware(start, "start")

    def __call__(self) -> datetime:
        return self._now

    def set(self, value: datetime) -> None:
        self._now = require_aware(value, "value")

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
