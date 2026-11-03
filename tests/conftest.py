from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from context_engine import AudioSegment, ContextEngine, EngineConfig, SimulatedClock, VideoObservation

T0 = datetime(2026, 3, 14, 12, 0, 0, tzinfo=timezone.utc)


def at(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


def audio(event_id: str, start: float, end: float, text: str = "hello", session: str = "s1", **kw) -> AudioSegment:
    return AudioSegment(
        event_id=event_id, session_id=session, start_time=at(start), end_time=at(end), text=text, **kw
    )


def video(event_id: str, ts: float, description: str = "a room", session: str = "s1", **kw) -> VideoObservation:
    return VideoObservation(event_id=event_id, session_id=session, timestamp=at(ts), description=description, **kw)


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock(T0)


@pytest.fixture
def engine(clock: SimulatedClock):
    eng = ContextEngine(":memory:", EngineConfig(alignment_tolerance_seconds=0.0), clock=clock)
    yield eng
    eng.close()
