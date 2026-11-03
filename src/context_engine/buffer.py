"""Bounded in-memory rolling buffer (working memory) per session.

Each session buffer keeps events ordered by availability time (audio end_time,
video timestamp). It guarantees completeness only for times at or after
``complete_from_us``: every persisted event of that session with an
availability time >= ``complete_from_us`` is in the buffer. Requests that need
older data are served from SQLite by the engine instead, so eviction never
changes what a snapshot contains -- only where it is read from.

Eviction is driven by the session's high watermark (the latest availability
time ingested so far), never by wall-clock time, so replay is deterministic.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from collections.abc import Iterable
from typing import Callable, Optional

from .timeutil import Event, event_available, to_us

_MIN_US = -(2**62)


class SessionBuffer:
    def __init__(self, retention_us: int, max_events: int) -> None:
        self.retention_us = retention_us
        self.max_events = max_events
        self._keys: list[tuple[int, str]] = []  # (available_us, event_id), sorted
        self._events: dict[str, Event] = {}
        self.high_watermark_us: Optional[int] = None
        self.complete_from_us: int = _MIN_US
        self.evicted_count = 0

    def __len__(self) -> int:
        return len(self._events)

    def __contains__(self, event_id: str) -> bool:
        return event_id in self._events

    def add(self, event: Event) -> bool:
        """Insert an event; returns False if it is older than the retention horizon."""
        key = (to_us(event_available(event)), event.event_id)
        if event.event_id in self._events:
            return True
        if key[0] < self.complete_from_us:
            return False
        bisect.insort(self._keys, key)
        self._events[event.event_id] = event
        if self.high_watermark_us is None or key[0] > self.high_watermark_us:
            self.high_watermark_us = key[0]
        self._evict()
        return event.event_id in self._events

    def covers(self, from_us: int) -> bool:
        return from_us >= self.complete_from_us

    def range(self, from_us: int, to_us_inclusive: int) -> list[Event]:
        lo = bisect.bisect_left(self._keys, (from_us, ""))
        hi = bisect.bisect_right(self._keys, (to_us_inclusive, "\U0010ffff"))
        return [self._events[event_id] for _, event_id in self._keys[lo:hi]]

    def _evict(self) -> None:
        assert self.high_watermark_us is not None
        horizon = self.high_watermark_us - self.retention_us
        if horizon > self.complete_from_us:
            self.complete_from_us = horizon
        cut = bisect.bisect_left(self._keys, (self.complete_from_us, ""))
        while len(self._keys) - cut > self.max_events:
            # Count bound: drop the oldest; everything at or before its time is
            # no longer guaranteed complete.
            self.complete_from_us = self._keys[cut][0] + 1
            cut = bisect.bisect_left(self._keys, (self.complete_from_us, ""))
        for _, event_id in self._keys[:cut]:
            del self._events[event_id]
        self.evicted_count += cut
        del self._keys[:cut]


class BufferManager:
    """LRU-bounded collection of session buffers, lazily hydrated from storage."""

    def __init__(
        self,
        retention_us: int,
        max_events_per_session: int,
        max_sessions: int,
        hydrate: Callable[[str, int], tuple[Optional[int], Iterable[Event]]],
    ) -> None:
        self._retention_us = retention_us
        self._max_events = max_events_per_session
        self._max_sessions = max_sessions
        self._hydrate = hydrate
        self._sessions: "OrderedDict[str, SessionBuffer]" = OrderedDict()

    def get(self, session_id: str) -> SessionBuffer:
        buffer = self._sessions.get(session_id)
        if buffer is not None:
            self._sessions.move_to_end(session_id)
            return buffer
        buffer = SessionBuffer(self._retention_us, self._max_events)
        high_watermark, events = self._hydrate(session_id, self._retention_us)
        if high_watermark is not None:
            buffer.complete_from_us = high_watermark - self._retention_us
            for event in events:
                buffer.add(event)
        self._sessions[session_id] = buffer
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)
        return buffer

    def loaded_sessions(self) -> list[str]:
        return list(self._sessions)
