"""Normalization adapters for upstream shapes that differ from the v1 contracts.

ASSUMPTION: no perception schema was available when this was written. The two
input shapes below are *hypothetical* examples of common outputs (a
Whisper-style segment list with offsets relative to a recording start, and a
captioner emitting epoch-millisecond timestamps). Adjust them once the
perception owner publishes their real format; the core engine only ever sees
validated ``AudioSegment`` / ``VideoObservation`` objects.

Example::

    >>> from datetime import datetime, timezone
    >>> seg = {"id": 3, "start": 2.0, "end": 4.5, "text": " Alex is allergic to peanuts."}
    >>> a = audio_from_relative_segment(
    ...     seg, session_id="s1", recording_id="rec-42",
    ...     recording_started_at=datetime(2026, 3, 14, 18, 30, tzinfo=timezone.utc))
    >>> a.event_id, a.start_time.isoformat(), a.text
    ('asr:rec-42:3', '2026-03-14T18:30:02+00:00', 'Alex is allergic to peanuts.')
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .schemas import AudioSegment, Detection, VideoObservation
from .timeutil import require_aware


def audio_from_relative_segment(
    segment: Mapping[str, Any],
    *,
    session_id: str,
    recording_id: str,
    recording_started_at: datetime,
    speaker_id: Optional[str] = None,
) -> AudioSegment:
    """Map ``{"id", "start", "end", "text", ["avg_logprob"|"confidence"]}`` (seconds
    relative to ``recording_started_at``) to an AudioSegment.

    The event_id is derived deterministically from recording and segment id so
    that retries deduplicate.
    """
    base = require_aware(recording_started_at, "recording_started_at")
    return AudioSegment(
        event_id=f"asr:{recording_id}:{segment['id']}",
        session_id=session_id,
        start_time=base + timedelta(seconds=float(segment["start"])),
        end_time=base + timedelta(seconds=float(segment["end"])),
        text=str(segment["text"]).strip(),
        speaker_id=speaker_id or segment.get("speaker"),
        confidence=segment.get("confidence"),
        source={"adapter": "relative_segment", "recording_id": recording_id, "segment_id": segment["id"]},
    )


def video_from_epoch_caption(
    caption: Mapping[str, Any], *, session_id: str, camera_id: str
) -> VideoObservation:
    """Map ``{"frame_id", "ts_ms", "caption", "objects": [{"name", "score"}]}``
    (epoch milliseconds, UTC) to a VideoObservation."""
    return VideoObservation(
        event_id=f"cam:{camera_id}:{caption['frame_id']}",
        session_id=session_id,
        timestamp=datetime.fromtimestamp(int(caption["ts_ms"]) / 1000, tz=timezone.utc),
        description=str(caption["caption"]),
        detections=[
            Detection(label=o["name"], confidence=float(o["score"]))
            for o in caption.get("objects", [])
        ],
        frame_ref=f"{camera_id}/{caption['frame_id']}",
        source={"adapter": "epoch_caption", "camera_id": camera_id},
    )
