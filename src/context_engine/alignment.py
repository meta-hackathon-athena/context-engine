"""Timestamp-based association of video observations with audio segments.

Rule: video observation ``v`` is associated with audio segment ``a`` iff

    a.start_time - tolerance <= v.timestamp <= a.end_time + tolerance

Both bounds are inclusive. The relation is many-to-many: one segment may match
several frames, and when audio segments overlap (e.g. two speakers talking at
once) a frame inside both is associated with both. Associations are id
references, so transcript text is never duplicated. Output order is fully
determined by (audio start, audio end, audio id, video timestamp, video id).
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from datetime import timedelta

from .schemas import AudioSegment, Association, DerivedContext, VideoObservation


def audio_sort_key(segment: AudioSegment) -> tuple:
    return (segment.start_time, segment.end_time, segment.event_id)


def video_sort_key(observation: VideoObservation) -> tuple:
    return (observation.timestamp, observation.event_id)


def align(
    audio: Sequence[AudioSegment],
    video: Sequence[VideoObservation],
    tolerance_seconds: float = 0.0,
) -> DerivedContext:
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be >= 0")
    tolerance = timedelta(seconds=tolerance_seconds)
    audio_sorted = sorted(audio, key=audio_sort_key)
    video_sorted = sorted(video, key=video_sort_key)
    video_times = [v.timestamp for v in video_sorted]

    associations: list[Association] = []
    matched_video: set[str] = set()
    unmatched_audio: list[str] = []
    for segment in audio_sorted:
        lo = bisect.bisect_left(video_times, segment.start_time - tolerance)
        hi = bisect.bisect_right(video_times, segment.end_time + tolerance)
        if lo == hi:
            unmatched_audio.append(segment.event_id)
            continue
        for observation in video_sorted[lo:hi]:
            matched_video.add(observation.event_id)
            associations.append(
                Association(
                    audio_event_id=segment.event_id,
                    video_event_id=observation.event_id,
                    offset_seconds=(observation.timestamp - segment.start_time).total_seconds(),
                    within_interval=segment.start_time
                    <= observation.timestamp
                    <= segment.end_time,
                )
            )

    return DerivedContext(
        alignment_tolerance_seconds=tolerance_seconds,
        associations=associations,
        unmatched_audio_event_ids=unmatched_audio,
        unmatched_video_event_ids=[
            v.event_id for v in video_sorted if v.event_id not in matched_video
        ],
    )
