from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from context_engine import AudioSegment, Location, VideoObservation
from context_engine.adapters import audio_from_relative_segment, video_from_epoch_caption

BASE = {"event_id": "a1", "session_id": "s1", "text": "hi"}


def make_audio(**kw):
    return AudioSegment.model_validate(
        {**BASE, "start_time": "2026-03-14T12:00:00Z", "end_time": "2026-03-14T12:00:02Z", **kw}
    )


def test_naive_timestamps_rejected():
    with pytest.raises(ValidationError, match="timezone"):
        make_audio(start_time="2026-03-14T12:00:00")
    with pytest.raises(ValidationError):
        VideoObservation(event_id="v", session_id="s", timestamp=datetime(2026, 3, 14), description="x")


def test_offsets_normalized_to_utc():
    seg = make_audio(start_time="2026-03-14T20:00:00+08:00", end_time="2026-03-14T12:00:01Z")
    assert seg.start_time == datetime(2026, 3, 14, 12, 0, tzinfo=timezone.utc)
    assert seg.start_time.utcoffset().total_seconds() == 0


def test_numeric_epoch_rejected():
    with pytest.raises(ValidationError, match="numeric epoch"):
        make_audio(start_time=1773489600)


def test_invalid_interval_rejected():
    with pytest.raises(ValidationError, match="end_time must be >= start_time"):
        make_audio(start_time="2026-03-14T12:00:05Z")


def test_zero_length_interval_allowed():
    seg = make_audio(end_time="2026-03-14T12:00:00Z")
    assert seg.start_time == seg.end_time


def test_partial_transcripts_rejected():
    with pytest.raises(ValidationError, match="partial"):
        make_audio(is_final=False)


def test_unknown_fields_and_versions_rejected():
    with pytest.raises(ValidationError, match="Extra inputs"):
        make_audio(words=[])
    with pytest.raises(ValidationError):
        make_audio(schema_version="vixy.audio_segment.v2")


@pytest.mark.parametrize("field,value", [("text", "   "), ("confidence", 1.5), ("event_id", "")])
def test_field_constraints(field, value):
    with pytest.raises(ValidationError):
        make_audio(**{field: value})


def test_detection_and_location_bounds():
    with pytest.raises(ValidationError):
        VideoObservation.model_validate(
            {"event_id": "v", "session_id": "s", "timestamp": "2026-03-14T12:00:00Z",
             "description": "x", "detections": [{"label": "cup", "confidence": 2}]}
        )
    with pytest.raises(ValidationError):
        Location(latitude=91, longitude=0)
    assert make_audio().location is None  # location is never required


def test_adapters_produce_valid_events():
    base = datetime(2026, 3, 14, 18, 30, tzinfo=timezone.utc)
    seg = audio_from_relative_segment(
        {"id": 7, "start": 1.5, "end": 3.0, "text": " hi "}, session_id="s", recording_id="r", recording_started_at=base
    )
    assert seg.event_id == "asr:r:7" and seg.text == "hi"
    assert (seg.end_time - base).total_seconds() == 3.0
    obs = video_from_epoch_caption(
        {"frame_id": 12, "ts_ms": 1773513000000, "caption": "kitchen", "objects": [{"name": "cup", "score": 0.7}]},
        session_id="s", camera_id="cam0",
    )
    assert obs.timestamp.tzinfo is not None and obs.detections[0].label == "cup"
