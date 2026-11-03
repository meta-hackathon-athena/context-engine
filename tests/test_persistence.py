import io
import json

from context_engine import ContextEngine, EngineConfig, EventConflictError
from context_engine.store import canonical_payload

from .conftest import at, audio, video


def test_restart_rebuilds_context_and_keeps_dedup(tmp_path, clock):
    db = tmp_path / "events.sqlite3"
    eng = ContextEngine(db, clock=clock)
    eng.ingest_audio(audio("a1", 0, 2, speaker_id="spk_0"))
    eng.ingest_video(video("v1", 1))
    before = eng.get_context("s1", as_of=at(10))
    eng.close()

    reopened = ContextEngine(db, clock=clock)
    after = reopened.get_context("s1", as_of=at(10))
    assert after == before and after.served_from == "buffer"
    assert reopened.ingest_audio(audio("a1", 0, 2, speaker_id="spk_0")).status == "duplicate"
    try:
        reopened.ingest_audio(audio("a1", 0, 2, "different"))
        raise AssertionError("expected conflict")
    except EventConflictError:
        pass
    assert reopened.store.count() == 2


def test_restart_rehydrates_only_retention_horizon(tmp_path, clock):
    db = tmp_path / "events.sqlite3"
    config = EngineConfig(buffer_retention_seconds=60)
    eng = ContextEngine(db, config, clock=clock)
    for i in range(0, 1000, 100):
        eng.ingest_video(video(f"v{i}", i))
    eng.close()
    reopened = ContextEngine(db, config, clock=clock)
    assert reopened.buffer_stats("s1") == {
        "session_id": "s1", "buffered_events": 1, "evicted_events": 0, "persisted_events": 10,
    }


def test_export_is_chronological_half_open_and_traceable(engine, clock):
    clock.set(at(100))
    engine.ingest_video(video("v_late_arrival", 5))
    engine.ingest_audio(audio("a1", 1, 3, source={"device": "mic0"}))
    engine.ingest_video(video("v_at_end", 10))
    engine.ingest_audio(audio("a_other", 2, 3, session="other"))
    engine.ingest_audio(audio("a_tie", 5, 6))

    records = engine.export_events("s1", start_time=at(0), end_time=at(10))
    assert [r.event_id for r in records] == ["a1", "v_late_arrival", "a_tie"]  # end bound exclusive
    assert [r.event_id for r in engine.export_events("s1", start_time=at(10))] == ["v_at_end"]
    assert [r.event_id for r in engine.export_events("s1", start_time=at(1), end_time=at(1.5))] == ["a1"]

    first = records[0]
    assert first.kind == "audio" and first.event_time == at(1) and first.event_end_time == at(3)
    assert first.event["source"] == {"device": "mic0"} and first.ingested_at == at(100)
    original = audio("a1", 1, 3, source={"device": "mic0"})
    assert first.payload_sha256 == canonical_payload(original)[1]
    assert first.ingest_seq == 2

    buf = io.StringIO()
    assert engine.export_jsonl("s1", buf) == 4
    lines = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert [line["event_id"] for line in lines] == ["a1", "v_late_arrival", "a_tie", "v_at_end"]
    assert all(line["export_schema_version"] == "vixy.dreamer_event.v1" for line in lines)
    assert lines[0]["event"]["start_time"] == "2026-03-14T12:00:01Z"
