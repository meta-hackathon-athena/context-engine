from context_engine import ContextEngine

from .conftest import audio, video


def test_get_event_record_returns_provenance(engine):
    engine.ingest_audio(audio("a1", 0, 2, text="Alex is allergic to peanuts.", speaker_id="spk_1"))
    rec = engine.get_event_record("a1")
    assert rec.kind == "audio" and rec.event["text"] == "Alex is allergic to peanuts."
    assert rec.ingest_seq == 1 and len(rec.payload_sha256) == 64
    assert engine.get_event_record("missing") is None


def test_records_after_resumes_in_ingest_order(tmp_path, clock):
    db = tmp_path / "events.sqlite3"
    eng = ContextEngine(db, clock=clock)
    # ingest order differs from event-time order on purpose
    eng.ingest_video(video("v2", 20))
    eng.ingest_audio(audio("a1", 0, 2))
    eng.ingest_video(video("v1", 10))
    eng.ingest_video(video("other", 5, session="s2"))
    eng.close()

    reopened = ContextEngine(db, clock=clock)
    all_s1 = reopened.records_after("s1")
    assert [r.event_id for r in all_s1] == ["v2", "a1", "v1"]
    assert [r.event_id for r in reopened.records_after("s1", all_s1[0].ingest_seq)] == ["a1", "v1"]
    assert [r.event_id for r in reopened.records_after("s1", 0, limit=1)] == ["v2"]
    assert reopened.records_after("s1", all_s1[-1].ingest_seq) == []
    reopened.close()
