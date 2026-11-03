import threading

import pytest

from context_engine import ContextEngine, EngineConfig, EventConflictError

from .conftest import at, audio, video


def ids(snapshot):
    return snapshot.source_event_ids


def test_event_available_immediately(engine, clock):
    clock.set(at(3))
    result = engine.ingest_audio(audio("a1", 0, 2))
    assert result.status == "accepted" and result.buffered
    assert ids(engine.get_context("s1")) == ["a1"]  # no batching delay


def test_window_boundaries_are_closed(engine):
    engine.ingest_video(video("v_edge_lo", 0))
    engine.ingest_video(video("v_edge_hi", 300))
    engine.ingest_video(video("v_before", -0.001))
    snap = engine.get_context("s1", as_of=at(300), window_seconds=300)
    assert ids(snap) == ["v_edge_lo", "v_edge_hi"]
    assert snap.window_start == at(0) and snap.window_end == at(300)


def test_audio_keyed_by_end_time_and_kept_whole(engine):
    # Started before the window, ended inside it: included, boundaries intact.
    engine.ingest_audio(audio("a_straddle", -10, 5, "long utterance"))
    engine.ingest_audio(audio("a_old", -20, -1))
    snap = engine.get_context("s1", as_of=at(300), window_seconds=300)
    assert ids(snap) == ["a_straddle"]
    assert snap.observed.transcript[0].start_time == at(-10)


def test_future_events_excluded(engine):
    engine.ingest_video(video("v_past", 10))
    engine.ingest_video(video("v_future", 20))
    engine.ingest_audio(audio("a_in_progress", 14, 16))  # not finished at as_of
    snap = engine.get_context("s1", as_of=at(15))
    assert ids(snap) == ["v_past"]


def test_as_of_defaults_to_clock(engine, clock):
    engine.ingest_video(video("v1", 10))
    clock.set(at(5))
    assert ids(engine.get_context("s1")) == []
    clock.set(at(10))
    assert ids(engine.get_context("s1")) == ["v1"]


def test_out_of_order_ingestion_is_sorted(engine):
    for e in [video("v3", 3), audio("a1", 0, 1), video("v2", 2), audio("a0", -1, 0.5)]:
        (engine.ingest_audio if e.__class__.__name__ == "AudioSegment" else engine.ingest_video)(e)
    snap = engine.get_context("s1", as_of=at(10))
    assert [a.event_id for a in snap.observed.transcript] == ["a0", "a1"]
    assert [v.event_id for v in snap.observed.video] == ["v2", "v3"]
    assert ids(snap) == ["a0", "a1", "v2", "v3"]


def test_duplicate_retry_is_idempotent(engine):
    seg = audio("a1", 0, 1)
    assert engine.ingest_audio(seg).status == "accepted"
    assert engine.ingest_audio(seg.model_dump()).status == "duplicate"
    assert ids(engine.get_context("s1", as_of=at(5))) == ["a1"]
    assert engine.store.count() == 1


def test_conflicting_reuse_of_event_id_rejected(engine):
    engine.ingest_audio(audio("a1", 0, 1, "first"))
    # A revised transcript under the same id must not be appended as a new utterance.
    with pytest.raises(EventConflictError):
        engine.ingest_audio(audio("a1", 0, 1.2, "first, revised"))
    with pytest.raises(EventConflictError):
        engine.ingest_video(video("a1", 0))
    snap = engine.get_context("s1", as_of=at(5))
    assert [a.text for a in snap.observed.transcript] == ["first"]


def test_sessions_isolated(engine):
    engine.ingest_audio(audio("a1", 0, 1, session="alice"))
    engine.ingest_audio(audio("b1", 0, 1, session="bob"))
    assert ids(engine.get_context("alice", as_of=at(5))) == ["a1"]
    assert ids(engine.get_context("bob", as_of=at(5))) == ["b1"]
    assert ids(engine.get_context("nobody", as_of=at(5))) == []


def test_speakers_and_segments_not_merged(engine):
    engine.ingest_audio(audio("a1", 0, 2, "hi", speaker_id="spk_0"))
    engine.ingest_audio(audio("a2", 2, 4, "hello", speaker_id="spk_0"))
    engine.ingest_audio(audio("a3", 1, 3, "hey", speaker_id="spk_1"))
    snap = engine.get_context("s1", as_of=at(5))
    assert [(a.event_id, a.speaker_id, a.text) for a in snap.observed.transcript] == [
        ("a1", "spk_0", "hi"), ("a3", "spk_1", "hey"), ("a2", "spk_0", "hello"),
    ]


def test_buffer_is_bounded_and_eviction_keeps_history(clock):
    config = EngineConfig(buffer_retention_seconds=60, max_buffer_events_per_session=5)
    eng = ContextEngine(":memory:", config, clock=clock)
    for i in range(20):
        eng.ingest_video(video(f"v{i:02d}", i))
    stats = eng.buffer_stats("s1")
    assert stats["buffered_events"] == 5 and stats["persisted_events"] == 20
    # A window larger than what memory holds is answered from SQLite, unchanged.
    snap = eng.get_context("s1", as_of=at(19), window_seconds=300)
    assert snap.served_from == "store" and len(ids(snap)) == 20
    recent = eng.get_context("s1", as_of=at(19), window_seconds=3)
    assert recent.served_from == "buffer" and ids(recent) == ["v16", "v17", "v18", "v19"]


def test_time_based_eviction_and_late_event(clock):
    eng = ContextEngine(":memory:", EngineConfig(buffer_retention_seconds=60), clock=clock)
    eng.ingest_video(video("v_old", 0))
    eng.ingest_video(video("v_new", 1000))
    assert eng.buffer_stats("s1")["buffered_events"] == 1
    late = eng.ingest_video(video("v_late", 500))  # older than the retention horizon
    assert late.status == "accepted" and not late.buffered
    snap = eng.get_context("s1", as_of=at(600), window_seconds=300)
    assert ids(snap) == ["v_late"] and snap.served_from == "store"


def test_session_count_bounded(clock):
    eng = ContextEngine(":memory:", EngineConfig(max_buffered_sessions=2), clock=clock)
    for s in ["a", "b", "c"]:
        eng.ingest_video(video(f"v_{s}", 0, session=s))
    assert eng._buffers.loaded_sessions() == ["b", "c"]
    assert ids(eng.get_context("a", as_of=at(1))) == ["v_a"]  # rehydrated from SQLite


def test_snapshot_cap_records_truncation(engine):
    for i in range(5):
        engine.ingest_video(video(f"v{i}", i))
    snap = engine.get_context("s1", as_of=at(10), max_events=2)
    assert ids(snap) == ["v3", "v4"]
    assert snap.truncation.truncated
    assert snap.truncation.omitted_video_event_ids == ["v0", "v1", "v2"]


def test_invalid_context_arguments(engine):
    from datetime import datetime

    with pytest.raises(ValueError):
        engine.get_context("s1", as_of=datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        engine.get_context("s1", window_seconds=0)


def test_concurrent_ingestion(tmp_path, clock):
    eng = ContextEngine(tmp_path / "c.sqlite3", clock=clock)
    errors = []

    def worker(n):
        try:
            for i in range(50):
                eng.ingest_video(video(f"v{i}", i))  # every thread sends the same ids
                eng.ingest_audio(audio(f"t{n}-a{i}", i, i + 0.5))
                eng.get_context("s1", as_of=at(60))
        except Exception as exc:  # pragma: no cover - surfaced by the assert
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert eng.store.count() == 50 + 200
    assert len(ids(eng.get_context("s1", as_of=at(60), window_seconds=120))) == 250
