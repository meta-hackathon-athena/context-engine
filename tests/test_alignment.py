from context_engine import align, render_prompt

from .conftest import at, audio, video


def pairs(derived):
    return [(a.audio_event_id, a.video_event_id) for a in derived.associations]


def test_multiple_frames_match_one_segment_without_duplicating_text(engine):
    engine.ingest_audio(audio("a1", 0, 4, "Alex is allergic to peanuts."))
    for i, ts in enumerate([0.5, 1.5, 3.9]):
        engine.ingest_video(video(f"v{i}", ts))
    snap = engine.get_context("s1", as_of=at(10))
    assert pairs(snap.derived) == [("a1", "v0"), ("a1", "v1"), ("a1", "v2")]
    assert len(snap.observed.transcript) == 1
    assert render_prompt(snap).text.count("Alex is allergic to peanuts.") == 1


def test_overlapping_audio_is_deterministic():
    a = [audio("a_spk1", 1, 5, speaker_id="spk_1"), audio("a_spk0", 0, 3, speaker_id="spk_0")]
    v = [video("v_both", 2), video("v_late", 4)]
    first = align(a, v)
    assert pairs(first) == [("a_spk0", "v_both"), ("a_spk1", "v_both"), ("a_spk1", "v_late")]
    assert align(list(reversed(a)), list(reversed(v))) == first


def test_unmatched_events_preserved():
    derived = align([audio("a_alone", 0, 1), audio("a_hit", 10, 12)], [video("v_hit", 11), video("v_alone", 20)])
    assert pairs(derived) == [("a_hit", "v_hit")]
    assert derived.unmatched_audio_event_ids == ["a_alone"]
    assert derived.unmatched_video_event_ids == ["v_alone"]


def test_interval_bounds_inclusive_and_tolerance():
    a = [audio("a", 10, 12)]
    assert pairs(align(a, [video("v_start", 10), video("v_end", 12)])) == [("a", "v_start"), ("a", "v_end")]
    near = [video("v_near", 12.3)]
    assert pairs(align(a, near, tolerance_seconds=0.0)) == []
    with_tol = align(a, near, tolerance_seconds=0.5)
    assert pairs(with_tol) == [("a", "v_near")]
    assert with_tol.associations[0].within_interval is False
    assert with_tol.associations[0].offset_seconds == 2.3
    assert with_tol.alignment_tolerance_seconds == 0.5
