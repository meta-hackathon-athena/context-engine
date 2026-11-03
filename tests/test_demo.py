import io
import json
from datetime import datetime

from context_engine.demo import load_fixture, run_demo


def _run(tmp_path, name):
    out = io.StringIO()
    result = run_demo(tmp_path / f"{name}.sqlite3", tmp_path / name, out)
    return result, out.getvalue()


def test_fixture_arrivals_are_monotonic():
    for name in ["day1.jsonl", "day2.jsonl"]:
        arrivals = [datetime.fromisoformat(line["arrive_at"].replace("Z", "+00:00")) for line in load_fixture(name)]
        assert arrivals == sorted(arrivals)


def test_two_day_replay_end_to_end(tmp_path):
    result, output = _run(tmp_path, "run1")
    snapshot = result["snapshot"]
    assert snapshot.source_event_ids == ["d2-v1", "d2-a1", "d2-v2", "d2-v3"]
    assert [a.text for a in snapshot.observed.transcript] == ["I'm getting peanuts for Alex."]
    assert "allergic" not in result["prompt"].text  # short-term context does not carry day one

    exported = [json.loads(line) for line in result["export_path"].read_text().splitlines()]
    assert len(exported) == 9 == result["exported"]
    assert any(r["event"].get("text") == "Alex is allergic to peanuts." for r in exported)
    assert "-> duplicate" in output and "events in context: []" in output


def test_replay_is_deterministic(tmp_path):
    first, out1 = _run(tmp_path, "a")
    second, out2 = _run(tmp_path, "b")
    assert first["snapshot"] == second["snapshot"]
    assert first["export_path"].read_text() == second["export_path"].read_text()
    assert first["payload_path"].read_text() == second["payload_path"].read_text()
    assert out1.replace(str(tmp_path / "a"), "X").replace("a.sqlite3", "D") == out2.replace(
        str(tmp_path / "b"), "X"
    ).replace("b.sqlite3", "D")
