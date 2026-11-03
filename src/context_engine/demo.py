"""Deterministic two-day replay demo.

    python -m context_engine.demo [--db PATH] [--out-dir DIR]

Replays synthetic fixtures through the real engine with a simulated clock.
Nothing here calls a model; the final payload is what an active agent *would*
receive. This component does not recall day-one facts on day two -- that is
the job of long-term memory (dreamer / knowledge graph).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any, Optional, TextIO

from .engine import ContextEngine
from .prompt import render_prompt
from .schemas import AudioSegment, VideoObservation
from .timeutil import SimulatedClock

SESSION_ID = "demo-user-01"
DAY1_START = datetime.fromisoformat("2026-03-14T00:00:00+00:00")
DAY2_START = datetime.fromisoformat("2026-03-15T00:00:00+00:00")


def load_fixture(name: str) -> list[dict[str, Any]]:
    text = resources.files("context_engine").joinpath("fixtures", name).read_text("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _ingest(engine: ContextEngine, clock: SimulatedClock, line: dict[str, Any]) -> str:
    clock.set(datetime.fromisoformat(line["arrive_at"].replace("Z", "+00:00")))
    if line["kind"] == "audio":
        result = engine.ingest_audio(AudioSegment.model_validate(line["event"]))
    else:
        result = engine.ingest_video(VideoObservation.model_validate(line["event"]))
    return (
        f"  [clock {clock().isoformat(timespec='milliseconds')}] {line['kind']:5} "
        f"{result.event_id:6} -> {result.status}"
    )


def run_demo(db_path: Path, out_dir: Path, out: TextIO = sys.stdout) -> dict[str, Any]:
    def say(text: str = "") -> None:
        print(text, file=out)

    def step(n: int, title: str) -> None:
        say()
        say(f"=== STEP {n}: {title} ===")

    clock = SimulatedClock(DAY1_START)
    day1, day2 = load_fixture("day1.jsonl"), load_fixture("day2.jsonl")
    day1_ids = {line["event"]["event_id"] for line in day1}
    day2_ids = {line["event"]["event_id"] for line in day2}

    engine = ContextEngine(db_path, clock=clock)
    step(1, "Ingest day-one observations (fixture order = arrival order)")
    say("  d1-v3 arrives late (out of order); d1-a2 is re-sent once (retry).")
    for line in day1:
        say(_ingest(engine, clock, line))
    day1_snapshot = engine.get_context(SESSION_ID)
    say(f"  Day-one context at {clock().isoformat()}: {day1_snapshot.source_event_ids}")
    say(f"  Associations (audio <- video): "
        f"{[(a.audio_event_id, a.video_event_id) for a in day1_snapshot.derived.associations]}")
    say(f"  Unmatched video: {day1_snapshot.derived.unmatched_video_event_ids}")

    step(2, "Export day-one events for the dreamer")
    export_path = out_dir / "dreamer_day1.jsonl"
    count = engine.export_jsonl(SESSION_ID, export_path, start_time=DAY1_START, end_time=DAY2_START)
    say(f"  Wrote {count} records to {export_path}  (range [{DAY1_START.date()}, {DAY2_START.date()}))")
    first = json.loads(export_path.read_text("utf-8").splitlines()[0])
    say(f"  First record: kind={first['kind']} event_id={first['event_id']} "
        f"event_time={first['event_time']} sha256={first['payload_sha256'][:12]}...")

    step(3, "Advance the simulated clock to day two")
    clock.set(datetime.fromisoformat("2026-03-15T10:14:58+00:00"))
    say(f"  clock = {clock().isoformat()}")

    step(4, "Day-one events are outside the 5-minute working context")
    early = engine.get_context(SESSION_ID)
    say(f"  window = [{early.window_start.isoformat()}, {early.window_end.isoformat()}]")
    say(f"  events in context: {early.source_event_ids}")
    assert not day1_ids & set(early.source_event_ids)

    step(5, "Day-one events remain in persistent history (fresh engine = restart)")
    engine.close()
    engine = ContextEngine(db_path, clock=clock)
    history = engine.export_events(SESSION_ID, start_time=DAY1_START, end_time=DAY2_START)
    say(f"  Reopened {db_path.name}; day-one history: {[r.event_id for r in history]}")
    assert {r.event_id for r in history} == day1_ids

    step(6, "Day-two audio appears in context immediately after ingestion")
    say(_ingest(engine, clock, day2[0]))
    say(_ingest(engine, clock, day2[1]))
    immediate = engine.get_context(SESSION_ID)
    say(f"  context at {clock().isoformat(timespec='milliseconds')}: {immediate.source_event_ids}")
    assert "d2-a1" in immediate.source_event_ids
    for line in day2[2:]:
        say(_ingest(engine, clock, line))
    stats = engine.buffer_stats(SESSION_ID)
    say(f"  working memory: {stats['buffered_events']} buffered, {stats['evicted_events']} evicted; "
        f"SQLite: {stats['persisted_events']} persisted")

    step(7, "Exact payload the active agent would receive")
    clock.advance(0.5)
    snapshot = engine.get_context(SESSION_ID)
    prompt = render_prompt(snapshot, max_chars=4000)
    assert not day1_ids & set(snapshot.source_event_ids)
    assert set(snapshot.source_event_ids) == day2_ids
    payload = {"snapshot": snapshot.model_dump(mode="json"), "prompt": prompt.model_dump(mode="json")}
    payload_path = out_dir / "agent_payload_day2.json"
    payload_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8")
    say(f"  Written to {payload_path}")
    say("  Structured snapshot (JSON):")
    say()
    say(json.dumps(payload["snapshot"], indent=2, ensure_ascii=False))
    say()
    say("  Rendered prompt (prompt.text):")
    say()
    say(prompt.text)
    say()
    say("NOTE: The day-two context contains no day-one events, so this component alone does")
    say("NOT know that Alex is allergic to peanuts. Cross-day recall must come from long-term")
    say("memory retrieval (dreamer / graph), using the export in step 2. See")
    say("examples/mock_memory_integration.py for a clearly MOCKED illustration of that boundary.")
    engine.close()
    return {
        "export_path": export_path,
        "payload_path": payload_path,
        "exported": count,
        "snapshot": snapshot,
        "prompt": prompt,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", type=Path, help="SQLite path (default: fresh temporary file)")
    parser.add_argument("--out-dir", type=Path, default=Path("demo_output"))
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.db is not None:
        run_demo(args.db, args.out_dir)
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        run_demo(Path(tmp) / "demo.sqlite3", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
