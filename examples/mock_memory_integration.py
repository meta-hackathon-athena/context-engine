"""MOCKED integration example -- NOT a working dreamer or knowledge-graph integration.

Shows where long-term memory plugs in next to the context engine:

    active-agent input = context-engine snapshot (real)  +  memory provider results (MOCKED here)

``MockMemoryProvider`` returns one hard-coded memory. Its provenance points at
the real day-one event id that the dreamer would have consolidated, so the
team can see the shape of the handoff, but no retrieval, consolidation, or
inference happens. No model is called and no alert is produced.

Run (after the package is installed):  python examples/mock_memory_integration.py
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from context_engine import ContextEngine, ContextSnapshot, SimulatedClock, render_prompt
from context_engine.demo import SESSION_ID, load_fixture


@dataclass(frozen=True)
class MemoryRecord:
    statement: str
    source_event_ids: list[str]
    provider: str


class MemoryProvider(Protocol):
    """The interface the active agent would call. Owned by the dreamer/graph side."""

    def retrieve(self, snapshot: ContextSnapshot) -> list[MemoryRecord]: ...


class MockMemoryProvider:
    """MOCK: always returns the same fact. Stand-in for graph retrieval."""

    def retrieve(self, snapshot: ContextSnapshot) -> list[MemoryRecord]:
        return [
            MemoryRecord(
                statement="Alex is allergic to peanuts.",
                source_event_ids=["d1-a2"],
                provider="MOCK (hard-coded; not retrieved from any graph)",
            )
        ]


def main() -> None:
    clock = SimulatedClock(datetime.fromisoformat("2026-03-14T00:00:00+00:00"))
    with tempfile.TemporaryDirectory() as tmp:
        engine = ContextEngine(Path(tmp) / "mock.sqlite3", clock=clock)
        for line in load_fixture("day1.jsonl") + load_fixture("day2.jsonl"):
            clock.set(datetime.fromisoformat(line["arrive_at"].replace("Z", "+00:00")))
            ingest = engine.ingest_audio if line["kind"] == "audio" else engine.ingest_video
            ingest(line["event"])

        snapshot = engine.get_context(SESSION_ID)  # real: day-two events only
        memories = MockMemoryProvider().retrieve(snapshot)  # MOCKED

        # Provenance check against the real persisted history.
        exported = {r.event_id: r for r in engine.export_events(SESSION_ID)}
        for memory in memories:
            for event_id in memory.source_event_ids:
                record = exported[event_id]
                print(f"provenance {event_id}: {record.event['text']!r} at {record.event_time.isoformat()}")
        engine.close()

    print()
    print(render_prompt(snapshot).text)
    print()
    print("=== LONG-TERM MEMORY (MOCKED PROVIDER -- NOT REAL RETRIEVAL) ===")
    print(json.dumps([asdict(m) for m in memories], indent=2))
    print()
    print("No model was called. Deciding whether to intervene is the active agent's job.")


if __name__ == "__main__":
    main()
