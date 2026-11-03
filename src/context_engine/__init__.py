"""Vixy context engine: ingestion, temporal alignment, rolling context, dreamer export."""

from .alignment import align
from .engine import ContextEngine, EngineConfig, write_jsonl
from .prompt import render_prompt
from .schemas import (
    AudioSegment,
    ContextSnapshot,
    Detection,
    DreamerEventRecord,
    IngestResult,
    Location,
    RenderedPrompt,
    VideoObservation,
)
from .store import EventConflictError
from .timeutil import SimulatedClock

__all__ = [
    "AudioSegment",
    "ContextEngine",
    "ContextSnapshot",
    "Detection",
    "DreamerEventRecord",
    "EngineConfig",
    "EventConflictError",
    "IngestResult",
    "Location",
    "RenderedPrompt",
    "SimulatedClock",
    "VideoObservation",
    "align",
    "render_prompt",
    "write_jsonl",
]
