"""Minimal FastAPI wrapper around ContextEngine.

Run:  CONTEXT_ENGINE_DB_PATH=./data/context_engine.sqlite3 \
      uvicorn context_engine.api:app --host 127.0.0.1 --port 8080

All errors use one shape: {"error": {"code": str, "message": str, "details": list}}.
"""

from __future__ import annotations

import os
from typing import Annotated, Optional

from fastapi import FastAPI, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from .engine import ContextEngine, EngineConfig, iter_jsonl_lines
from .prompt import DEFAULT_MAX_CHARS, render_prompt
from .schemas import (
    AUDIO_SCHEMA_VERSION,
    EXPORT_SCHEMA_VERSION,
    PROMPT_SCHEMA_VERSION,
    SNAPSHOT_SCHEMA_VERSION,
    VIDEO_SCHEMA_VERSION,
    AudioSegment,
    ContextSnapshot,
    IngestResult,
    RenderedPrompt,
    UtcDatetime,
    VideoObservation,
)
from .store import EventConflictError

DEFAULT_DB_PATH = "./data/context_engine.sqlite3"


class ContextResponse(BaseModel):
    snapshot: ContextSnapshot
    prompt: Optional[RenderedPrompt] = None


def _error(status: int, code: str, message: str, details: Optional[list] = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "details": jsonable_encoder(details or [])}},
    )


def create_app(engine: Optional[ContextEngine] = None) -> FastAPI:
    if engine is None:
        engine = ContextEngine(
            os.environ.get("CONTEXT_ENGINE_DB_PATH", DEFAULT_DB_PATH),
            EngineConfig(
                default_window_seconds=float(os.environ.get("CONTEXT_ENGINE_WINDOW_SECONDS", 300)),
                alignment_tolerance_seconds=float(
                    os.environ.get("CONTEXT_ENGINE_ALIGNMENT_TOLERANCE_SECONDS", 0.5)
                ),
            ),
        )
    # Interactive docs pull JS/CSS from a CDN, so they are off unless asked for.
    docs = os.environ.get("CONTEXT_ENGINE_ENABLE_DOCS") == "1"
    app = FastAPI(
        title="Vixy context-engine",
        version="0.1.0",
        docs_url="/docs" if docs else None,
        redoc_url=None,
    )
    app.state.engine = engine

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [{k: v for k, v in err.items() if k != "input"} for err in exc.errors()]
        return _error(422, "validation_error", "request failed validation", details)

    @app.exception_handler(EventConflictError)
    async def _conflict(_: Request, exc: EventConflictError) -> JSONResponse:
        return _error(409, "event_conflict", str(exc), [{"event_id": exc.event_id}])

    @app.exception_handler(ValueError)
    async def _bad_request(_: Request, exc: ValueError) -> JSONResponse:
        return _error(400, "bad_request", str(exc))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, "http_error", str(exc.detail))

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "schemas": {
                "audio": AUDIO_SCHEMA_VERSION,
                "video": VIDEO_SCHEMA_VERSION,
                "snapshot": SNAPSHOT_SCHEMA_VERSION,
                "prompt": PROMPT_SCHEMA_VERSION,
                "export": EXPORT_SCHEMA_VERSION,
            },
        }

    def _ack(result: IngestResult, response: Response) -> IngestResult:
        response.status_code = 201 if result.status == "accepted" else 200
        return result

    @app.post("/events/audio", response_model=IngestResult, status_code=201)
    def post_audio(segment: AudioSegment, response: Response) -> IngestResult:
        return _ack(engine.ingest_audio(segment), response)

    @app.post("/events/video", response_model=IngestResult, status_code=201)
    def post_video(observation: VideoObservation, response: Response) -> IngestResult:
        return _ack(engine.ingest_video(observation), response)

    @app.get("/context", response_model=ContextResponse)
    def get_context(
        session_id: Annotated[str, Query(min_length=1)],
        as_of: Optional[UtcDatetime] = None,
        window_seconds: Annotated[Optional[float], Query(gt=0, le=86400)] = None,
        include_prompt: bool = True,
        max_chars: Annotated[int, Query(ge=200, le=200_000)] = DEFAULT_MAX_CHARS,
    ) -> ContextResponse:
        snapshot = engine.get_context(session_id, as_of=as_of, window_seconds=window_seconds)
        prompt = render_prompt(snapshot, max_chars=max_chars) if include_prompt else None
        return ContextResponse(snapshot=snapshot, prompt=prompt)

    @app.get("/events/export")
    def export(
        session_id: Annotated[str, Query(min_length=1)],
        start_time: Optional[UtcDatetime] = None,
        end_time: Optional[UtcDatetime] = None,
    ) -> StreamingResponse:
        records = engine.export_events(session_id, start_time, end_time)
        return StreamingResponse(iter_jsonl_lines(records), media_type="application/x-ndjson")

    return app


def __getattr__(name: str):
    # Lazily build the module-level app so importing this module (e.g. in
    # tests) does not create a database file.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
