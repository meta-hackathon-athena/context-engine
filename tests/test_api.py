import json

import pytest
from fastapi.testclient import TestClient

from context_engine import ContextEngine, SimulatedClock
from context_engine.api import create_app

from .conftest import T0

AUDIO = {
    "event_id": "a1", "session_id": "s1", "start_time": "2026-03-14T12:00:00Z",
    "end_time": "2026-03-14T12:00:02Z", "text": "hello", "speaker_id": "spk_0",
}
VIDEO = {"event_id": "v1", "session_id": "s1", "timestamp": "2026-03-14T12:00:01Z", "description": "kitchen"}


@pytest.fixture
def client():
    engine = ContextEngine(":memory:", clock=SimulatedClock(T0))
    with TestClient(create_app(engine)) as c:
        yield c
    engine.close()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["schemas"]["snapshot"] == "vixy.context_snapshot.v1"


def test_ingest_and_retry_status_codes(client):
    r = client.post("/events/audio", json=AUDIO)
    assert r.status_code == 201 and r.json()["status"] == "accepted"
    r = client.post("/events/audio", json=AUDIO)
    assert r.status_code == 200 and r.json()["status"] == "duplicate"
    r = client.post("/events/audio", json={**AUDIO, "text": "changed"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "event_conflict"
    assert client.post("/events/video", json=VIDEO).status_code == 201


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/events/audio", {**AUDIO, "start_time": "2026-03-14T12:00:00"}),
        ("/events/audio", {**AUDIO, "end_time": "2026-03-14T11:00:00Z"}),
        ("/events/video", {**VIDEO, "extra": 1}),
    ],
)
def test_validation_errors_have_consistent_shape(client, path, payload):
    r = client.post(path, json=payload)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "validation_error" and err["details"]
    assert "input" not in err["details"][0]  # payload text is not echoed back


def test_context_endpoint(client):
    client.post("/events/audio", json=AUDIO)
    client.post("/events/video", json=VIDEO)
    r = client.get("/context", params={"session_id": "s1", "as_of": "2026-03-14T12:01:00Z", "max_chars": 2000})
    assert r.status_code == 200
    body = r.json()
    assert body["snapshot"]["source_event_ids"] == ["a1", "v1"]
    assert body["snapshot"]["derived"]["associations"][0]["video_event_id"] == "v1"
    assert '"hello"' in body["prompt"]["text"]

    no_prompt = client.get("/context", params={"session_id": "s1", "as_of": "2026-03-14T12:01:00Z", "include_prompt": False})
    assert no_prompt.json()["prompt"] is None


def test_context_requires_session_and_aware_as_of(client):
    assert client.get("/context").status_code == 422
    r = client.get("/context", params={"session_id": "s1", "as_of": "2026-03-14T12:01:00"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    assert client.get("/context", params={"session_id": "s1", "window_seconds": 0}).status_code == 422


def test_export_endpoint(client):
    client.post("/events/video", json=VIDEO)
    client.post("/events/audio", json=AUDIO)
    assert client.get("/events/export").status_code == 422
    r = client.get("/events/export", params={"session_id": "s1", "start_time": "2026-03-14T12:00:00Z"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in r.text.splitlines()]
    assert [line["event_id"] for line in lines] == ["a1", "v1"]
    bad = client.get(
        "/events/export",
        params={"session_id": "s1", "start_time": "2026-03-14T13:00:00Z", "end_time": "2026-03-14T12:00:00Z"},
    )
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "bad_request"
