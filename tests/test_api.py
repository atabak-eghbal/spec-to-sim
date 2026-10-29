"""Web API smoke tests with the pipeline replaced by a fake (no model calls, no network)."""
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the app/ package lives at the repo root
from fastapi.testclient import TestClient

import app.main as m


@pytest.fixture
def client(monkeypatch):
    def fake_run_task(prompt, run_id=None, **kw):
        yield {"type": "start", "ts": time.time(), "run_id": run_id, "prompt": prompt}
        time.sleep(0.2)
        yield {"type": "done", "ts": time.time(), "success": False, "attempts": 0, "total_cost_usd": 0.0}

    monkeypatch.setattr(m, "run_task", fake_run_task)
    monkeypatch.setattr(m, "check_budget", lambda: None)
    monkeypatch.setattr(m, "_last_start", {})
    return TestClient(m.app)


def test_health_and_featured_replay(client):
    assert client.get("/healthz").json() == {"ok": True}
    ex = client.get("/api/examples").json()
    hero = next(f for f in ex["featured"] if f.get("hero"))
    body = client.get(f"/api/runs/{hero['run_id']}/events").text
    assert '"type": "start"' in body and '"type": "done"' in body


def test_one_live_run_per_visitor_per_minute(client):
    r1 = client.post("/api/runs", json={"prompt": "Stack the red cube on the blue cube"})
    assert r1.status_code == 200
    r2 = client.post("/api/runs", json={"prompt": "Stack the red cube on the blue cube"})
    assert r2.status_code == 429 and "per minute" in r2.json()["detail"]
    other = client.post("/api/runs", json={"prompt": "Stack the red cube on the blue cube"},
                        headers={"x-forwarded-for": "203.0.113.9"})
    assert other.status_code == 200


def test_queue_cap(client, monkeypatch):
    monkeypatch.setattr(m, "MAX_PENDING", 0)
    r = client.post("/api/runs", json={"prompt": "Stack the red cube on the blue cube"},
                    headers={"x-forwarded-for": "198.51.100.7"})
    assert r.status_code == 429 and "busy" in r.json()["detail"]


def test_export_and_bad_requests(client):
    ex = client.get("/api/examples").json()
    rid = ex["featured"][0]["run_id"]
    z = client.get(f"/api/runs/{rid}/export")
    assert z.status_code == 200 and z.content[:2] == b"PK"
    assert client.post("/api/runs/nope/try", json={"attempt": 1}).status_code == 404
    assert client.post("/api/runs", json={"prompt": "short"}).status_code == 422
    assert client.get("/media/../.env").status_code == 404
