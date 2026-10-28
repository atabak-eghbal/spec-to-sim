"""Web app: FastAPI backend that runs the pipeline and streams each step to the UI over SSE.

- POST /api/runs            start a run in a worker thread, returns {run_id}
- GET  /api/runs/{id}/events Server-Sent Events: every pipeline event, replayed from the start
- GET  /api/runs/{id}       all events so far (JSON), for replaying saved runs
- GET  /api/examples        curated prompts + featured (pre-recorded) runs
- GET  /media/...           videos and frames under runs/
Every run's events are persisted to runs/<id>/events.jsonl, so finished runs replay instantly
for free - the featured runs on the landing page work even if an upstream API is slow.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from spec2sim.loop import RUNS, run_task

STATIC = Path(__file__).parent / "static"
EXAMPLES_FILE = Path(__file__).parent / "examples.json"
MAX_CONCURRENT = 2

FEATURED = Path(__file__).parent / "featured"

app = FastAPI(title="Spec-to-Sim")
RUNS.mkdir(exist_ok=True)
# Featured runs are committed under app/featured/ and copied into runs/ so they replay anywhere.
for _src in FEATURED.glob("*/"):
    if not (RUNS / _src.name).exists():
        shutil.copytree(_src, RUNS / _src.name)
app.mount("/media", StaticFiles(directory=RUNS), name="media")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

_slots = threading.Semaphore(MAX_CONCURRENT)


class _Live:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.done = False


_live: dict[str, _Live] = {}


class RunRequest(BaseModel):
    prompt: str = Field(min_length=8, max_length=300)


def _worker(run_id: str, prompt: str, live: _Live) -> None:
    out = RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "events.jsonl").open("a")

    def emit(ev: dict) -> None:
        live.events.append(ev)
        log.write(json.dumps(ev) + "\n")
        log.flush()

    if not _slots.acquire(blocking=False):
        emit({"type": "queued", "ts": time.time(), "message": "Another simulation is running; waiting for a free slot."})
        _slots.acquire()
    try:
        for ev in run_task(prompt, run_id=run_id):
            emit(ev)
    except Exception as e:  # noqa: BLE001 - any failure must reach the UI as an event
        traceback.print_exc()
        emit({"type": "error", "ts": time.time(), "message": f"{type(e).__name__}: {e}"})
    finally:
        _slots.release()
        live.done = True
        log.close()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/examples")
def examples() -> dict:
    return json.loads(EXAMPLES_FILE.read_text()) if EXAMPLES_FILE.exists() else {"prompts": [], "featured": []}


@app.post("/api/runs")
def start_run(req: RunRequest) -> dict:
    run_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    live = _Live()
    _live[run_id] = live
    threading.Thread(target=_worker, args=(run_id, req.prompt.strip(), live), daemon=True).start()
    return {"run_id": run_id}


def _saved(run_id: str) -> list[dict] | None:
    f = RUNS / run_id / "events.jsonl"
    if "/" in run_id or ".." in run_id or not f.exists():
        return None
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    if run_id in _live:
        return {"events": _live[run_id].events, "done": _live[run_id].done}
    events = _saved(run_id)
    if events is None:
        raise HTTPException(404, "run not found")
    return {"events": events, "done": True}


@app.get("/api/runs/{run_id}/events")
async def stream(run_id: str) -> EventSourceResponse:
    live = _live.get(run_id)
    saved = None if live else _saved(run_id)
    if live is None and saved is None:
        raise HTTPException(404, "run not found")

    async def gen():
        if saved is not None:
            for ev in saved:
                yield {"data": json.dumps(ev)}
            yield {"event": "end", "data": "{}"}
            return
        i = 0
        while True:
            while i < len(live.events):
                yield {"data": json.dumps(live.events[i])}
                i += 1
            if live.done and i >= len(live.events):
                yield {"event": "end", "data": "{}"}
                return
            await asyncio.sleep(0.15)

    return EventSourceResponse(gen(), ping=10)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}
