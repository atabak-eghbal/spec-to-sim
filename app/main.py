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
import gzip
import io
import json
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from spec2sim.llm import BudgetError, check_budget
from spec2sim.loop import RUNS, _public, run_task
from spec2sim.schemas import Build, Goal
from spec2sim.sim import runner
from spec2sim.sim.scene import build_scene, objects_mjcf
from spec2sim.sim.webviz import export_meshes

STATIC = Path(__file__).parent / "static"
EXAMPLES_FILE = Path(__file__).parent / "examples.json"
MAX_CONCURRENT = 2

FEATURED = Path(__file__).parent / "featured"

app = FastAPI(title="Spec-to-Sim")
RUNS.mkdir(exist_ok=True)
# Featured runs are committed under app/featured/ and copied into runs/ so they replay anywhere.
for _src in FEATURED.glob("*/"):  # app/featured is the source of truth: refresh on every start
    shutil.copytree(_src, RUNS / _src.name, dirs_exist_ok=True)
app.mount("/media", StaticFiles(directory=RUNS), name="media")

# Robot meshes for the browser 3D replay: exported once from the compiled MuJoCo model, served
# gzip-compressed (the browser caches them).
VIZ = Path(__file__).parent / ".viz"
if not (VIZ / "meshes.json").exists():
    export_meshes(build_scene([]).model, VIZ)
_MESH_GZ = gzip.compress((VIZ / "meshes.bin").read_bytes(), 6)
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
    try:
        check_budget()
    except BudgetError as e:
        raise HTTPException(429, f"Live runs are paused: {e}. The recorded runs below still replay.") from e
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


@app.get("/viz/meshes.json")
def meshes_json() -> FileResponse:
    return FileResponse(VIZ / "meshes.json", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/viz/meshes.bin")
def meshes_bin() -> Response:
    return Response(_MESH_GZ, media_type="application/octet-stream",
                    headers={"Content-Encoding": "gzip", "Cache-Control": "public, max-age=86400"})


ROBOLAB = {"on": "object_on_top", "inside": "object_in_container", "left_of": "object_left_of",
           "right_of": "object_right_of", "in_front_of": "object_in_front_of", "behind": "object_behind",
           "upright": "object_upright (no RoboLab equivalent)", "at": "object_at_position", "lifted": "object_lifted"}

EXPORT_README = """# Spec-to-Sim scene export

Generated from the prompt:  "{prompt}"
Run {run_id}, attempt {attempt} ({result}).

| File | What it is |
|---|---|
| scene_objects.xml | The objects as MuJoCo MJCF (no robot). Sizes and masses come from web sources (task.json). |
| task.json | The task: objects with real-world sizes and sources, and success predicates (RoboLab names included). |
| plan.json | The motion plan that solved it: tool-point waypoints and gripper open/close. |

## Load it with the Franka Panda from MuJoCo Menagerie
```python
import mujoco
robot = mujoco.MjSpec.from_file("mujoco_menagerie/franka_emika_panda/scene.xml")
objects = mujoco.MjSpec.from_file("scene_objects.xml")
robot.attach(objects, frame=robot.worldbody.add_frame(), prefix="task/")
model = robot.compile()
```
World frame: robot base at the origin, +x forward, +y left, z up, floor at z = 0. Units: m, kg.

## Success predicates
Each goal in task.json is checked on the final simulator state (see spec2sim/sim/metrics.py):
`on` (resting on top, height +-1.5 cm, centred within half the target's width), `inside` (object
centroid within the bin's cavity, in the bin's frame), `left_of`/`right_of`/`in_front_of`/`behind`
(>= 3 cm along the robot-frame axis), `upright` (tilt <= 20 deg). Use them to score your own policy.
"""


@app.get("/api/runs/{run_id}/export")
def export(run_id: str, attempt: int | None = None) -> Response:
    """Zip of the generated scene (MJCF), task + success predicates (JSON) and the motion plan."""
    events = _live[run_id].events if run_id in _live else _saved(run_id)
    if not events:
        raise HTTPException(404, "run not found")
    done = lambda step: [e for e in events if e.get("step") == step and e.get("status") == "done"]
    task = done("planner")[0]["data"] if done("planner") else None
    builds = {e["attempt"]: e["data"] for e in done("builder") + done("repair")}
    results = {e["attempt"]: e["data"] for e in done("runner")}
    if not task or not builds:
        raise HTTPException(409, "this run has no scene yet")
    if attempt is None:  # default: the successful attempt, else the latest
        ok = [a for a, r in results.items() if r["success"]]
        attempt = ok[0] if ok else max(builds)
    if attempt not in builds:
        raise HTTPException(404, f"attempt {attempt} not found")
    b = Build(**builds[attempt])
    grounded = done("grounder")[0] if done("grounder") else {}
    facts = {f["name"]: f for f in grounded.get("data", {}).get("facts", [])}
    res = results.get(attempt)
    prompt = next((e.get("prompt") for e in events if e["type"] == "start"), "")
    task_json = {
        "prompt": prompt, "summary": task["summary"], "robot": "franka_emika_panda (MuJoCo Menagerie)",
        "objects": [{"name": o.name, "shape": o.shape, "size_m": o.size, "mass_kg": o.mass, "start_pos_m": o.pos,
                     "fixed": o.fixed or o.shape == "bin", "source_url": facts.get(o.name, {}).get("source_url"),
                     "note": facts.get(o.name, {}).get("note")} for o in b.objects],
        "goals": [{**g, "robolab": ROBOLAB.get(g["type"])} for g in task["goals"]],
        "result": {"success": res["success"], "goals": [{"ok": g["ok"], "detail": g["detail"]} for g in res["goals"]]}
        if res else None,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("scene_objects.xml", objects_mjcf(b.objects))
        z.writestr("task.json", json.dumps(task_json, indent=2))
        z.writestr("plan.json", json.dumps(b.plan.model_dump(), indent=2))
        z.writestr("README.md", EXPORT_README.format(
            prompt=prompt, run_id=run_id, attempt=attempt,
            result="succeeded" if res and res["success"] else "failed"))
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="spec2sim-{run_id}-attempt{attempt}.zip"'})


class TryRequest(BaseModel):
    attempt: int
    moves: dict[str, tuple[float, float]] = Field(default_factory=dict, max_length=8)


_try_lock = threading.Lock()


@app.post("/api/runs/{run_id}/try")
def try_it(run_id: str, req: TryRequest) -> dict:
    """'Try to break it': re-run a saved plan with objects moved to new (x, y) spots.
    Physics only, no model calls; the plan's steps are relative to objects, so it adapts."""
    events = _live[run_id].events if run_id in _live else _saved(run_id)
    if not events:
        raise HTTPException(404, "run not found")
    done = lambda step: [e for e in events if e.get("step") == step and e.get("status") == "done"]
    task = done("planner")[0]["data"] if done("planner") else None
    builds = {e["attempt"]: e["data"] for e in done("builder") + done("repair")}
    if not task or req.attempt not in builds:
        raise HTTPException(404, "attempt not found")
    b = Build(**builds[req.attempt])
    objs = []
    for o in b.objects:
        if o.name in req.moves:
            x, y = req.moves[o.name]
            o = o.model_copy(update={"pos": (round(x, 4), round(y, 4), o.pos[2])})
        objs.append(o)
    b = b.model_copy(update={"objects": objs})
    goals = [Goal(**g) for g in task["goals"]]
    with _try_lock:  # one at a time: it's CPU work on the web server
        n = len(list((RUNS / run_id).glob("try*"))) + 1
        r = runner.run(b, goals, RUNS / run_id / f"try{n}", {o.name for o in b.objects})
    return {"try": n, "attempt": req.attempt, "result": _public(r, run_id)}


class DatasetRequest(BaseModel):
    episodes: int = Field(20, ge=5, le=50)


_datasets: dict[str, dict] = {}


@app.post("/api/runs/{run_id}/dataset")
def make_dataset(run_id: str, req: DatasetRequest) -> dict:
    """Training data from a solved task: N successful perturbed demonstrations in LeRobot v2.1
    layout (parquet + mp4), generated in the background with physics only (no model calls)."""
    job = _datasets.get(run_id)
    if job and job["state"] == "running":
        return job
    events = _live[run_id].events if run_id in _live else _saved(run_id)
    if not events:
        raise HTTPException(404, "run not found")
    done = lambda step: [e for e in events if e.get("step") == step and e.get("status") == "done"]
    ok = [e["attempt"] for e in done("runner") if e["data"]["success"]]
    if not ok:
        raise HTTPException(409, "only solved tasks can generate training data")
    builds = {e["attempt"]: e["data"] for e in done("builder") + done("repair")}
    task = done("planner")[0]["data"]
    prompt = next((e.get("prompt") for e in events if e["type"] == "start"), task["summary"])
    job = {"state": "running", "done": 0, "total": req.episodes, "tried": 0, "result": None, "error": None}
    _datasets[run_id] = job

    def work():
        from spec2sim.sim.dataset import generate
        def progress(k, n, tried):
            job.update(done=k, tried=tried)
        try:
            with _try_lock:
                out = RUNS / run_id / f"dataset_{req.episodes}"
                res = generate(Build(**builds[ok[-1]]), [Goal(**g) for g in task["goals"]], prompt, out,
                               episodes=req.episodes, progress=progress)
            res["zip"] = str(Path(res["zip"]).relative_to(RUNS))
            job.update(state="done", result=res, done=res["episodes"])
        except Exception as e:  # noqa: BLE001 - report any failure to the UI
            traceback.print_exc()
            job.update(state="error", error=f"{type(e).__name__}: {e}")

    threading.Thread(target=work, daemon=True).start()
    return job


@app.get("/api/runs/{run_id}/dataset")
def dataset_status(run_id: str) -> dict:
    job = _datasets.get(run_id)
    if job:
        return job
    for z in sorted((RUNS / run_id).glob("dataset_*.zip")) if "/" not in run_id else []:
        return {"state": "done", "result": {"zip": str(z.relative_to(RUNS)), "episodes": None}}
    return {"state": "none"}


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}
