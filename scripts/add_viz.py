"""Add browser 3D replay data (viz.json + traj.bin) to runs recorded before it existed.

Re-simulates every attempt from the Build saved in the run's events.jsonl. The simulator is
deterministic, so the replay matches the recorded video; the script checks that pass/fail agrees
and patches each runner event with its new `viz_path`. No model calls, no cost.

    uv run python scripts/add_viz.py app/featured/*/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from spec2sim.schemas import Build, Goal
from spec2sim.sim.runner import run


def add_viz(run_dir: Path) -> None:
    f = run_dir / "events.jsonl"
    events = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    task = next(e["data"] for e in events if e.get("step") == "planner" and e.get("status") == "done")
    goals = [Goal(**g) for g in task["goals"]]
    builds = {e["attempt"]: e["data"] for e in events
              if e.get("step") in ("builder", "repair") and e.get("status") == "done"}
    for e in events:
        if not (e.get("step") == "runner" and e.get("status") == "done") or e["data"].get("viz_path"):
            continue
        att = e["attempt"]
        if e["data"].get("error") and not e["data"].get("video_path"):
            continue  # scene never loaded: nothing to replay
        out = run_dir / f"attempt{att}"
        tmp = out / "_viz_tmp"
        r = run(Build(**builds[att]), goals, tmp)
        for name in ("viz.json", "traj.bin"):
            (tmp / name).replace(out / name)
        for p in tmp.iterdir():
            p.unlink()
        tmp.rmdir()
        same = r.success == e["data"]["success"]
        e["data"]["viz_path"] = f"{run_dir.name}/attempt{att}/viz.json"
        print(f"{run_dir.name} attempt {att}: success {r.success} ({'matches' if same else 'DIFFERS from'} recording)")
    f.write_text("".join(json.dumps(e) + "\n" for e in events))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        add_viz(Path(d))
