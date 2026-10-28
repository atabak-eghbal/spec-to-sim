"""Add a stress-test step to recorded runs that succeeded (physics only, no model calls).

    uv run python scripts/add_stress.py app/featured/*/
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from spec2sim import config
from spec2sim.schemas import Build, Goal
from spec2sim.sim.runner import stress_test


def add(run_dir: Path) -> None:
    f = run_dir / "events.jsonl"
    events = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    if any(e.get("step") == "stress" for e in events):
        return
    done = next((e for e in events if e["type"] == "done"), None)
    if not done or not done.get("success"):
        return
    task = next(e["data"] for e in events if e.get("step") == "planner" and e.get("status") == "done")
    build = [e["data"] for e in events if e.get("step") in ("builder", "repair") and e.get("status") == "done"][-1]
    st = stress_test(Build(**build), [Goal(**g) for g in task["goals"]], trials=config.STRESS_TRIALS)
    ts = done["ts"]
    new = [{"type": "step", "ts": ts - 0.002, "step": "stress", "status": "running", "trials": st["trials"]},
           {"type": "step", "ts": ts - 0.001, "step": "stress", "status": "done", "data": st}]
    i = events.index(done)
    events[i:i] = new
    f.write_text("".join(json.dumps(e) + "\n" for e in events))
    print(f"{run_dir.name}: {st['passed']}/{st['trials']} robust ({st['seconds']}s)")


if __name__ == "__main__":
    t0 = time.time()
    for d in sys.argv[1:]:
        add(Path(d))
    print(f"done in {time.time() - t0:.0f}s")
