"""Re-simulate recorded runs with the current simulator (physics only, no model calls).

The model outputs of a recorded run (task, facts, builds, critiques) are kept; every attempt's
build is executed again, its results and media replaced, and the stress test re-run on the final
plan - and on the first plan when the run needed a repair, so "before vs after repair" robustness
comes from data. The run gets a 'resimulated' marker so the UI and anyone reading events.jsonl
can tell. Exits non-zero if any attempt's pass/fail changed (the story of that run changed).

    uv run python scripts/resimulate.py app/featured/*/
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

from spec2sim import config
from spec2sim.schemas import Build, Goal, GroundedFacts
from spec2sim.sim import runner


def _rel(p: str | None, run_dir: Path) -> str | None:
    return f"{run_dir.name}/{Path(p).relative_to(run_dir)}" if p else None


def resimulate(run_dir: Path) -> list[str]:
    f = run_dir / "events.jsonl"
    events = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    done = lambda step: [e for e in events if e.get("step") == step and e.get("status") == "done"]
    task = done("planner")[0]["data"]
    goals = [Goal(**g) for g in task["goals"]]
    g = done("grounder")
    facts = GroundedFacts(**g[0]["data"]) if g else GroundedFacts(facts=[])
    physics = facts.physics()
    builds = {e["attempt"]: Build(**e["data"]) for e in done("builder") + done("repair")}
    allowed = {o["name"] for o in task["objects"]}
    changed, final = [], None
    for e in done("runner"):
        a = e["attempt"]
        if a not in builds:  # drag-and-rerun tests etc. are not part of the recorded story
            continue
        out = run_dir / f"attempt{a}"
        shutil.rmtree(out, ignore_errors=True)
        r = runner.run(builds[a], goals, out, allowed, physics)
        if r.success != e["data"]["success"]:
            changed.append(f"attempt {a}: {'pass' if e['data']['success'] else 'fail'} -> {'pass' if r.success else 'fail'}")
        d = r.model_dump()
        d.update(video_path=_rel(r.video_path, run_dir), contact_sheet_path=_rel(r.contact_sheet_path, run_dir),
                 viz_path=_rel(r.viz_path, run_dir), frame_paths=[_rel(p, run_dir) for p in r.frame_paths])
        e["data"] = d
        if r.success:
            final = a
    end = next(e for e in events if e["type"] == "done")
    events = [e for e in events if e.get("step") != "stress" and e.get("type") != "resimulated"]
    end["success"] = final is not None
    if final is not None and config.STRESS_TRIALS > 0:
        st = runner.stress_test(builds[final], goals, trials=config.STRESS_TRIALS, physics=physics)
        first = min(builds)
        if first != final:
            fp = runner.stress_test(builds[first], goals, trials=config.STRESS_TRIALS, physics=physics)
            if fp["trials"]:  # 0 = the first plan never made a valid scene
                st["first_plan"] = {"attempt": first, "passed": fp["passed"], "trials": fp["trials"]}
        i = events.index(end)
        ts = end["ts"]
        events[i:i] = [{"type": "step", "ts": ts, "step": "stress", "status": "running", "trials": st["trials"]},
                       {"type": "step", "ts": ts, "step": "stress", "status": "done", "data": st}]
    events.append({"type": "resimulated", "ts": round(time.time(), 3),
                   "note": "Re-simulated with the current simulator (no model calls). Model outputs, including "
                           "the Critic's text, are from the original run."})
    f.write_text("".join(json.dumps(e) + "\n" for e in events))
    s = next((e["data"] for e in events if e.get("step") == "stress" and e.get("status") == "done"), None)
    rob = f"stress {s['passed']}/{s['trials']}" + (f" (first plan {s['first_plan']['passed']}/{s['first_plan']['trials']})"
                                                   if s and "first_plan" in s else "") if s else "no stress test"
    print(f"{run_dir.name}: {'PASS' if final else 'FAIL'}, {rob}" + (f"  CHANGED: {changed}" if changed else ""), flush=True)
    return changed


if __name__ == "__main__":
    bad = [d for d in sys.argv[1:] if resimulate(Path(d).resolve())]
    sys.exit(1 if bad else 0)
