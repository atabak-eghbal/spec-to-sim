"""Synthetic robot training data from plain-English tasks (entrypoint for a Nebius Serverless Job).

For each prompt: run the full pipeline (plan -> ground -> build -> simulate -> repair), and if the
task is solved, re-run the plan with randomized starts, masses and friction and keep the
successful episodes as a LeRobot v2.1 dataset (parquet + mp4).

    uv run python scripts/generate_data.py --episodes 50 --out datasets \
        "Put the apple and the lime into the fruit basket" "Knock over the tower of three wooden toy blocks"
    uv run python scripts/generate_data.py --run-id 20261028-123520-54e9 --episodes 20   # from a saved run
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from spec2sim.loop import RUNS, run_task
from spec2sim.provenance import provenance
from spec2sim.schemas import Build, Goal, GroundedFacts
from spec2sim.sim.dataset import generate


def from_events(events: list[dict]):
    done = lambda step: [e for e in events if e.get("step") == step and e.get("status") == "done"]
    ok = [e["attempt"] for e in done("runner") if e["data"]["success"]]
    if not ok:
        return None
    builds = {e["attempt"]: e["data"] for e in done("builder") + done("repair")}
    task = done("planner")[0]["data"]
    prompt = next((e.get("prompt") for e in events if e["type"] == "start"), task["summary"])
    g = done("grounder")
    facts = GroundedFacts(**g[0]["data"]) if g else GroundedFacts(facts=[])
    return Build(**builds[ok[-1]]), [Goal(**g) for g in task["goals"]], prompt, facts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompts", nargs="*")
    ap.add_argument("--run-id", action="append", default=[], help="use a saved run instead of a prompt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default="datasets")
    ap.add_argument("--no-images", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    sources = [("run", r) for r in args.run_id] + [("prompt", p) for p in args.prompts]
    for kind, src in sources:
        t0 = time.time()
        if kind == "run":
            events = [json.loads(x) for x in (RUNS / src / "events.jsonl").read_text().splitlines() if x.strip()]
        else:
            events = list(run_task(src))
        got = from_events(events)
        run_id = next(e["run_id"] for e in events if e["type"] == "start")
        if got is None:
            print(f"SKIP {src[:60]!r}: task not solved", flush=True)
            summary.append({"source": src, "run_id": run_id, "solved": False})
            continue
        build, goals, prompt, facts = got
        res = generate(build, goals, prompt, out / run_id, episodes=args.episodes, images=not args.no_images,
                       physics=facts.physics(), provenance=provenance(facts))
        res.update(source=src, run_id=run_id, solved=True, total_seconds=round(time.time() - t0, 1))
        summary.append(res)
        print(f"OK   {prompt[:60]!r}: {res['episodes']} episodes / {res['tried']} tries, {res['frames']} frames "
              f"-> {res['zip']}", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
