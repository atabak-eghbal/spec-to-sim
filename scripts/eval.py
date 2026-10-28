"""Batch evaluation: run a fixed prompt suite through the full pipeline and report success rates.

Locally:   uv run python scripts/eval.py --reps 2
On Nebius: the same command is the entrypoint of the Serverless Job (see deploy/nebius/).

Writes eval/<stamp>/results.jsonl (one line per run) and eval/<stamp>/summary.md (the table in
docs/eval.md). Each run's events.jsonl lands in runs/<id>/ as usual, so any run can be replayed in
the web UI with ?run=<id>.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from spec2sim.loop import RUNS, run_task

ROOT = Path(__file__).resolve().parents[1]

SUITE = [
    # (difficulty, prompt)
    ("easy", "A Franka arm stacks the red cube on the blue cube"),
    ("easy", "Put the Rubik's cube on top of the soda can"),
    ("medium", "Pick up the tennis ball and place it on the Rubik's cube"),
    ("medium", "Stack three wooden toy blocks into a tower: red at the bottom, then green, then blue on top"),
    ("medium", "Clear the table: put the red, green and blue toy blocks into the storage bin"),
    ("hard", "Put the apple and the lime into the fruit basket, and the soda can into the recycling bin"),
    ("hard", "Tidy up: stack the red block on the blue block, then put the green block and the tennis ball into the tray"),
    ("hard", "Put the soda can to the left of the Rubik's cube and keep it upright"),
    ("beyond", "Open the drawer, put the Rubik's cube inside, and close it"),
    ("beyond", "Put the tennis ball and the Rubik's cube into the drawer and close it"),
    ("beyond", "Knock over the tower of three wooden toy blocks"),
]


def one(difficulty: str, prompt: str, rep: int) -> dict:
    t0, run_id, last, first_ok = time.time(), None, {}, None
    log = None
    try:
        for ev in run_task(prompt):
            if ev["type"] == "start":
                run_id = ev["run_id"]
                (RUNS / run_id).mkdir(parents=True, exist_ok=True)
                log = (RUNS / run_id / "events.jsonl").open("w")
            if log:
                log.write(json.dumps(ev) + "\n")
            if ev["type"] == "step" and ev["step"] == "runner" and ev["status"] == "done" and first_ok is None:
                first_ok = ev["data"]["success"]
            last = ev
        err = None
    except Exception as e:  # noqa: BLE001 - one broken run must not stop the batch
        err = f"{type(e).__name__}: {e}"[:300]
    finally:
        if log:
            log.close()
    return {"difficulty": difficulty, "prompt": prompt, "rep": rep, "run_id": run_id,
            "success": bool(last.get("success")), "first_try": bool(first_ok),
            "attempts": last.get("attempts"), "infeasible": last.get("infeasible"),
            "cost_usd": last.get("total_cost_usd", 0.0), "wall_s": round(time.time() - t0, 1), "error": err}


def summarize(rows: list[dict]) -> str:
    lines = ["| Difficulty | Prompt | Runs | Success | First try | Avg attempts | Avg cost | Avg time |",
             "|---|---|---|---|---|---|---|---|"]
    for diff, prompt in dict.fromkeys((r["difficulty"], r["prompt"]) for r in rows):
        rs = [r for r in rows if r["prompt"] == prompt]
        n = len(rs)
        att = [r["attempts"] for r in rs if r["attempts"]]
        lines.append(f"| {diff} | {prompt} | {n} | {sum(r['success'] for r in rs)}/{n} | "
                     f"{sum(r['first_try'] for r in rs)}/{n} | {sum(att) / max(1, len(att)):.1f} | "
                     f"${sum(r['cost_usd'] for r in rs) / n:.3f} | {sum(r['wall_s'] for r in rs) / n:.0f} s |")
    n = len(rows)
    overall = (f"**Overall: {sum(r['success'] for r in rows)}/{n} succeeded "
               f"({sum(r['first_try'] for r in rows)}/{n} on the first try), "
               f"total LLM cost ${sum(r['cost_usd'] for r in rows):.2f}.**")
    lines += ["", overall]
    errs = [r for r in rows if r["error"]]
    if errs:
        lines += ["", "Crashed runs:"] + [f"- {r['prompt']}: {r['error']}" for r in errs]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2, help="parallel runs (rendering is CPU-bound)")
    ap.add_argument("--only", choices=["easy", "medium", "hard", "beyond"], default=None)
    args = ap.parse_args()

    jobs = [(d, p, r) for r in range(args.reps) for d, p in SUITE if not args.only or d == args.only]
    out = ROOT / "eval" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(args.workers) as ex, (out / "results.jsonl").open("w") as f:
        for row in ex.map(lambda j: one(*j), jobs):
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{'OK ' if row['success'] else 'BAD'} {row['attempts']} att ${row['cost_usd']:.3f} "
                  f"{row['wall_s']:>5}s  {row['prompt'][:70]}", flush=True)
    summary = summarize(rows)
    (out / "summary.md").write_text(summary + "\n")
    print("\n" + summary + f"\n\nSaved to {out}")


if __name__ == "__main__":
    main()
