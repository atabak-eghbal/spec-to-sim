"""Batch evaluation: run a fixed prompt suite through the full pipeline and report success rates.

Locally:   uv run python scripts/eval.py --reps 3 --no-critic-reps 1
Packaged to run as a Nebius Serverless Job (see deploy/nebius/); so far it has run locally.

Writes eval/<stamp>/results.jsonl (one line per run) and eval/<stamp>/summary.md (the table in
docs/eval.md). Each run's events.jsonl lands in runs/<id>/ as usual, so any run can be replayed in
the web UI with ?run=<id>.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from spec2sim.loop import RUNS, run_task

ROOT = Path(__file__).resolve().parents[1]

SUITE = [
    # (difficulty, prompt) - the development suite: these prompts were used while building the system.
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

# Held out: written and committed before any of them was run, and never used to change the code.
# The last one is meant to be refused (a basketball does not fit the gripper): a refusal counts as a pass.
HELDOUT = [
    ("easy", "Put the lime on the red toy block"),
    ("easy", "Stack the blue block on the green block"),
    ("easy", "Put the Rubik's cube into the storage bin"),
    ("easy", "Pick up the tennis ball and drop it into the fruit basket"),
    ("medium", "Place the apple to the right of the soda can"),
    ("medium", "Put the red block in front of the blue block"),
    ("medium", "Put the orange and the apple into the fruit basket"),
    ("medium", "Build a tower of two Rubik's cubes"),
    ("medium", "Put the wooden block behind the soda can"),
    ("hard", "Stack the green block on the Rubik's cube, then put the lime to the left of the Rubik's cube"),
    ("hard", "Sort the toys: the red block goes into the red bin and the blue block into the blue bin"),
    ("beyond", "Open the drawer"),
    ("beyond", "Put the tennis ball into the drawer and close it"),
    ("beyond", "Knock the soda can over"),
    ("refuse", "Put the basketball into the storage bin"),
]


def one(suite: str, difficulty: str, prompt: str, rep: int, use_critic: bool = True) -> dict:
    t0, run_id, last, first_ok = time.time(), None, {}, None
    log, tavily, stress, scene_ok = None, 0, None, None
    try:
        for ev in run_task(prompt, use_critic=use_critic):
            if ev["type"] == "start":
                run_id = ev["run_id"]
                (RUNS / run_id).mkdir(parents=True, exist_ok=True)
                log = (RUNS / run_id / "events.jsonl").open("w")
            if log:
                log.write(json.dumps(ev) + "\n")
            if ev["type"] == "step" and ev["step"] == "runner" and ev["status"] == "done" and first_ok is None:
                first_ok = ev["data"]["success"]
            if ev["type"] == "step" and ev["step"] == "grounder" and ev["status"] == "done":
                t = ev.get("tavily") or {}
                tavily = len(t.get("queries", [])) - t.get("cached", 0) + len(t.get("physics_searched", []))
            if ev["type"] == "step" and ev["step"] == "scenecheck" and ev["status"] == "done":
                scene_ok = (ev.get("data") or {}).get("ok")
            if ev["type"] == "step" and ev["step"] == "stress" and ev["status"] == "done":
                stress = [ev["data"]["passed"], ev["data"]["trials"]]
            last = ev
        err = None
    except Exception as e:  # noqa: BLE001 - one broken run must not stop the batch
        err = f"{type(e).__name__}: {e}"[:300]
    finally:
        if log:
            log.close()
    refused = bool(last.get("infeasible"))
    passed = refused if difficulty == "refuse" else bool(last.get("success"))
    return {"suite": suite, "critic": use_critic, "difficulty": difficulty, "prompt": prompt, "rep": rep, "run_id": run_id,
            "success": passed, "first_try": refused if difficulty == "refuse" else bool(first_ok),
            "stress": stress, "tavily_uncached": tavily, "scene_ok": scene_ok,
            "attempts": last.get("attempts"), "infeasible": last.get("infeasible"),
            "cost_usd": last.get("total_cost_usd", 0.0), "wall_s": round(time.time() - t0, 1), "error": err}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def rate(rows: list[dict], key: str = "success") -> str:
    k, n = sum(r[key] for r in rows), len(rows)
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / max(1, n):.0f} %, 95 % CI {100 * lo:.0f}-{100 * hi:.0f} %)"


def summarize(rows: list[dict]) -> str:
    lines = ["| Suite | Critic | Difficulty | Prompt | Runs | Success | First try | Avg attempts | Avg cost | Avg time | Stress (passed/trials) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for suite, critic, diff, prompt in dict.fromkeys((r["suite"], r["critic"], r["difficulty"], r["prompt"]) for r in rows):
        rs = [r for r in rows if r["prompt"] == prompt and r["critic"] == critic]
        n = len(rs)
        att = [r["attempts"] for r in rs if r["attempts"]]
        st = [r["stress"] for r in rs if r.get("stress")]
        stress = ", ".join(f"{a}/{b}" for a, b in st) or "-"
        lines.append(f"| {suite} | {'on' if critic else 'off'} | {diff} | {prompt} | {n} | {sum(r['success'] for r in rs)}/{n} | "
                     f"{sum(r['first_try'] for r in rs)}/{n} | {sum(att) / max(1, len(att)):.1f} | "
                     f"${sum(r['cost_usd'] for r in rs) / n:.3f} | {sum(r['wall_s'] for r in rs) / n:.0f} s | {stress} |")
    lines.append("")
    for suite in dict.fromkeys(r["suite"] for r in rows):
        for critic in (True, False):
            rs = [r for r in rows if r["suite"] == suite and r["critic"] == critic]
            if not rs:
                continue
            costs = sorted(r["cost_usd"] for r in rs)
            st = [r["stress"] for r in rs if r.get("stress")]
            lines.append(f"- **{suite}, Critic {'on' if critic else 'off'}:** success {rate(rs)}; first try "
                         f"{rate(rs, 'first_try')}; model cost mean ${sum(costs) / len(costs):.3f}, median "
                         f"${costs[len(costs) // 2]:.3f}, total ${sum(costs):.2f}; uncached Tavily searches "
                         f"{sum(r.get('tavily_uncached') or 0 for r in rs)}"
                         + (f"; stress test {sum(a for a, _ in st)}/{sum(b for _, b in st)} perturbed re-runs passed"
                            if st else ""))
    # Does the vision scene check (before the robot moves) predict whether attempt 1 fails?
    sc = [r for r in rows if r.get("scene_ok") is not None]
    if sc:
        flag_fail = sum(1 for r in sc if r["scene_ok"] is False and not r["first_try"])
        flag_ok = sum(1 for r in sc if r["scene_ok"] is False and r["first_try"])
        pass_fail = sum(1 for r in sc if r["scene_ok"] and not r["first_try"])
        pass_ok = sum(1 for r in sc if r["scene_ok"] and r["first_try"])
        lines += ["", (f"- **Vision scene check vs attempt 1:** flagged and attempt 1 failed {flag_fail}; flagged but it "
                       f"passed {flag_ok}; said ok but it failed {pass_fail}; said ok and it passed {pass_ok} (of {len(sc)})")]
    errs = [r for r in rows if r["error"]]
    if errs:
        lines += ["", "Crashed runs (counted as failures):"] + [f"- {r['prompt']}: {r['error']}" for r in errs]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2, help="parallel runs (rendering is CPU-bound)")
    ap.add_argument("--suite", choices=["dev", "heldout", "all"], default="all")
    ap.add_argument("--no-critic-reps", type=int, default=0,
                    help="extra reps of each prompt with the VLM Critic off (repairs see measurements only)")
    ap.add_argument("--max-cost", type=float, default=4.0, help="stop starting new runs past this model spend, $")
    args = ap.parse_args()

    suites = {"dev": SUITE, "heldout": HELDOUT}
    chosen = [k for k in suites if args.suite in (k, "all")]
    jobs = [(k, d, p, r, True) for r in range(args.reps) for k in chosen for d, p in suites[k]]
    jobs += [(k, d, p, r, False) for r in range(args.no_critic_reps) for k in chosen for d, p in suites[k]]
    out = ROOT / "eval" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "commit.txt").write_text(subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True,
                                                   cwd=ROOT).stdout)
    rows, spent, lock = [], [0.0], threading.Lock()

    def guarded(job):
        with lock:
            if spent[0] > args.max_cost:
                return None
        row = one(*job)
        with lock:
            spent[0] += row["cost_usd"] or 0.0
        return row

    with ThreadPoolExecutor(args.workers) as ex, (out / "results.jsonl").open("w") as f:
        for row in ex.map(guarded, jobs):
            if row is None:
                continue
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{'OK ' if row['success'] else 'BAD'} {row['suite']:7} critic={'on ' if row['critic'] else 'off'} "
                  f"{row['attempts']} att ${row['cost_usd']:.3f} {row['wall_s']:>5}s  {row['prompt'][:60]}", flush=True)
    summary = summarize(rows)
    if len(rows) < len(jobs):
        summary += f"\n\nStopped early at the ${args.max_cost:.2f} cap: {len(rows)}/{len(jobs)} runs done."
    (out / "summary.md").write_text(summary + "\n")
    print("\n" + summary + f"\n\nSaved to {out}")


if __name__ == "__main__":
    main()
