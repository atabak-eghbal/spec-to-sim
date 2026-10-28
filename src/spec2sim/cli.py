"""Command-line entry point: run one task end to end and print each pipeline step.

Usage: uv run python -m spec2sim.cli "a Franka arm stacks the red cube on the blue cube"
"""
from __future__ import annotations

import argparse
import json

from spec2sim.loop import run_task


def _fmt_stats(s: dict | None) -> str:
    if not s:
        return ""
    return (f"[{s['label']}] {s['latency_s']:.1f}s  {s['tokens_in']}->{s['tokens_out']} tok  "
            f"${s['cost_usd']:.4f}" + (f"  ({s['calls']} calls)" if s["calls"] > 1 else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prompt")
    ap.add_argument("--max-iterations", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true", help="print full step outputs")
    args = ap.parse_args()

    for ev in run_task(args.prompt, max_iterations=args.max_iterations):
        t = ev["type"]
        if t == "start":
            print(f"run {ev['run_id']}  trace: {ev['trace_url']}")
        elif t == "step" and ev["status"] == "running":
            extra = "  ⚠ ESCALATED to Ultra" if ev.get("escalated") else ""
            print(f"\n▶ {ev['step']}" + (f" (attempt {ev['attempt']})" if "attempt" in ev else "") + extra)
        elif t == "step":
            if ev.get("stats"):
                print("  " + _fmt_stats(ev["stats"]))
            d, step = ev.get("data", {}), ev["step"]
            if step == "planner":
                print(f"  {d['summary']}\n  objects: {[o['name'] for o in d['objects']]}  goals: {d['goals']}")
            elif step == "grounder":
                tv = ev["tavily"]
                print(f"  tavily: {len(tv['queries'])} queries ({tv['cached']} cached) {tv['latency_s']}s")
                for f in d["facts"]:
                    print(f"  - {f['name']}: {f['size']} m, {f['mass']} kg  ({f['source_url'] or 'estimate'})")
            elif step in ("builder", "repair"):
                print(f"  {len(d['plan']['actions'])} actions; {d['rationale']}")
            elif step == "runner":
                print(f"  {'✅ SUCCESS' if d['success'] else '❌ FAIL'}  sim {d['duration_s']}s  video {d['video_path']}")
                if d["error"]:
                    print(f"  error: {d['error']}")
                for g in d["goals"]:
                    print(f"  {'PASS' if g['ok'] else 'FAIL'} {g['detail']}")
                for e in d["events"]:
                    print(f"  · {e}")
            elif step == "critic":
                print(f"  [{d['category']}] {d['what_happened']}\n  cause: {d['root_cause']}\n  fix: {d['suggested_fix']}")
            if args.verbose:
                print(json.dumps(d, indent=1)[:3000])
        elif t == "done":
            print(f"\n{'✅' if ev['success'] else '❌'} done in {ev['attempts']} attempt(s), "
                  f"LLM cost ${ev['total_cost_usd']:.4f}")


if __name__ == "__main__":
    main()
