"""Policy report card: train a behaviour-cloning policy on a run's generated demos, then test it.

    uv run python scripts/train_policy.py --run-id 20261028-093211-b9fb [--episodes 50] [--trials 16]

Physics only; no model/API calls. Writes runs/<run_id>/policy_report.json.
"""
from __future__ import annotations

import argparse
import json

from generate_data import from_events

from spec2sim.loop import RUNS
from spec2sim.policy import evaluate, make_fk, train
from spec2sim.provenance import provenance
from spec2sim.sim.dataset import generate
from spec2sim.sim.runner import STRESS, _Sim


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--kind", default="mlp", choices=["mlp", "knn"])
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--members", type=int, default=5, help="MLPs averaged in the ensemble")
    args = ap.parse_args()
    run_dir = RUNS / args.run_id
    events = [json.loads(x) for x in (run_dir / "events.jsonl").read_text().splitlines() if x.strip()]
    got = from_events(events)
    if got is None:
        raise SystemExit("run was not solved")
    build, goals, prompt, facts = got
    physics = facts.physics()
    ds = run_dir / f"dataset_{args.episodes}"
    if not (ds / "meta/info.json").exists():
        print(f"generating {args.episodes} demos ...", flush=True)
        res = generate(build, goals, prompt, ds, episodes=args.episodes, images=False, physics=physics,
                       provenance=provenance(facts))
        print(f"  {res['episodes']} kept / {res['tried']} tried in {res['seconds']} s", flush=True)
    fk = make_fk(_Sim(build, None, physics=physics))
    policy, secs = train(ds, kind=args.kind, epochs=args.epochs, fk=fk, members=args.members if args.kind == "mlp" else 1)
    print(f"trained {args.kind} on {policy.episodes_trained} demos in {secs} s", flush=True)
    seed = 2000  # held-out variants: not the dataset's seed (1) nor the one used to tune (1000)
    scripted = evaluate(build, goals, args.trials, seed, None, physics)
    learned = evaluate(build, goals, args.trials, seed, policy, physics)
    report = {"episodes_trained": policy.episodes_trained, "train_seconds": secs, "policy_kind": f"{args.kind} x{args.members}" if args.kind == "mlp" else args.kind,
              "scripted": {"passed": scripted["passed"], "trials": scripted["trials"], "per_goal": scripted["per_goal"]},
              "learned": {k: learned[k] for k in ("passed", "trials", "per_goal", "failures")},
              "perturbation": {"position_cm": STRESS["pos_jitter"] * 100, "yaw_deg": 30, "mass_pct": 20,
                               "friction_pct": 30, "sensing_cm_scripted_only": STRESS["sense_error"] * 100}}
    (run_dir / "policy_report.json").write_text(json.dumps(report, indent=2))
    print(f"\n{prompt[:70]}\n{'goal':44} scripted  learned")
    for s, l in zip(scripted["per_goal"], learned["per_goal"], strict=True):
        print(f"{json.dumps(s['goal'])[:44]:44} {s['passed']:>4}/{scripted['trials']:<4} {l['passed']:>4}/{learned['trials']}")
    print(f"{'ALL GOALS':44} {scripted['passed']:>4}/{scripted['trials']:<4} {learned['passed']:>4}/{learned['trials']}")
    for f in learned["failures"][:4]:
        print("  fail:", f)


if __name__ == "__main__":
    main()
