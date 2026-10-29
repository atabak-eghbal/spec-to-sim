"""Run your policy on a Spec-to-Sim scene and report how often it satisfies the task's goals.

    python evaluate.py --policy my_policy.py --episodes 50 --menagerie path/to/mujoco_menagerie

my_policy.py defines `act(obs) -> action` (7 arm joint targets in rad + gripper 0..255) and may define
`reset()`, called at the start of each episode. `obs` is what Spec2SimEnv.reset/step return.
Every episode is seeded (seed + i), so results are reproducible. Success is judged on the final state.
Writes results.json next to the printed table.
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from spec2sim_env import CTRL_DT, Spec2SimEnv


def load_policy(path: str):
    spec = importlib.util.spec_from_file_location("user_policy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "act"):
        raise SystemExit(f"{path} must define act(obs) -> action")
    return mod


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="python file defining act(obs)")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--horizon", type=float, default=15.0, help="seconds per episode (default 15)")
    ap.add_argument("--seed", type=int, default=0, help="episode i uses seed + i")
    ap.add_argument("--menagerie", help="mujoco_menagerie clone (or set $MENAGERIE)")
    ap.add_argument("--scene-dir", default=str(here), help="folder with task.json and scene_objects.xml")
    ap.add_argument("--grasp-assist", action="store_true", help="weld objects the closed gripper holds, like the repo")
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()

    env = Spec2SimEnv(a.scene_dir, a.menagerie, grasp_assist=a.grasp_assist)
    policy = load_policy(a.policy)
    steps = round(a.horizon / CTRL_DT)
    goals = env.goals
    passed = [0] * len(goals)
    episodes, wins, t0 = [], 0, time.time()
    for i in range(a.episodes):
        obs = env.reset(seed=a.seed + i)
        if hasattr(policy, "reset"):
            policy.reset()
        for _ in range(steps):
            obs = env.step(np.asarray(policy.act(obs), dtype=float))
        res = env.success()
        ok = all(r["ok"] for r in res)
        wins += ok
        for k, r in enumerate(res):
            passed[k] += r["ok"]
        episodes.append({"seed": a.seed + i, "success": ok, "goals": [{"ok": r["ok"], "detail": r["detail"]} for r in res],
                         "variation": env.variation})
        print(f"episode {i + 1}/{a.episodes} seed {a.seed + i}: {'PASS' if ok else 'fail'}", flush=True)

    print(f"\n{'goal':<52} {'passed':>9}")
    for g, n in zip(goals, passed):
        label = f"{g['type']} {g['obj']}" + (f" -> {g['target']}" if g.get("target") else "")
        print(f"{label:<52} {n:>4}/{a.episodes:<4}")
    print(f"\nOverall success: {wins}/{a.episodes} = {100 * wins / max(1, a.episodes):.1f} %  "
          f"({a.horizon:g} s episodes, grasp assist {'on' if a.grasp_assist else 'off'}, {time.time() - t0:.0f} s)")
    Path(a.out).write_text(json.dumps({
        "episodes": a.episodes, "horizon_s": a.horizon, "grasp_assist": a.grasp_assist, "success_rate": wins / max(1, a.episodes),
        "per_goal": [{"goal": {k: v for k, v in g.items() if v is not None}, "passed": n}
                     for g, n in zip(goals, passed)], "results": episodes}, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
