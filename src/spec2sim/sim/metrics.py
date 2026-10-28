"""Success checks computed from simulator state - the ground truth for pass/fail.

The Critic (a VLM) explains failures, but never decides them: every goal is checked here
with explicit tolerances.
"""
from __future__ import annotations

import mujoco
import numpy as np

from spec2sim.schemas import Goal, GoalResult
from spec2sim.sim.scene import OBJ_PREFIX, Scene

ON_Z_TOL = 0.015  # resting-contact height tolerance, m
AT_TOL = 0.03  # distance tolerance for 'at', m
LIFT_MIN = 0.05  # bottom must be this far above floor for 'lifted', m


def _half_h(scene: Scene, name: str) -> float:
    return scene.objects[name].size[2] / 2


def check_goal(scene: Scene, data: mujoco.MjData, g: Goal, held: str | None) -> GoalResult:
    def pos(n: str) -> np.ndarray:
        return data.body(OBJ_PREFIX + n).xpos

    for n in (g.obj, g.target):
        if n is not None and n not in scene.objects:
            return GoalResult(goal=g, ok=False, error_m=1.0, detail=f"unknown object '{n}'")
    p = pos(g.obj)

    if g.type == "on":
        if g.target is None:
            return GoalResult(goal=g, ok=False, error_m=1.0, detail="'on' goal needs a target")
        t = scene.objects[g.target]
        q = pos(g.target)
        dz = p[2] - (q[2] + _half_h(scene, g.target) + _half_h(scene, g.obj))
        dxy = float(np.linalg.norm(p[:2] - q[:2]))
        xy_tol = max(0.015, 0.5 * min(t.size[0], t.size[1]))
        ok = abs(dz) < ON_Z_TOL and dxy < xy_tol and held != g.obj
        detail = (f"{g.obj} is {dxy * 100:.1f} cm off-center from {g.target} (tol {xy_tol * 100:.1f}), "
                  f"height error {dz * 100:+.1f} cm (tol ±{ON_Z_TOL * 100:.1f})")
        if held == g.obj:
            detail += "; still held by gripper"
        return GoalResult(goal=g, ok=ok, error_m=float(np.hypot(dz, dxy)), detail=detail)

    if g.type == "at":
        if g.xyz is None:
            return GoalResult(goal=g, ok=False, error_m=1.0, detail="'at' goal needs xyz")
        d = float(np.linalg.norm(p - np.array(g.xyz)))
        return GoalResult(goal=g, ok=d < AT_TOL and held != g.obj, error_m=d,
                          detail=f"{g.obj} is {d * 100:.1f} cm from target (tol {AT_TOL * 100:.0f})")

    # lifted
    bottom = p[2] - _half_h(scene, g.obj)
    return GoalResult(goal=g, ok=bottom > LIFT_MIN, error_m=max(0.0, LIFT_MIN - bottom),
                      detail=f"{g.obj} bottom is {bottom * 100:.1f} cm above floor (need >{LIFT_MIN * 100:.0f})")
