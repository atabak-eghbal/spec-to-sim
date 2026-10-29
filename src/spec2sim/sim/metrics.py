"""Success checks computed from simulator state - the ground truth for pass/fail.

The Critic (a VLM) explains failures, but never decides them: every goal is checked here
with explicit tolerances. Predicates mirror NVIDIA RoboLab's task conditionals
(github.com/NVlabs/RoboLab, docs/task_conditionals.md): object_on_top -> on,
object_in_container -> inside (centroid inside the container's cavity, in the container's own
frame), object_left_of/... -> left_of/... in the robot's frame (+x forward, +y left).
"""
from __future__ import annotations

import mujoco
import numpy as np

from spec2sim.schemas import Goal, GoalResult
from spec2sim.sim.scene import BIN_WALL, CLOSED_MAX, OBJ_PREFIX, OPENED_MIN, Scene

ON_Z_TOL = 0.015  # resting-contact height tolerance, m
AT_TOL = 0.03  # distance tolerance for 'at', m
LIFT_MIN = 0.05  # bottom must be this far above floor for 'lifted', m
SIDE_MARGIN = 0.03  # left_of etc.: centers must differ by this much along the axis, m
UPRIGHT_MAX_TILT = 20.0  # degrees
# (axis index, sign) in the robot frame: +x is away from the robot, +y is the robot's left.
SIDES = {"left_of": (1, 1), "right_of": (1, -1), "in_front_of": (0, 1), "behind": (0, -1)}
NEEDS_TARGET = {"on", "inside", *SIDES}
TOPPLE_TILT = 45.0  # degrees, or the object dropped by half its height from where it started


def _half_h(scene: Scene, name: str) -> float:
    return scene.objects[name].size[2] / 2


PLACED = ("on", "inside", "at")


def _lean(scene: Scene, data: mujoco.MjData, name: str) -> float:
    """Degrees an object leans away from resting flat: for a box, from its nearest flat face (a box
    rolled onto another face still rests flat); for a cylinder, from standing on its end."""
    o = scene.objects[name]
    R = data.body(OBJ_PREFIX + name).xmat.reshape(3, 3)
    c = np.max(np.abs(R[2, :])) if o.shape == "box" else R[2, 2]
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def check_goal(scene: Scene, data: mujoco.MjData, g: Goal, held: str | None,
               peaks: dict[str, float] | None = None) -> GoalResult:
    """peaks: highest resting height of each loose object during the run (for 'toppled').

    A placement goal (on / inside / at) also requires the object to end the way it started: a soda
    can lying on its side in the bin was dropped, not placed."""
    r = _check(scene, data, g, held, peaks)
    if r.ok and g.type in PLACED and scene.objects[g.obj].shape in ("box", "cylinder"):
        lean = _lean(scene, data, g.obj)
        limit = TOPPLE_TILT if scene.objects[g.obj].shape == "cylinder" else UPRIGHT_MAX_TILT
        if lean >= limit:
            return r.model_copy(update={"ok": False, "detail": r.detail + f"; but it ended tipped over "
                                        f"({lean:.0f} deg): placed objects must not fall over"})
    return r


def _check(scene: Scene, data: mujoco.MjData, g: Goal, held: str | None,
           peaks: dict[str, float] | None = None) -> GoalResult:
    def pos(n: str) -> np.ndarray:
        return data.body(OBJ_PREFIX + n).xpos

    for n in (g.obj, g.target):
        if n is not None and n not in scene.objects:
            return GoalResult(goal=g, ok=False, error_m=1.0, detail=f"unknown object '{n}'")
    p = pos(g.obj)
    if g.type in NEEDS_TARGET and g.target is None:
        return GoalResult(goal=g, ok=False, error_m=1.0, detail=f"'{g.type}' goal needs a target")
    held_note = "; still held by gripper" if held == g.obj else ""

    if g.type in ("opened", "closed"):
        if g.obj not in scene.drawer_qpos:
            return GoalResult(goal=g, ok=False, error_m=1.0, detail=f"{g.obj} is not a drawer")
        out = max(0.0, float(data.qpos[scene.drawer_qpos[g.obj]]))
        if g.type == "opened":
            return GoalResult(goal=g, ok=out >= OPENED_MIN, error_m=max(0.0, OPENED_MIN - out),
                              detail=f"{g.obj} is {out * 100:.1f} cm open (need >= {OPENED_MIN * 100:.0f})")
        return GoalResult(goal=g, ok=out <= CLOSED_MAX and held != g.obj, error_m=max(0.0, out - CLOSED_MAX),
                          detail=f"{g.obj} is {out * 100:.1f} cm open (closed means <= {CLOSED_MAX * 100:.1f}){held_note}")

    if g.type == "toppled":
        o = scene.objects[g.obj]
        R = data.body(OBJ_PREFIX + g.obj).xmat.reshape(3, 3)
        if o.shape == "box":  # a box that rolled onto another face still rests flat
            tilt = float(np.degrees(np.arccos(np.clip(np.max(np.abs(R[2, :])), -1, 1))))
        else:
            tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        peak = (peaks or {}).get(g.obj, o.pos[2])
        drop = peak - float(p[2])
        tipped = o.shape == "cylinder" and tilt >= TOPPLE_TILT or o.shape == "box" and tilt >= 20
        ok = (tipped or drop >= o.size[2] / 2) and held != g.obj
        return GoalResult(goal=g, ok=ok, error_m=0.0 if ok else max(0.0, o.size[2] / 2 - drop),
                          detail=f"{g.obj} is {drop * 100:.1f} cm below the highest point it rested at"
                                 f"{f' and leans {tilt:.0f} deg' if tipped else ''} (toppled = fell at least "
                                 f"{o.size[2] / 2 * 100:.1f} cm or tipped over){held_note}")

    if g.type == "inside" and scene.objects[g.target].shape == "drawer":
        t = scene.objects[g.target]
        b = data.body(f"{OBJ_PREFIX}{g.target}/drawer")
        local = b.xmat.reshape(3, 3).T @ (p - b.xpos)
        inner = np.array(t.size[:2]) / 2 - 2 * BIN_WALL
        out_xy = np.maximum(np.abs(local[:2]) - inner, 0)
        ok_z = -t.size[2] / 2 + BIN_WALL <= local[2] <= t.size[2] / 2
        err = float(np.linalg.norm(out_xy)) + (0.0 if ok_z else 0.05)
        ok = bool(err == 0 and ok_z) and held != g.obj
        where = "inside" if err == 0 and ok_z else f"outside (off by {err * 100:.1f} cm)"
        return GoalResult(goal=g, ok=ok, error_m=err, detail=f"{g.obj} centroid is {where} {g.target}'s drawer{held_note}")

    if g.type == "inside":
        t = scene.objects[g.target]
        if t.shape != "bin":
            return GoalResult(goal=g, ok=False, error_m=1.0, detail=f"{g.target} is not a bin")
        b = data.body(OBJ_PREFIX + g.target)
        local = b.xmat.reshape(3, 3).T @ (p - b.xpos)  # centroid in the bin's frame
        inner = np.array(t.size[:2]) / 2 - BIN_WALL
        out_xy = np.maximum(np.abs(local[:2]) - inner, 0)
        floor, rim = -t.size[2] / 2 + BIN_WALL, t.size[2] / 2
        ok_z = floor <= local[2] <= rim + 0.5 * scene.objects[g.obj].size[2]
        err = float(np.linalg.norm(out_xy)) + (0.0 if ok_z else abs(local[2] - rim))
        ok = bool(err == 0 and ok_z) and held != g.obj
        where = "inside" if err == 0 and ok_z else f"outside (off by {err * 100:.1f} cm)"
        return GoalResult(goal=g, ok=ok, error_m=err,
                          detail=f"{g.obj} centroid is {where} {g.target}'s cavity{held_note}")

    if g.type in SIDES:
        axis, sign = SIDES[g.type]
        gap = sign * float(p[axis] - pos(g.target)[axis])
        words = g.type.replace("_", " ")
        return GoalResult(goal=g, ok=gap >= SIDE_MARGIN and held != g.obj, error_m=max(0.0, SIDE_MARGIN - gap),
                          detail=f"{g.obj} is {gap * 100:+.1f} cm {words} {g.target} "
                                 f"(need >= {SIDE_MARGIN * 100:.0f}){held_note}")

    if g.type == "upright":
        up = data.body(OBJ_PREFIX + g.obj).xmat.reshape(3, 3)[:, 2]
        tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
        return GoalResult(goal=g, ok=tilt <= UPRIGHT_MAX_TILT, error_m=0.0,
                          detail=f"{g.obj} is tilted {tilt:.0f} deg (max {UPRIGHT_MAX_TILT:.0f})")

    if g.type == "on":
        t = scene.objects[g.target]
        q = pos(g.target)
        dz = p[2] - (q[2] + _half_h(scene, g.target) + _half_h(scene, g.obj))
        dxy = float(np.linalg.norm(p[:2] - q[:2]))
        # Supported on top = right height and centered within half the LARGER footprint: a 6.5 cm ball
        # resting on a 3 cm block cannot be centered to within 1.5 cm (RoboLab judges support, not centering).
        o = scene.objects[g.obj]
        xy_tol = max(0.015, 0.5 * max(min(t.size[0], t.size[1]), min(o.size[0], o.size[1])))
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
