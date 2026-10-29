"""Spec2SimEnv: a standalone test environment for a Spec-to-Sim scene (needs only `mujoco` and `numpy`).

It loads the Franka Panda from MuJoCo Menagerie, attaches `scene_objects.xml`, reads the objects and
goals from `task.json`, and gives you a gym-like loop:

    env = Spec2SimEnv(".", menagerie="path/to/mujoco_menagerie")
    obs = env.reset(seed=0)                  # same randomization as the Spec-to-Sim stress test
    obs = env.step(action)                   # 7 arm joint targets (rad) + gripper (0 closed .. 255 open)
    env.success()                            # per-goal pass/fail, same predicates and tolerances as the repo

Get the robot once:  git clone https://github.com/google-deepmind/mujoco_menagerie
(`menagerie` may be the clone, its franka_emika_panda folder, or scene.xml itself; or set $MENAGERIE.)

Differences from the Spec-to-Sim runner, stated plainly:
- Grasp assist is OFF by default. The runner welds an object to the hand when both finger pads touch it
  (or the fingers closed to its width); `grasp_assist=True` mimics that rule (Controller.try_grasp).
  Without it, grasping is pure contact physics, which is harder: a policy that passes with assist may
  slip without it.
- The control period is 0.1 s (the runner's is 0.01 s). Gravity compensation is applied every 10 ms.
- The runner's simulated camera error on object positions (`sense_error`) is not applied: your policy
  reads true poses from `obs`, so add noise yourself if you want it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import mujoco
import numpy as np

# BEGIN GENERATED CONSTANTS (src/spec2sim/harness.py rewrites this block from the repo's own values)
ON_Z_TOL = 0.015
AT_TOL = 0.03
LIFT_MIN = 0.05
SIDE_MARGIN = 0.03
UPRIGHT_MAX_TILT = 20.0
TOPPLE_TILT = 45.0
BIN_WALL = 0.008
OPENED_MIN = 0.08
CLOSED_MAX = 0.012
MAX_GRASP_WIDTH = 0.078
ONE_PAD_SLACK = 0.006
MIN_FINGER_GAP = 0.002
REACH_X = (0.3, 0.7)
REACH_Y = (-0.4, 0.4)
REACH_R = 0.78
HOME_Q = (0, 0, 0, -1.57079, 0, 1.57079, -0.7853)
TCP_OFFSET = 0.1034
VARIATIONS = {"pos_jitter_m": 0.04, "yaw_jitter_rad": 0.52, "mass_scale": 0.2, "friction_scale": 0.3}
# END GENERATED CONSTANTS

CTRL_DT = 0.1  # seconds per env.step (one policy decision)
TICK = 0.01  # internally the joint target moves smoothly toward the new action every 10 ms, as in Spec-to-Sim
GRAV_COMP_DT = 0.01  # gravity compensation is refreshed this often inside a step (as in the runner)
GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
SIDES = {"left_of": (1, 1), "right_of": (1, -1), "in_front_of": (0, 1), "behind": (0, -1)}
NEEDS_TARGET = {"on", "inside", *SIDES}
PLACED = ("on", "inside", "at")
PREFIX = "task/"  # prefix used when attaching scene_objects.xml
OBJ = PREFIX + "obj/"  # body-name prefix of every object in that file



def vertical_half(shape: str, size, R) -> float:
    """Half of an object's current vertical extent (it may be lying on its side)."""
    sx, sy, sz = size
    if shape == "sphere":
        return sx / 2
    c = min(1.0, abs(float(R[2, 2])))
    if shape == "cylinder":
        return c * sz / 2 + (1 - c * c) ** 0.5 * max(sx, sy) / 2
    return float(sum(abs(float(R[2, i])) * size[i] / 2 for i in range(3)))

class Obj:
    """One object from task.json (size in metres: x, y, z)."""

    def __init__(self, d: dict):
        self.name, self.shape = d["name"], d["shape"]
        self.size = tuple(d["size_m"])
        self.fixed = bool(d.get("fixed")) or self.shape in ("bin", "drawer")
        self.z0 = d["start_pos_m"][2]


def _find_robot_xml(path: str | None) -> str:
    path = path or os.environ.get("MENAGERIE")
    if not path:
        raise ValueError("Menagerie path missing: pass menagerie=... or set $MENAGERIE "
                         "(git clone https://github.com/google-deepmind/mujoco_menagerie)")
    p = Path(path).expanduser()
    for c in (p, p / "scene.xml", p / "franka_emika_panda" / "scene.xml"):
        if c.is_file() and c.suffix == ".xml":
            return str(c)
    raise FileNotFoundError(f"no franka_emika_panda/scene.xml under {p}")


class Spec2SimEnv:
    def __init__(self, scene_dir: str = ".", menagerie: str | None = None, grasp_assist: bool = False):
        d = Path(scene_dir)
        task = json.loads((d / "task.json").read_text())
        self.objs = {o["name"]: Obj(o) for o in task["objects"]}
        self.goals = [{k: g.get(k) for k in ("type", "obj", "target", "xyz")} for g in task["goals"]]
        self.loose = [n for n, o in self.objs.items() if not o.fixed]
        self.grasp_assist = grasp_assist

        robot = mujoco.MjSpec.from_file(_find_robot_xml(menagerie))
        objects = mujoco.MjSpec.from_file(str(d / "scene_objects.xml"))
        # The runner gives the robot's geoms contact priority 2 when objects carry web-sourced physics
        # (priority 1), so a bouncy ball is not kicked away by a fingertip. Same rule here.
        webphys = any(g.priority == 1 for b in objects.bodies for g in b.geoms)
        robot.attach(objects, frame=robot.worldbody.add_frame(), prefix=PREFIX)
        if webphys:
            for b in robot.bodies:
                if b.name and not b.name.startswith(PREFIX) and b.name != "world":
                    for g in b.geoms:
                        g.priority = 2
        if grasp_assist:  # one inactive weld per graspable object (a drawer: its sliding body)
            for n, o in self.objs.items():
                if not o.fixed or o.shape == "drawer":
                    robot.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                       name1="hand", name2=OBJ + n + ("/drawer" if o.shape == "drawer" else ""),
                                       active=False, name=f"grip/{n}")
        self.m = robot.compile()
        self.d = mujoco.MjData(self.m)
        m = self.m
        joints = [m.joint(f"joint{i}") for i in range(1, 8)]
        self.arm_qpos = np.array([j.qposadr[0] for j in joints])
        self.arm_dof = np.array([j.dofadr[0] for j in joints])
        self.finger_qpos = [int(m.joint(f"finger_joint{i}").qposadr[0]) for i in (1, 2)]
        self.finger_dof = [int(m.joint(f"finger_joint{i}").dofadr[0]) for i in (1, 2)]
        self.hand = m.body("hand").id
        self.n_sub = max(1, round(GRAV_COMP_DT / m.opt.timestep))
        self.drawer_qpos = {n: int(m.joint(f"{PREFIX}{n}/slide").qposadr[0])
                            for n, o in self.objs.items() if o.shape == "drawer"}
        # Free-joint address and starting pose of each loose object; baseline physics to randomize around.
        self.free = {n: int(m.jnt_qposadr[m.body_jntadr[m.body(OBJ + n).id]]) for n in self.loose}
        self.start_pose = {n: self.d.qpos[a:a + 7].copy() for n, a in self.free.items()}
        mujoco.mj_forward(m, self.d)
        self.start_xpos = {n: self.d.body(OBJ + n).xpos.copy() for n in self.objs}
        self.obj_geoms = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith(OBJ)]
        self.base_fric = m.geom_friction[:, 0].copy()
        self.base_mass, self.base_inertia = m.body_mass.copy(), m.body_inertia.copy()
        self.weld = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"grip/{n}")
                     for n in self.objs if grasp_assist and (not self.objs[n].fixed or self.objs[n].shape == "drawer")}
        self.held: str | None = None
        self.peaks: dict[str, float] = {}
        self.variation: dict = {}
        self.action = np.array([*HOME_Q, GRIP_OPEN])

    # ---------- episode ----------
    def _overlap(self, pos: dict[str, np.ndarray]) -> bool:
        """Axis-aligned footprint overlap, the same test the scene builder uses (furniture included)."""
        names = list(self.objs)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                A, B, pa, pb = self.objs[a], self.objs[b], pos[a], pos[b]
                if (abs(pa[0] - pb[0]) < (A.size[0] + B.size[0]) / 2 - 0.002
                        and abs(pa[1] - pb[1]) < (A.size[1] + B.size[1]) / 2 - 0.002
                        and abs(pa[2] - pb[2]) < (A.size[2] + B.size[2]) / 2 - 0.002):
                    return True
        return False

    def _draw(self, rng) -> dict:
        """One random variant, drawn in the same order as the stress test's perturb(): per loose object
        dx, dy, (yaw unless a sphere), mass factor; then one friction factor for all objects."""
        v = VARIATIONS
        out = {}
        for n in self.loose:
            dx, dy = rng.uniform(-v["pos_jitter_m"], v["pos_jitter_m"], 2)
            dyaw = rng.uniform(-v["yaw_jitter_rad"], v["yaw_jitter_rad"]) if self.objs[n].shape != "sphere" else 0.0
            out[n] = (float(dx), float(dy), float(dyaw), float(rng.uniform(1 - v["mass_scale"], 1 + v["mass_scale"])))
        out["_friction"] = float(rng.uniform(1 - v["friction_scale"], 1 + v["friction_scale"]))
        return out

    def reset(self, seed: int | None = None) -> dict:
        """Start a new episode with the robot at home and the objects randomly perturbed
        (loose objects +-4 cm in x/y, +-30 deg yaw, mass +-20 %; friction of all objects +-30 %;
        bins and drawers never move). Variants with overlapping or out-of-reach objects are redrawn."""
        rng = np.random.default_rng(seed)
        for _ in range(50):
            var = self._draw(rng)
            pos = dict(self.start_xpos)
            for n in self.loose:
                pos[n] = self.start_xpos[n] + np.array([var[n][0], var[n][1], 0.0])
            if not self._overlap(pos) and all(
                    REACH_X[0] <= pos[n][0] <= REACH_X[1] and REACH_Y[0] <= pos[n][1] <= REACH_Y[1]
                    and np.hypot(pos[n][0], pos[n][1]) <= REACH_R for n in self.loose):
                break
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_qpos] = HOME_Q
        d.qpos[self.finger_qpos] = 0.04
        for n in self.loose:
            dx, dy, dyaw, k = var[n]
            a = self.free[n]
            p = self.start_pose[n]
            d.qpos[a:a + 7] = p
            d.qpos[a], d.qpos[a + 1] = p[0] + dx, p[1] + dy
            qz = np.array([np.cos(dyaw / 2), 0, 0, np.sin(dyaw / 2)])
            mujoco.mju_mulQuat(d.qpos[a + 3:a + 7], qz, p[3:7])
            b = m.body(OBJ + n).id
            m.body_mass[b], m.body_inertia[b] = self.base_mass[b] * k, self.base_inertia[b] * k
        m.geom_friction[self.obj_geoms, 0] = self.base_fric[self.obj_geoms] * var["_friction"]
        self.held, self._tried = None, False
        self.action = np.array([*HOME_Q, GRIP_OPEN])
        d.ctrl[:8] = self.action
        mujoco.mj_forward(m, d)
        self.peaks = {n: float(d.body(OBJ + n).xpos[2]) for n in self.loose}
        self._next_peak = 0.0
        self.variation = {"seed": seed, "friction_x": round(var["_friction"], 3),
                          "objects": {n: {"dx_cm": round(var[n][0] * 100, 1), "dy_cm": round(var[n][1] * 100, 1),
                                          "yaw_deg": round(float(np.degrees(var[n][2])), 1),
                                          "mass_x": round(var[n][3], 2)} for n in self.loose}}
        return self.obs()

    def tcp_pos(self) -> np.ndarray:
        """Tool point: the center between the fingertips (TCP_OFFSET below the hand flange)."""
        return self.d.xpos[self.hand] + self.d.xmat[self.hand].reshape(3, 3)[:, 2] * TCP_OFFSET

    def obs(self) -> dict:
        d = self.d
        return {"time": float(d.time), "arm_qpos": d.qpos[self.arm_qpos].copy(),
                "finger_qpos": d.qpos[self.finger_qpos].copy(), "tcp_pos": self.tcp_pos(),
                "drawers": {n: float(d.qpos[q]) for n, q in self.drawer_qpos.items()},
                "objects": {n: {"pos": d.body(OBJ + n).xpos.copy(), "quat": d.body(OBJ + n).xquat.copy()}
                            for n in self.loose}}

    def step(self, action) -> dict:
        """action: 7 arm joint position targets (rad) + gripper (0 closed .. 255 open). Advances 0.1 s."""
        a = np.asarray(action, dtype=float).reshape(8)
        prev, self.action = self.action.copy(), a
        m, d = self.m, self.d
        ticks = max(1, round(CTRL_DT / TICK))
        sub = max(1, round(TICK / m.opt.timestep))
        for i in range(ticks):  # move the target smoothly, like a real position controller's trajectory
            d.ctrl[:8] = prev + (a - prev) * (i + 1) / ticks
            d.qfrc_applied[self.arm_dof] = d.qfrc_bias[self.arm_dof]  # gravity compensation
            mujoco.mj_step(m, d, nstep=sub)
        if self.grasp_assist:
            if a[7] < 128 and self.held is None and not self._tried:
                # Judge the grasp once per close, after the fingers have stopped moving (as Spec-to-Sim does).
                if np.abs(d.qvel[self.finger_dof]).max() < 0.01:
                    self._try_grasp()
                    self._tried = True
            elif a[7] >= 128:
                self._tried = False
                if self.held is not None:
                    self._weld(self.held, False)
                    self.held = None
        # Highest height each loose object has RESTED at (not held, nearly still): 'toppled' uses it.
        for n in self.loose:
            if n != self.held and np.linalg.norm(d.body(OBJ + n).cvel[3:]) < 0.02:
                self.peaks[n] = max(self.peaks[n], float(d.body(OBJ + n).xpos[2]))
        return self.obs()

    # ---------- grasp assist (opt-in; mirrors Controller.try_grasp) ----------
    def _width_across(self, o: Obj, R: np.ndarray, axis: np.ndarray) -> float:
        if o.shape == "sphere":
            return max(o.size)
        if o.shape == "cylinder":
            c = abs(float(R[:, 2] @ axis))
            return max(o.size[:2]) * np.sqrt(max(0.0, 1 - c * c)) + o.size[2] * c
        return float(np.sum(np.abs(R.T @ axis) * np.array(o.size)))

    def _try_grasp(self) -> None:
        m, d = self.m, self.d
        fingers = {m.body("left_finger").id: "left", m.body("right_finger").id: "right"}
        touch: dict[int, set[str]] = {}
        for c in d.contact[: d.ncon]:
            if c.dist > 0.002:
                continue
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            for f, other in ((b1, b2), (b2, b1)):
                if f in fingers:
                    touch.setdefault(int(other), set()).add(fingers[f])
        axis = d.xmat[self.hand].reshape(3, 3)[:, 1]  # fingers slide along hand y
        gap = float(d.qpos[self.finger_qpos].sum())
        for n, o in self.objs.items():
            if o.fixed and o.shape != "drawer":
                continue
            if o.shape == "drawer":  # only the handle bar can be grasped
                gid = m.geom(f"{PREFIX}{n}/handle").id
                body = int(m.geom_bodyid[gid])
                width = float(np.sum(np.abs(d.geom_xmat[gid].reshape(3, 3).T @ axis) * 2 * m.geom_size[gid]))
                center, top = d.geom_xpos[gid], d.geom_xpos[gid][2] + m.geom_size[gid][2]
            else:
                b = d.body(OBJ + n)
                body, width = b.id, self._width_across(o, b.xmat.reshape(3, 3), axis)
                center, top = b.xpos, b.xpos[2] + o.size[2] / 2
            sides = touch.get(body, set())
            if not sides or width > MAX_GRASP_WIDTH:
                continue
            tcp = self.tcp_pos()  # the object must be between the fingers, not under a fingertip
            if abs(float((center - tcp) @ axis)) > gap / 2 + 0.003 or tcp[2] > top - 0.004:
                continue
            if len(sides) < 2 and gap - width > ONE_PAD_SLACK:
                continue
            if gap < MIN_FINGER_GAP:
                continue
            self._weld(n, True)
            self.held = n
            return

    def _weld(self, name: str, on: bool) -> None:
        m, d = self.m, self.d
        eq = self.weld[name]
        if on:  # relpose = object pose in the hand frame, so the weld holds it exactly where it is now
            o = self.objs[name]
            obj = d.body(OBJ + name + ("/drawer" if o.shape == "drawer" else ""))
            hR = d.xmat[self.hand].reshape(3, 3)
            hq_inv, rel_q = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(hq_inv, d.xquat[self.hand])
            mujoco.mju_mulQuat(rel_q, hq_inv, obj.xquat)
            m.eq_data[eq, :3] = 0
            m.eq_data[eq, 3:6] = hR.T @ (obj.xpos - d.xpos[self.hand])
            m.eq_data[eq, 6:10] = rel_q
            m.eq_data[eq, 10] = 1.0
        d.eq_active[eq] = on

    # ---------- success predicates (ported from spec2sim/sim/metrics.py) ----------
    def success(self) -> list[dict]:
        """Per-goal results on the current state: [{"goal": {...}, "ok": bool, "detail": str}, ...]."""
        return [self.check(g) for g in self.goals]

    def all_ok(self) -> bool:
        return all(r["ok"] for r in self.success())

    def _lean(self, name: str) -> float:
        """Degrees an object leans away from resting flat (box: from its nearest flat face)."""
        R = self.d.body(OBJ + name).xmat.reshape(3, 3)
        c = np.max(np.abs(R[2, :])) if self.objs[name].shape == "box" else R[2, 2]
        return float(np.degrees(np.arccos(np.clip(c, -1, 1))))

    def check(self, g: dict) -> dict:
        r = self._check(g)
        if r["ok"] and g["type"] in PLACED and self.objs[g["obj"]].shape in ("box", "cylinder"):
            lean = self._lean(g["obj"])
            limit = TOPPLE_TILT if self.objs[g["obj"]].shape == "cylinder" else UPRIGHT_MAX_TILT
            if lean >= limit:
                return {**r, "ok": False, "detail": r["detail"] + f"; but it ended tipped over ({lean:.0f} deg)"}
        return r

    def _check(self, g: dict) -> dict:
        d, objs, held = self.d, self.objs, self.held

        def res(ok, detail):
            return {"goal": g, "ok": bool(ok), "detail": detail}

        def pos(n):
            return d.body(OBJ + n).xpos

        for n in (g["obj"], g.get("target")):
            if n is not None and n not in objs:
                return res(False, f"unknown object '{n}'")
        o, p, typ = objs[g["obj"]], pos(g["obj"]), g["type"]
        if typ in NEEDS_TARGET and g.get("target") is None:
            return res(False, f"'{typ}' goal needs a target")
        held_note = "; still held by gripper" if held == g["obj"] else ""
        not_held = held != g["obj"]

        if typ in ("opened", "closed"):
            if g["obj"] not in self.drawer_qpos:
                return res(False, f"{g['obj']} is not a drawer")
            out = max(0.0, float(d.qpos[self.drawer_qpos[g["obj"]]]))
            if typ == "opened":
                return res(out >= OPENED_MIN, f"{g['obj']} is {out * 100:.1f} cm open (need >= {OPENED_MIN * 100:.0f})")
            return res(out <= CLOSED_MAX and not_held,
                       f"{g['obj']} is {out * 100:.1f} cm open (closed means <= {CLOSED_MAX * 100:.1f}){held_note}")

        if typ == "toppled":
            R = d.body(OBJ + g["obj"]).xmat.reshape(3, 3)
            c = np.max(np.abs(R[2, :])) if o.shape == "box" else R[2, 2]  # a box on another face rests flat
            tilt = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
            drop = self.peaks.get(g["obj"], o.z0) - float(p[2])
            tipped = o.shape == "cylinder" and tilt >= TOPPLE_TILT or o.shape == "box" and tilt >= 20
            return res((tipped or drop >= o.size[2] / 2) and not_held,
                       f"{g['obj']} is {drop * 100:.1f} cm below its highest rest, leans {tilt:.0f} deg "
                       f"(toppled = fell >= {o.size[2] / 2 * 100:.1f} cm or tipped over){held_note}")

        if typ == "inside" and objs[g["target"]].shape == "drawer":
            t = objs[g["target"]]
            b = d.body(f"{OBJ}{g['target']}/drawer")
            local = b.xmat.reshape(3, 3).T @ (p - b.xpos)
            inner = np.array(t.size[:2]) / 2 - 2 * BIN_WALL
            out_xy = np.maximum(np.abs(local[:2]) - inner, 0)
            ok_z = -t.size[2] / 2 + BIN_WALL <= local[2] <= t.size[2] / 2
            err = float(np.linalg.norm(out_xy)) + (0.0 if ok_z else 0.05)
            where = "inside" if err == 0 and ok_z else f"outside (off by {err * 100:.1f} cm)"
            return res(err == 0 and ok_z and not_held, f"{g['obj']} centroid is {where} {g['target']}'s drawer{held_note}")

        if typ == "inside":
            t = objs[g["target"]]
            if t.shape != "bin":
                return res(False, f"{g['target']} is not a bin")
            b = d.body(OBJ + g["target"])
            local = b.xmat.reshape(3, 3).T @ (p - b.xpos)  # centroid in the bin's frame
            inner = np.array(t.size[:2]) / 2 - BIN_WALL
            out_xy = np.maximum(np.abs(local[:2]) - inner, 0)
            floor, rim = -t.size[2] / 2 + BIN_WALL, t.size[2] / 2
            # In the cavity: centroid above the floor and the object's BOTTOM below the rim.
            bottom = local[2] - vertical_half(o.shape, o.size, d.body(OBJ + g["obj"]).xmat.reshape(3, 3))
            ok_z = floor <= local[2] and bottom <= rim - 0.005
            err = float(np.linalg.norm(out_xy)) + (0.0 if ok_z else abs(local[2] - rim))
            where = "inside" if err == 0 and ok_z else f"outside (off by {err * 100:.1f} cm)"
            return res(err == 0 and ok_z and not_held, f"{g['obj']} centroid is {where} {g['target']}'s cavity{held_note}")

        if typ in SIDES:
            axis, sign = SIDES[typ]
            gap = sign * float(p[axis] - pos(g["target"])[axis])
            return res(gap >= SIDE_MARGIN and not_held,
                       f"{g['obj']} is {gap * 100:+.1f} cm {typ.replace('_', ' ')} {g['target']} "
                       f"(need >= {SIDE_MARGIN * 100:.0f}){held_note}")

        if typ == "upright":
            tilt = float(np.degrees(np.arccos(np.clip(d.body(OBJ + g["obj"]).xmat.reshape(3, 3)[2, 2], -1, 1))))
            return res(tilt <= UPRIGHT_MAX_TILT, f"{g['obj']} is tilted {tilt:.0f} deg (max {UPRIGHT_MAX_TILT:.0f})")

        if typ == "on":
            t, q = objs[g["target"]], pos(g["target"])
            dz = p[2] - (q[2] + t.size[2] / 2 + o.size[2] / 2)
            dxy = float(np.linalg.norm(p[:2] - q[:2]))
            # Supported on top = right height and centred within half the LARGER footprint.
            xy_tol = max(0.015, 0.5 * max(min(t.size[0], t.size[1]), min(o.size[0], o.size[1])))
            return res(abs(dz) < ON_Z_TOL and dxy < xy_tol and not_held,
                       f"{g['obj']} is {dxy * 100:.1f} cm off-center from {g['target']} (tol {xy_tol * 100:.1f}), "
                       f"height error {dz * 100:+.1f} cm (tol +-{ON_Z_TOL * 100:.1f}){held_note}")

        if typ == "at":
            if g.get("xyz") is None:
                return res(False, "'at' goal needs xyz")
            dist = float(np.linalg.norm(p - np.array(g["xyz"])))
            return res(dist < AT_TOL and not_held, f"{g['obj']} is {dist * 100:.1f} cm from target (tol {AT_TOL * 100:.0f})")

        bottom = p[2] - o.size[2] / 2  # lifted
        return res(bottom > LIFT_MIN, f"{g['obj']} bottom is {bottom * 100:.1f} cm above floor (need >{LIFT_MIN * 100:.0f})")
