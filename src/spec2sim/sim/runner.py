"""Runner: execute a Build (scene + motion plan) in MuJoCo and return a RunResult.

Deterministic, no LLM calls. Produces the MP4, sampled frames, a contact sheet for the Critic,
a list of notable events, and per-goal pass/fail from `metrics.py`.
"""
from __future__ import annotations

import time
from pathlib import Path

import mujoco
import numpy as np

from spec2sim.schemas import Action, Build, Goal, RunResult
from spec2sim.sim.controller import GRIP_CLOSED, GRIP_OPEN, HOME_Q, Controller, IKError
from spec2sim.sim.metrics import check_goal
from spec2sim.sim.render import FPS, Recorder
from spec2sim.sim.scene import BIN_WALL, OBJ_PREFIX, SceneError, build_scene

CTRL_DT = 0.01  # seconds between IK/control updates
SETTLE_START, SETTLE_END = 0.3, 0.8
HAND_BELOW_TCP = 0.038  # lowest point of the Panda hand body above the tool point, m
GRIP_SETTLE = 0.15  # s after a close before the grasp is judged
MAX_SIM_TIME = 32.0  # room for 4-5 pick-and-place moves


def _smoothstep(x: float) -> float:
    return x * x * (3 - 2 * x)


class _Sim:
    def __init__(self, build: Build, out_dir: Path | None, allowed: set[str] | None = None,
                 tweak=None, physics: dict | None = None):
        """out_dir=None: physics only, nothing rendered or recorded (used by the stress test).
        tweak(model): optional hook to perturb the compiled model (e.g. friction)."""
        self.scene = build_scene(build.objects, allowed, physics)
        self.m = self.scene.model
        self.d = mujoco.MjData(self.m)
        self.d.qpos[self.scene.arm_qpos] = HOME_Q
        self.d.qpos[self.m.joint("finger_joint1").qposadr[0]] = 0.04
        self.d.qpos[self.m.joint("finger_joint2").qposadr[0]] = 0.04
        if tweak:
            tweak(self.m)
        mujoco.mj_forward(self.m, self.d)
        self.ctl = Controller(self.scene, self.d)
        self.rec = Recorder(self.m, out_dir) if out_dir is not None else None
        self.events: list[str] = []
        self.n_sub = round(CTRL_DT / self.m.opt.timestep)
        self.next_frame = 0.0
        # Highest point each loose object has RESTED at (not held, nearly still): 'toppled' is
        # judged against it, so "stack it, then knock it down" works and carrying doesn't count.
        self.movable = [n for n, o in self.scene.objects.items() if not o.fixed]
        self.peaks = {n: float(self.d.body(OBJ_PREFIX + n).xpos[2]) for n in self.movable}
        self.next_peak = 0.0
        self.on_tick = None  # optional per-step hook (dataset recording)

    def tick(self) -> None:
        self.ctl.apply()
        mujoco.mj_step(self.m, self.d, nstep=self.n_sub)
        if self.on_tick:
            self.on_tick()
        if self.rec and self.d.time >= self.next_frame:
            self.rec.capture(self.d)
            self.next_frame += 1.0 / FPS
        if self.d.time >= self.next_peak:
            self.next_peak += 0.1
            for n in self.movable:
                if n == self.ctl.held:
                    continue
                b = self.d.body(OBJ_PREFIX + n)
                if np.linalg.norm(b.cvel[3:]) < 0.02:
                    self.peaks[n] = max(self.peaks[n], float(b.xpos[2]))
        if self.d.time > MAX_SIM_TIME:
            raise IKError(f"plan exceeds {MAX_SIM_TIME:.0f} s of sim time")

    def hold(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds / CTRL_DT))):
            self.tick()

    def event(self, msg: str) -> None:
        self.events.append(f"t={self.d.time:.1f}s {msg}")

    # ---------- skills: waypoints computed in code from real sizes and current poses ----------
    def _top(self, name: str) -> float:
        o = self.scene.objects[name]
        return float(self.d.body(OBJ_PREFIX + name).xpos[2]) + o.size[2] / 2

    def _clear_z(self, held_h: float, skip: str | None = None) -> float:
        """Tool-point height that carries an object of height held_h 6 cm above everything else."""
        tops = [self._top(n) for n in self.scene.objects if n != skip and self.scene.objects[n].shape != "drawer"]
        return max(tops + [0.0]) + 0.06 + held_h / 2

    def _mv(self, xyz, yaw: float, dur: float) -> None:
        self.do(Action(op="move_to", xyz=tuple(float(v) for v in xyz), yaw=yaw, duration=dur))

    def skill(self, a: Action) -> None:
        ctl, objs = self.ctl, self.scene.objects
        for ref in (a.object, a.target):
            if ref is not None and ref not in objs:
                raise IKError(f"{a.op}: unknown object '{ref}'")
        if a.op == "pick":
            if a.object is None:
                raise IKError("pick needs 'object'")
            o = objs[a.object]
            c = ctl.object_pos(a.object)
            self._mv(c + [0, 0, max(0.12, o.size[2] / 2 + 0.08)], a.yaw, 1.0)
            g = ctl.object_pos(a.object)
            # Tall objects are gripped high: the hand body sits HAND_BELOW_TCP above the fingertips.
            g[2] = max(g[2], g[2] + o.size[2] / 2 - HAND_BELOW_TCP - 0.008)
            self._mv(g, a.yaw, 0.6)
            self.do(Action(op="close", duration=0.4))
            if ctl.held != a.object:
                return  # the close event already says what happened
            p = ctl.tcp_pos()
            self._mv([p[0], p[1], max(p[2] + 0.1, self._clear_z(o.size[2], a.object))], a.yaw, 0.8)
            return
        if a.op in ("place_on", "drop_into"):
            held = ctl.held
            if held is None or a.target is None:
                raise IKError(f"{a.op} needs a held object and a 'target' (pick something first)")
            h = objs[held].size[2]
            t = objs[a.target]
            tc = ctl.object_pos(a.target)
            # How the object actually sits in the hand: aim the OBJECT, not the tool point.
            grip = ctl.tcp_pos() - self.d.body(OBJ_PREFIX + held).xpos
            xy = tc[:2] + np.array(a.offset[:2])
            if a.op == "place_on":
                rest = self._top(a.target) + h / 2 + 0.003
            else:
                if t.shape != "bin":
                    raise IKError(f"drop_into needs a bin target, '{a.target}' is a {t.shape}")
                floor = float(tc[2]) - t.size[2] / 2 + BIN_WALL
                rim = float(tc[2]) + t.size[2] / 2
                # Lower as deep as the hand can go without its body touching the rim.
                rest = max(floor + h / 2 + 0.005, rim + 0.012 - HAND_BELOW_TCP - grip[2])
            carry = max(self._clear_z(h, held), rest + grip[2] + 0.08)
            p = ctl.tcp_pos()
            self._mv([p[0], p[1], carry], a.yaw, 0.5)
            self._mv([xy[0] + grip[0], xy[1] + grip[1], carry], a.yaw, 1.0)
            self._mv([xy[0] + grip[0], xy[1] + grip[1], rest + grip[2]], a.yaw, 0.8)
            self.do(Action(op="open", duration=0.3))
            q = ctl.tcp_pos()
            self._mv([q[0], q[1], q[2] + 0.12], a.yaw, 0.6)
            return
        if a.op == "push":
            if a.object is None or a.direction is None:
                raise IKError("push needs 'object' and 'direction'")
            o = objs[a.object]
            c = ctl.object_pos(a.object)
            ax = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}[a.direction]
            d = np.array([ax[0], ax[1], 0.0])
            half = (o.size[0] if ax[0] else o.size[1]) / 2
            bottom = float(c[2]) - o.size[2] / 2
            z = bottom + 0.8 * o.size[2]  # high on the object: tips it rather than sliding it
            start = c - d * (half + 0.06)
            start[2] = z
            yaw = 1.5708 if ax[0] else 0.0  # fingers across the push direction: the closed fingertips lead
            self.do(Action(op="close", duration=0.3))
            self._mv(start + [0, 0, 0.12], yaw, 0.8)
            self._mv(start, yaw, 0.6)
            end = c + d * (half + 0.12)
            end[2] = z
            self._mv(end, yaw, 0.45)
            self._mv(end + [0, 0, 0.12], yaw, 0.5)
            return
        raise IKError(f"unknown skill '{a.op}'")

    def do(self, a: Action) -> None:
        ctl = self.ctl
        if a.op in ("pick", "place_on", "drop_into", "push"):
            self.skill(a)
            return
        if a.op == "wait":
            self.hold(a.duration)
        elif a.op in ("open", "close"):
            if a.op == "open":
                released = ctl.release()
                if released and self.scene.objects[released].shape == "drawer":
                    out = max(0.0, self.d.qpos[self.scene.drawer_qpos[released]])
                    self.event(f"released {released} handle: drawer is {out * 100:.1f} cm open")
                elif released:
                    self.event(f"released {released}: {self._support_gap(released)}")
            start, end = ctl.grip, GRIP_OPEN if a.op == "open" else GRIP_CLOSED
            steps = max(1, round(a.duration / CTRL_DT))
            for k in range(steps):
                ctl.grip = start + (end - start) * _smoothstep((k + 1) / steps)
                self.tick()
            if a.op == "close":
                self.hold(GRIP_SETTLE)  # let the fingers finish closing before judging the grasp
                got, msg = ctl.try_grasp()
                if got and self.scene.objects[got].shape == "drawer":
                    msg = f"grasped {got} handle"
                self.event(msg)
        else:  # move_to
            if a.object is not None:
                target = ctl.object_pos(a.object) + np.array(a.offset)
            elif a.xyz is not None:
                target = np.array(a.xyz) + np.array(a.offset)
            else:
                raise IKError("move_to needs either 'object' or 'xyz'")
            if target[2] < 0.005:
                self.event(f"target z={target[2]:.3f} is below the floor; clamped to 0.005")
                target[2] = 0.005
            start = ctl.tcp_pos()
            # Interpolate yaw from current orientation (tracked in ctl) to the new one.
            yaw0 = getattr(ctl, "yaw", 0.0)
            steps = max(1, round(a.duration / CTRL_DT))
            for k in range(steps):
                s = _smoothstep((k + 1) / steps)
                p = start + s * (target - start)
                ctl.q_target = ctl.ik(p, yaw0 + s * (a.yaw - yaw0), ctl.q_target, iters=60 if k < steps - 1 else 200)
                self.tick()
            ctl.yaw = a.yaw
            err = np.linalg.norm(ctl.tcp_pos() - target)
            if err > 0.035:
                self.event(f"arm blocked: tcp stopped {err * 100:.1f} cm short of the move_to target (collision?)")

    def _support_gap(self, name: str) -> str:
        """Describe how far the released object's bottom is above whatever is below it."""
        o = self.scene.objects[name]
        p = self.d.body(OBJ_PREFIX + name).xpos
        bottom = p[2] - o.size[2] / 2
        below, top = "the floor", 0.0
        for other, q in self.scene.objects.items():
            if other == name:
                continue
            c = self.d.body(OBJ_PREFIX + other).xpos
            # A bin's support surface is its floor, not its rim.
            surface = c[2] - q.size[2] / 2 + BIN_WALL if q.shape == "bin" else c[2] + q.size[2] / 2
            if q.shape == "drawer":  # the drawer's own floor, wherever it has slid to
                db = self.d.body(f"{OBJ_PREFIX}{other}/drawer").xpos
                surface = c[2] - q.size[2] / 2 + 2 * BIN_WALL
                if np.all(np.abs(p[:2] - db[:2]) < np.array(q.size[:2]) / 2) and surface <= bottom + 0.02 and surface > top:
                    below, top = f"{other}'s drawer floor", surface
                continue
            if np.all(np.abs(p[:2] - c[:2]) < np.array(q.size[:2]) / 2) and surface <= bottom + 0.02:
                t = surface
                if t > top:
                    below, top = other, t
        return (f"center at {np.round(p, 3).tolist()}, bottom {(bottom - top) * 100:.1f} cm above {below}")

    def tilt_events(self) -> None:
        for name, o in self.scene.objects.items():
            if o.fixed:
                continue
            b = self.d.body(OBJ_PREFIX + name)
            R = b.xmat.reshape(3, 3)
            tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
            # A box that rolled onto another face still rests flat; only report it as tipped
            # when no face points up (it leans on an edge or another object).
            face_tilt = np.degrees(np.arccos(np.clip(np.max(np.abs(R[2, :])), -1, 1)))
            if o.shape == "box" and tilt > 25:
                self.event(f"{name} tipped over ({face_tilt:.0f} deg off flat)" if face_tilt > 15
                           else f"{name} rolled onto another face (still flat)")
            elif o.shape == "cylinder" and tilt > 25:
                self.event(f"{name} tipped over ({tilt:.0f} deg)")
            if b.xpos[2] < -0.05 or np.linalg.norm(b.xpos[:2]) > 1.5:
                self.event(f"{name} fell out of the workspace")


def run(build: Build, goals: list[Goal], out_dir: Path, allowed: set[str] | None = None,
        physics: dict | None = None) -> RunResult:
    """`allowed`: the task's object names (the Builder may not add or drop objects).
    `physics`: object name -> (friction, restitution) from the web."""
    t0 = time.time()
    try:
        sim = _Sim(build, out_dir, allowed, physics=physics)
    except SceneError as e:
        return RunResult(success=False, goals=[], events=[], error=str(e))

    error = None
    try:
        sim.hold(SETTLE_START)
        # A placement goal that already holds before the robot moves means the scene was set up
        # to fake the result (a repair once "solved" a task by starting the orange in the basket).
        cheats = [g for g in goals if g.type in ("on", "inside")
                  and check_goal(sim.scene, sim.d, g, None).ok]
        if cheats:
            g = cheats[0]
            error = (f"{g.obj} already starts {g.type} {g.target}; objects must start away from "
                     "their goal so the robot does the task")
            if sim.rec:
                sim.rec.close()
            return RunResult(success=False, goals=[], events=[], error=error)
        for i, a in enumerate(build.plan.actions):
            try:
                sim.do(a)
            except IKError as e:
                error = f"action {i + 1} ({a.op}): {e}"
                sim.event(f"stopped: {error}")
                break
        sim.hold(SETTLE_END)
    except IKError as e:  # sim-time cap
        error = str(e)
    sim.tilt_events()

    try:
        results = [check_goal(sim.scene, sim.d, g, sim.ctl.held, sim.peaks) for g in goals]
        video, frames, sheet, viz = sim.rec.save(sim.events, [r.model_dump() for r in results])
    finally:
        sim.rec.close()  # release the GL context even if saving fails (long-lived web server)
    return RunResult(
        success=error is None and bool(results) and all(r.ok for r in results),
        goals=results, events=sim.events, error=error,
        video_path=video, frame_paths=frames, contact_sheet_path=sheet, viz_path=viz,
        duration_s=round(time.time() - t0, 2),
    )


# ---------- stress test ----------
def _execute(sim: _Sim, build: Build) -> str | None:
    """Run the plan in an existing sim. Returns an error string or None."""
    try:
        sim.hold(SETTLE_START)
        for i, a in enumerate(build.plan.actions):
            try:
                sim.do(a)
            except IKError as e:
                return f"action {i + 1} ({a.op}): {e}"
        sim.hold(SETTLE_END)
    except IKError as e:
        return str(e)
    return None


STRESS = {"pos_jitter": 0.04, "yaw_jitter": 0.52, "mass_scale": 0.2, "friction_scale": 0.3, "sense_error": 0.005}


def perturb(build: Build, rng, pos_jitter: float = STRESS["pos_jitter"], yaw_jitter: float = STRESS["yaw_jitter"],
            mass_scale: float = STRESS["mass_scale"], friction_scale: float = STRESS["friction_scale"]):
    """One random variant of a build: loose objects nudged, turned, heavier/lighter; one friction
    scale for all objects. Furniture stays put. Returns (build, model tweak, description)."""
    objs, change = [], {}
    for o in build.objects:
        if o.fixed or o.shape in ("bin", "drawer"):
            objs.append(o)
            continue
        dx, dy = rng.uniform(-pos_jitter, pos_jitter, 2)
        dyaw = rng.uniform(-yaw_jitter, yaw_jitter) if o.shape != "sphere" else 0.0
        k = rng.uniform(1 - mass_scale, 1 + mass_scale)
        objs.append(o.model_copy(update={"pos": (o.pos[0] + dx, o.pos[1] + dy, o.pos[2]),
                                         "yaw": o.yaw + dyaw, "mass": o.mass * k}))
        change[o.name] = {"dx_cm": round(dx * 100, 1), "dy_cm": round(dy * 100, 1),
                          "yaw_deg": round(float(np.degrees(dyaw)), 1), "mass_x": round(k, 2)}
    fk = rng.uniform(1 - friction_scale, 1 + friction_scale)

    def tweak(m, fk=fk):
        for g in range(m.ngeom):
            if m.body(m.geom_bodyid[g]).name.startswith(OBJ_PREFIX):
                m.geom_friction[g, 0] *= fk

    change["_friction_x"] = round(fk, 2)
    return build.model_copy(update={"objects": objs}), tweak, change


def stress_test(build: Build, goals: list[Goal], trials: int = 16, seed: int = 0,
                pos_jitter: float = STRESS["pos_jitter"], yaw_jitter: float = STRESS["yaw_jitter"],
                mass_scale: float = STRESS["mass_scale"], friction_scale: float = STRESS["friction_scale"],
                sense_error: float = STRESS["sense_error"], physics: dict | None = None) -> dict:
    """Re-run a plan under random perturbations, physics only (no rendering, no model calls).

    Each trial: every loose object starts up to 4 cm away and turned up to 30 degrees, weighs
    +-20 %, all objects get one +-30 % friction scale, and the robot sees each object up to 0.5 cm
    from where it really is (like a camera's error), so a grasp with no margin fails. Plan steps
    are relative to where the robot sees objects, so a good plan adapts to the nudges.
    Variants where the nudges make objects overlap are redrawn and counted in 'redrawn'.
    """
    rng = np.random.default_rng(seed)
    rows, goal_pass = [], {i: 0 for i in range(len(goals))}
    t0 = time.time()
    attempts = redrawn = 0
    while len(rows) < trials and attempts < trials * 6:
        attempts += 1
        b, tweak, change = perturb(build, rng, pos_jitter, yaw_jitter, mass_scale, friction_scale)
        fk = change.pop("_friction_x")
        try:
            sim = _Sim(b, None, tweak=tweak, physics=physics)
        except SceneError:
            redrawn += 1
            continue  # nudges made two objects overlap or left the workspace: draw again
        for name in change:
            err = np.append(rng.uniform(-sense_error, sense_error, 2), 0.0)
            sim.ctl.sense_error[name] = err
            change[name]["seen_off_cm"] = round(float(np.linalg.norm(err)) * 100, 1)
        err = _execute(sim, b)
        res = [check_goal(sim.scene, sim.d, g, sim.ctl.held, sim.peaks) for g in goals]
        ok = err is None and all(r.ok for r in res)
        for i, r in enumerate(res):
            goal_pass[i] += r.ok
        rows.append({"ok": ok, "error": err, "friction_x": fk, "objects": change,
                     "failed": [r.detail for r in res if not r.ok]})
    n = len(rows)
    return {"trials": n, "passed": sum(r["ok"] for r in rows), "redrawn": redrawn, "seed": seed,
            "per_goal": [{"goal": g.model_dump(exclude_none=True), "passed": goal_pass[i]} for i, g in enumerate(goals)],
            "perturbation": {"position_cm": pos_jitter * 100, "yaw_deg": round(float(np.degrees(yaw_jitter)), 1),
                             "mass_pct": mass_scale * 100, "friction_pct": friction_scale * 100,
                             "sensing_cm": sense_error * 100},
            "rows": rows, "seconds": round(time.time() - t0, 1)}


# ---------- before the run: a picture of the scene, and a fast physics-only score ----------
def snapshot(build: Build, path: Path, allowed: set[str] | None = None, physics: dict | None = None) -> str | None:
    """The scene as built, before the robot moves: two views side by side (front and top) as one JPG,
    for the vision model's scene check. None if the scene is invalid."""
    from PIL import Image

    from spec2sim.sim.render import RENDER_H, RENDER_W
    try:
        sim = _Sim(build, None, allowed, physics=physics)
    except SceneError:
        return None
    sim.hold(SETTLE_START)
    r = mujoco.Renderer(sim.m, height=RENDER_H, width=RENDER_W)
    views = []
    for lookat, dist, az, el in (((0.45, 0.0, 0.1), 1.25, 150, -25), ((0.5, 0.0, 0.0), 1.15, 180, -88)):
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, el
        r.update_scene(sim.d, cam)
        views.append(Image.fromarray(r.render()))
    r.close()
    sheet = Image.new("RGB", (RENDER_W * 2, RENDER_H))
    for i, v in enumerate(views):
        sheet.paste(v, (i * RENDER_W, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85)
    return str(path)


ROBUST_TRIALS = 4  # perturbed re-runs per repair candidate when choosing the best one


def quick_score(build: Build, goals: list[Goal], allowed: set[str] | None = None,
                physics: dict | None = None, robust_trials: int = ROBUST_TRIALS) -> dict:
    """Run a plan physics-only (no rendering, no model calls) and score it, to pick the best of
    several repair candidates: success first, then how many of `robust_trials` perturbed re-runs
    also pass (a plan that only works in the exact nominal scene is fragile), then goals passed,
    then total miss distance. Any crash scores as a failed candidate instead of killing the run."""
    try:
        sim = _Sim(build, None, allowed, physics=physics)
        err = _execute(sim, build)
        res = [check_goal(sim.scene, sim.d, g, sim.ctl.held, sim.peaks) for g in goals]
    except SceneError as e:
        return {"ok": False, "passed": -1, "robust": 0, "miss_m": 9.0, "error": str(e)[:200]}
    except Exception as e:  # noqa: BLE001 - one broken candidate must not stop the repair
        return {"ok": False, "passed": -1, "robust": 0, "miss_m": 9.0, "error": f"crashed: {e!r}"[:200]}
    ok = err is None and all(r.ok for r in res)
    robust = 0
    if ok and robust_trials:
        try:
            robust = stress_test(build, goals, trials=robust_trials, seed=7, physics=physics)["passed"]
        except Exception:  # noqa: BLE001
            robust = 0
    return {"ok": ok, "passed": sum(r.ok for r in res), "robust": robust, "robust_trials": robust_trials,
            "miss_m": round(sum(r.error_m for r in res), 4), "error": (err or "")[:200],
            "failed": [r.detail for r in res if not r.ok][:3]}
