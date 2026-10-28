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
from spec2sim.sim.scene import OBJ_PREFIX, SceneError, build_scene

CTRL_DT = 0.01  # seconds between IK/control updates
SETTLE_START, SETTLE_END = 0.3, 0.8
MAX_SIM_TIME = 16.0


def _smoothstep(x: float) -> float:
    return x * x * (3 - 2 * x)


class _Sim:
    def __init__(self, build: Build):
        self.scene = build_scene(build.objects)
        self.m = self.scene.model
        self.d = mujoco.MjData(self.m)
        self.d.qpos[self.scene.arm_qpos] = HOME_Q
        self.d.qpos[self.m.joint("finger_joint1").qposadr[0]] = 0.04
        self.d.qpos[self.m.joint("finger_joint2").qposadr[0]] = 0.04
        mujoco.mj_forward(self.m, self.d)
        self.ctl = Controller(self.scene, self.d)
        self.rec = Recorder(self.m)
        self.events: list[str] = []
        self.n_sub = round(CTRL_DT / self.m.opt.timestep)
        self.next_frame = 0.0

    def tick(self) -> None:
        self.ctl.apply()
        mujoco.mj_step(self.m, self.d, nstep=self.n_sub)
        if self.d.time >= self.next_frame:
            self.rec.capture(self.d)
            self.next_frame += 1.0 / FPS
        if self.d.time > MAX_SIM_TIME:
            raise IKError(f"plan exceeds {MAX_SIM_TIME:.0f} s of sim time")

    def hold(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds / CTRL_DT))):
            self.tick()

    def event(self, msg: str) -> None:
        self.events.append(f"t={self.d.time:.1f}s {msg}")

    def do(self, a: Action) -> None:
        ctl = self.ctl
        if a.op == "wait":
            self.hold(a.duration)
        elif a.op in ("open", "close"):
            if a.op == "open":
                released = ctl.release()
                if released:
                    self.event(f"released {released}: {self._support_gap(released)}")
            start, end = ctl.grip, GRIP_OPEN if a.op == "open" else GRIP_CLOSED
            steps = max(1, round(a.duration / CTRL_DT))
            for k in range(steps):
                ctl.grip = start + (end - start) * _smoothstep((k + 1) / steps)
                self.tick()
            if a.op == "close":
                _, msg = ctl.try_grasp()
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
            if err > 0.02:
                self.event(f"arm lagging: tcp {err * 100:.1f} cm from move_to target (contact or collision?)")

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
            if np.all(np.abs(p[:2] - c[:2]) < np.array(q.size[:2]) / 2) and c[2] + q.size[2] / 2 <= bottom + 0.02:
                t = c[2] + q.size[2] / 2
                if t > top:
                    below, top = other, t
        return (f"center at {np.round(p, 3).tolist()}, bottom {(bottom - top) * 100:.1f} cm above {below}")

    def tilt_events(self) -> None:
        for name, o in self.scene.objects.items():
            if o.fixed:
                continue
            b = self.d.body(OBJ_PREFIX + name)
            up = b.xmat.reshape(3, 3)[:, 2]
            tilt = np.degrees(np.arccos(np.clip(up[2], -1, 1)))
            if tilt > 25 and o.shape != "sphere":
                self.event(f"{name} tipped over ({tilt:.0f} deg)")
            if b.xpos[2] < -0.05 or np.linalg.norm(b.xpos[:2]) > 1.5:
                self.event(f"{name} fell out of the workspace")


def run(build: Build, goals: list[Goal], out_dir: Path) -> RunResult:
    t0 = time.time()
    try:
        sim = _Sim(build)
    except SceneError as e:
        return RunResult(success=False, goals=[], events=[], error=str(e))

    error = None
    try:
        sim.hold(SETTLE_START)
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

    results = [check_goal(sim.scene, sim.d, g, sim.ctl.held) for g in goals]
    video, frames, sheet = sim.rec.save(out_dir)
    sim.rec.close()
    return RunResult(
        success=error is None and bool(results) and all(r.ok for r in results),
        goals=results, events=sim.events, error=error,
        video_path=video, frame_paths=frames, contact_sheet_path=sheet,
        duration_s=round(time.time() - t0, 2),
    )
