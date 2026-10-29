"""Close the loop: learn a policy from the generated demonstrations and test it in the simulator.

Role in the pipeline: `sim.dataset.generate` turns a solved plan into N successful demos (LeRobot
v2.1 parquet). This module trains a small behaviour-cloning policy on them (pure numpy: an MLP or
a nearest-neighbour lookup, no torch) and rolls it out closed-loop in the SAME simulator, with the
same randomization (`runner.perturb`, `runner.STRESS`) and the same code-checked goals
(`metrics.check_goal`) as the scripted plan. The comparison scripted-vs-learned is the "policy
report card".

Observation = arm joints + fingers + pose of every loose object + elapsed time. Action = 7 joint
targets (predicted as a change from the current joints) + gripper command. The gripper command
drives the same grasp assist as the scripted controller: `try_grasp` once the command closes and
the fingers have stopped, `release` when it opens. The learned policy sees true object poses
(the scripted plan in the stress test sees them with +-0.5 cm sensing error).
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import pyarrow.parquet as pq

from spec2sim.schemas import Build, Goal
from spec2sim.sim.controller import GRIP_CLOSED, GRIP_OPEN, IKError
from spec2sim.sim.dataset import FPS
from spec2sim.sim.metrics import check_goal
from spec2sim.sim.runner import SETTLE_END, STRESS, _execute, _Sim, perturb
from spec2sim.sim.scene import OBJ_PREFIX, SceneError

T_SCALE = 30.0  # seconds, to keep the time feature near 0..1
CLOSE_BELOW, OPEN_ABOVE = 0.35, 0.5  # gripper command thresholds (1 = open)


# ---------------------------------------------------------------- data
def load_dataset(path: Path) -> dict:
    """Read <path>/data/chunk-000/*.parquet into per-episode arrays."""
    eps = []
    for f in sorted((Path(path) / "data/chunk-000").glob("*.parquet")):
        t = pq.read_table(f).to_pydict()
        eps.append({"state": np.array(t["observation.state"], np.float32),
                    "action": np.array(t["action"], np.float32),
                    "env": np.array(t["observation.environment_state"], np.float32).reshape(len(t["action"]), -1),
                    "t": np.array(t["timestamp"], np.float32)})
    if not eps:
        raise FileNotFoundError(f"no episodes under {path}/data/chunk-000")
    return {"episodes": eps, "fps": json.loads((Path(path) / "meta/info.json").read_text())["fps"]}


def make_fk(sim: _Sim):
    """states [N, 9] -> (tool-point positions [N, 3], tool yaw [N]) by forward kinematics (scratch MjData)."""
    m, sd, tcp = sim.m, mujoco.MjData(sim.m), sim.m.site("tcp").id
    R_home = sim.ctl.R_home

    def fk(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pos, yaw = [], []
        for s in np.atleast_2d(states):
            sd.qpos[sim.scene.arm_qpos] = s[:7]
            mujoco.mj_kinematics(m, sd)
            pos.append(sd.site_xpos[tcp].copy())
            Rz = sd.site_xmat[tcp].reshape(3, 3) @ R_home.T
            yaw.append(np.arctan2(Rz[1, 0], Rz[0, 0]))
        return np.array(pos, np.float32), np.array(yaw, np.float32)
    return fk


def features(state, env, t, tcp, yaw) -> np.ndarray:
    """Compact, task-relevant observation: time, tool point + yaw, finger gap, object positions and each
    object relative to the tool point. (Raw joints and quaternions made the policy brittle off the demos.)"""
    pos = [env[:, i:i + 3] for i in range(0, env.shape[1], 7)]
    cols = [(np.asarray(t, np.float32) / T_SCALE).reshape(-1, 1), tcp, np.cos(yaw)[:, None], np.sin(yaw)[:, None],
            state[:, 7:9].sum(1, keepdims=True)] + pos + [p - tcp for p in pos]
    return np.concatenate(cols, axis=1)


def targets(fk, state: np.ndarray, action: np.ndarray, delta: bool) -> np.ndarray:
    """Action in tool space: [tool-point target xyz, cos/sin yaw, gripper]. delta: xyz relative to the current tool point."""
    p, yaw = fk(action[:, :7])
    if delta:
        p = p - fk(state)[0]
    return np.concatenate([p, np.cos(yaw)[:, None], np.sin(yaw)[:, None], action[:, 7:8]], axis=1)


# ---------------------------------------------------------------- models
class Policy:
    """Behaviour-cloning policy. kind: 'mlp' (2 hidden layers, Adam) or 'knn' (nearest demo frames)."""

    def __init__(self, kind: str = "mlp", hidden: int = 128, epochs: int = 1000, lr: float = 2e-3,
                 k: int = 5, seed: int = 0, fk=None, delta: bool = False, noise: float = 0.1):
        self.kind, self.hidden, self.epochs, self.lr, self.k, self.seed = kind, hidden, epochs, lr, k, seed
        self.delta, self.noise = delta, noise
        self.fk = fk  # forward kinematics from make_fk (required)
        self.max_steps = 0
        self.episodes_trained = 0

    def _x(self, state, env, t) -> np.ndarray:
        tcp, yaw = self.fk(state)
        return features(state, env, t, tcp, yaw)

    def fit(self, data: dict) -> Policy:
        eps = data["episodes"]
        X = np.concatenate([self._x(e["state"], e["env"], e["t"]) for e in eps])
        Y = np.concatenate([targets(self.fk, e["state"], e["action"], self.delta) for e in eps])
        self.episodes_trained = len(eps)
        self.max_steps = max(len(e["t"]) for e in eps)
        self.xm, self.xs = X.mean(0), np.maximum(X.std(0), 0.03)  # floor: don't amplify constant dims
        self.ym, self.ys = Y.mean(0), Y.std(0) + 1e-4
        Xn, Yn = (X - self.xm) / self.xs, (Y - self.ym) / self.ys
        if self.kind == "knn":
            self.X, self.Y = Xn, Yn
            return self
        rng = np.random.default_rng(self.seed)
        sizes = [Xn.shape[1], self.hidden, self.hidden, Yn.shape[1]]
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in itertools.pairwise(sizes)]
        self.b = [np.zeros(b) for b in sizes[1:]]
        params = self.W + self.b
        m1 = [np.zeros_like(p) for p in params]
        m2 = [np.zeros_like(p) for p in params]
        step, bs = 0, 256
        for ep in range(self.epochs):
            lr = self.lr * (0.5 * (1 + np.cos(np.pi * ep / self.epochs)) * 0.95 + 0.05)
            idx = rng.permutation(len(Xn))
            for i in range(0, len(idx), bs):
                j = idx[i:i + bs]
                x = Xn[j] + rng.normal(0, self.noise, (len(j), Xn.shape[1]))  # small input noise: robustness
                acts = [x]
                for li in range(3):
                    z = acts[-1] @ self.W[li] + self.b[li]
                    acts.append(np.maximum(z, 0) if li < 2 else z)
                g = 2 * (acts[-1] - Yn[j]) / len(j)
                grads_w, grads_b = [None] * 3, [None] * 3
                for li in (2, 1, 0):
                    grads_w[li], grads_b[li] = acts[li].T @ g, g.sum(0)
                    if li:
                        g = (g @ self.W[li].T) * (acts[li] > 0)
                step += 1
                for p, gr, a, b in zip(params, grads_w + grads_b, m1, m2, strict=True):
                    a *= 0.9
                    a += 0.1 * gr
                    b *= 0.999
                    b += 0.001 * gr * gr
                    p -= lr * (a / (1 - 0.9 ** step)) / (np.sqrt(b / (1 - 0.999 ** step)) + 1e-8)
        return self

    def act(self, state: np.ndarray, env: np.ndarray, t: float) -> tuple[np.ndarray, float, float]:
        """-> (tool-point target xyz, tool yaw, gripper command 0..1)."""
        x = (self._x(state[None], env[None], np.array([t])) - self.xm) / self.xs
        if self.kind == "knn":
            d = ((self.X - x) ** 2).sum(1)
            nn = np.argpartition(d, self.k)[: self.k]
            w = 1 / (d[nn] + 1e-3)
            yn = (self.Y[nn] * w[:, None]).sum(0, keepdims=True) / w.sum()
        else:
            h = x
            for li in range(3):
                h = h @ self.W[li] + self.b[li]
                if li < 2:
                    h = np.maximum(h, 0)
            yn = h
        y = (yn * self.ys + self.ym)[0]
        pos = y[:3] + (self.fk(state[None])[0][0] if self.delta else 0.0)
        return pos, float(np.arctan2(y[4], y[3])), float(y[5])


class Ensemble:
    """Average of several MLPs trained from different seeds: less variance, steadier grasps."""

    def __init__(self, members: int = 5, **kw):
        self.kw, self.members = kw, members

    def fit(self, data: dict) -> Ensemble:
        seed = self.kw.pop("seed", 0)
        self.models = [Policy(seed=seed + i, **self.kw).fit(data) for i in range(self.members)]
        self.max_steps, self.episodes_trained = self.models[0].max_steps, self.models[0].episodes_trained
        return self

    def act(self, state, env, t):
        outs = [m.act(state, env, t) for m in self.models]
        pos = np.mean([o[0] for o in outs], 0)
        yaw = float(np.arctan2(np.mean([np.sin(o[1]) for o in outs]), np.mean([np.cos(o[1]) for o in outs])))
        return pos, yaw, float(np.mean([o[2] for o in outs]))


# ---------------------------------------------------------------- rollout
def _obs(sim: _Sim, fingers, objs) -> tuple[np.ndarray, np.ndarray]:
    d = sim.d
    state = np.concatenate([d.qpos[sim.scene.arm_qpos], d.qpos[fingers]]).astype(np.float32)
    env = np.concatenate([np.concatenate([d.body(OBJ_PREFIX + n).xpos, d.body(OBJ_PREFIX + n).xquat])
                          for n in objs]).astype(np.float32) if objs else np.zeros(0, np.float32)
    return state, env


def rollout(policy: Policy, sim: _Sim, steps: int | None = None) -> str | None:
    """Run the policy closed-loop at the dataset FPS; returns an error string or None."""
    ctl, m = sim.ctl, sim.m
    fingers = [m.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)]
    objs = [n for n, o in sim.scene.objects.items() if not o.fixed]  # same order as the dataset
    ticks = max(1, round(1 / FPS / 0.01))
    steps = steps or int(policy.max_steps * 1.3)
    try:
        t0 = sim.d.time  # demos start at t=0 with the initial settle included
        closing = 0
        for _ in range(steps):
            state, env = _obs(sim, fingers, objs)
            pos, yaw, g = policy.act(state, env, sim.d.time - t0)
            try:  # same IK as the scripted controller; an unreachable target keeps the last joint target
                ctl.q_target = ctl.ik(pos, yaw, ctl.q_target, iters=60)
            except IKError:
                pass
            g = float(np.clip(g, 0, 1))
            ctl.grip = GRIP_CLOSED + (GRIP_OPEN - GRIP_CLOSED) * g
            for _ in range(ticks):
                sim.tick()
            if g > OPEN_ABOVE and ctl.held is not None:
                ctl.release()
                closing = 0
            elif g < CLOSE_BELOW and ctl.held is None:
                closing += 1
                if closing >= 2 and ctl.finger_gap() >= 0 and ctl.finger_gap() < 0.078:  # fingers settled
                    ctl.try_grasp()
            else:
                closing = 0
        sim.hold(SETTLE_END)
    except IKError as e:
        return str(e)
    return None


# ---------------------------------------------------------------- evaluation
def evaluate(build: Build, goals: list[Goal], trials: int = 16, seed: int = 0, policy: Policy | None = None,
             physics: dict | None = None, sense_error: float = STRESS["sense_error"]) -> dict:
    """Score the scripted plan (policy=None) or a learned policy on the same seeded perturbed variants
    as `runner.stress_test` (same draws for a given seed)."""
    rng = np.random.default_rng(seed)
    rows, per_goal, attempts = [], [0] * len(goals), 0
    while len(rows) < trials and attempts < trials * 6:
        attempts += 1
        b, tweak, change = perturb(build, rng)
        change.pop("_friction_x")
        try:
            sim = _Sim(b, None, tweak=tweak, physics=physics)
        except SceneError:
            continue
        errs = {n: np.append(rng.uniform(-sense_error, sense_error, 2), 0.0) for n in change}
        if policy is None:
            sim.ctl.sense_error.update(errs)
            err = _execute(sim, b)
        else:
            err = rollout(policy, sim)
        res = [check_goal(sim.scene, sim.d, g, sim.ctl.held, sim.peaks) for g in goals]
        for i, r in enumerate(res):
            per_goal[i] += r.ok
        rows.append({"ok": err is None and all(r.ok for r in res), "error": err,
                     "failed": [r.detail for r in res if not r.ok]})
    return {"passed": sum(r["ok"] for r in rows), "trials": len(rows),
            "per_goal": [{"goal": g.model_dump(exclude_none=True), "passed": per_goal[i]} for i, g in enumerate(goals)],
            "failures": [(r["error"] or "; ".join(r["failed"]))[:160] for r in rows if not r["ok"]][:8]}


def train(data_dir: Path, members: int = 1, **kw) -> tuple[Policy | Ensemble, float]:
    """members > 1: an Ensemble of that many MLPs (kind 'mlp' only)."""
    t0 = time.time()
    model = Ensemble(members, **kw) if members > 1 else Policy(**kw)
    return model.fit(load_dataset(data_dir)), round(time.time() - t0, 1)
