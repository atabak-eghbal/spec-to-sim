"""Training data from one sentence: successful demonstrations in LeRobot v2.1 layout.

NVIDIA's synthetic-data pipelines (Isaac Lab Mimic, the GR00T blueprint) multiply a few demos
into many varied ones and keep the successful ones. We do the same from a generated plan: each
episode re-runs the plan with the objects nudged, turned, heavier or lighter and more or less
slippery (the stress-test perturbations). Successful episodes are saved; failures are dropped.

Layout (LeRobot dataset format v2.1, as used for GR00T post-training):
    meta/info.json, meta/tasks.jsonl, meta/episodes.jsonl, meta/episodes_stats.jsonl
    data/chunk-000/episode_000000.parquet      one row per frame
    videos/chunk-000/observation.images.front/episode_000000.mp4
Per frame: observation.state (7 arm joints + 2 fingers, rad / m), action (7 joint targets + gripper
command, 1 = open), observation.environment_state (position + quaternion of every movable object).
"""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from spec2sim.schemas import Build, Goal
from spec2sim.sim.controller import GRIP_OPEN
from spec2sim.sim.metrics import check_goal
from spec2sim.sim.render import SHADOWS
from spec2sim.sim.runner import STRESS, _execute, _Sim, perturb
from spec2sim.sim.scene import OBJ_PREFIX, SceneError

FPS = 10
IMG_W, IMG_H = 320, 240
CODEBASE = "v2.1"


class _Collector:
    """Hooked into the sim tick: samples state, action, object poses and a camera frame at FPS."""

    def __init__(self, sim: _Sim, images: bool):
        self.sim, self.m, self.d = sim, sim.m, sim.d
        self.objs = [n for n, o in sim.scene.objects.items() if not o.fixed]
        self.fingers = [self.m.joint(f"finger_joint{i}").qposadr[0] for i in (1, 2)]
        self.next_t, self.rows, self.frames = None, [], []
        self.r = None
        if images:
            self.r = mujoco.Renderer(self.m, height=IMG_H, width=IMG_W)
            self.cam = mujoco.MjvCamera()
            self.cam.lookat[:] = (0.45, 0.0, 0.12)
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.05, 155, -32

    def __call__(self):
        d = self.d
        if self.next_t is None:  # first call: start the 1/FPS grid here
            self.next_t = d.time
        if d.time + 1e-6 < self.next_t:
            return
        self.next_t += 1.0 / FPS
        ctl = self.sim.ctl
        state = np.concatenate([d.qpos[self.sim.scene.arm_qpos], d.qpos[self.fingers]]).astype(np.float32)
        action = np.concatenate([ctl.q_target, [ctl.grip / GRIP_OPEN]]).astype(np.float32)
        env = np.concatenate([np.concatenate([d.body(OBJ_PREFIX + n).xpos, d.body(OBJ_PREFIX + n).xquat])
                              for n in self.objs]).astype(np.float32) if self.objs else np.zeros(0, np.float32)
        self.rows.append((round(float(d.time), 4), state, action, env))
        if self.r:
            self.r.update_scene(d, self.cam)
            self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = SHADOWS
            self.r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            self.frames.append(self.r.render().copy())

    def close(self):
        if self.r:
            self.r.close()


def _stats(x: np.ndarray) -> dict:
    x = x.reshape(len(x), -1).astype(np.float64)
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(), "count": [len(x)]}


def generate(build: Build, goals: list[Goal], prompt: str, out_dir: Path, episodes: int = 20,
             seed: int = 1, images: bool = True, progress=None, physics: dict | None = None,
             provenance: dict | None = None) -> dict:
    """provenance: the real-world facts behind the scene (sizes, materials, robot limits, with
    their source URLs); saved in meta/info.json so the data stays traceable."""
    """Run up to 3x `episodes` perturbed variants; save the first `episodes` successes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    kept, tried, invalid, t0 = [], 0, 0, time.time()
    names = None
    # tried = valid variations simulated; invalid = draws where the jitter made objects overlap at the
    # start (skipped, not failures).
    while len(kept) < episodes and tried < episodes * 3 and tried + invalid < episodes * 10:
        b, tweak, _ = perturb(build, rng)
        try:
            sim = _Sim(b, None, tweak=tweak, physics=physics)
        except SceneError:
            invalid += 1
            continue
        tried += 1
        col = _Collector(sim, images)
        sim.on_tick = col
        err = _execute(sim, b)
        col.close()
        ok = err is None and all(check_goal(sim.scene, sim.d, g, sim.ctl.held, sim.peaks).ok for g in goals)
        if progress:
            progress(len(kept) + ok, episodes, tried)
        if not ok or not col.rows:
            continue
        names = col.objs
        kept.append(col)

    root = out_dir
    (root / "meta").mkdir(exist_ok=True)
    data_dir, vid_dir = root / "data/chunk-000", root / "videos/chunk-000/observation.images.front"
    data_dir.mkdir(parents=True, exist_ok=True)
    if images:
        vid_dir.mkdir(parents=True, exist_ok=True)
    index, ep_lines, stat_lines = 0, [], []
    for ei, col in enumerate(kept):
        ts = (np.arange(len(col.rows)) / FPS).astype(np.float32)  # LeRobot expects exact 1/fps spacing
        st, ac, en = (np.stack([r[k] for r in col.rows]) for k in (1, 2, 3))
        n = len(ts)
        table = pa.table({
            "observation.state": pa.array(st.tolist(), type=pa.list_(pa.float32(), st.shape[1])),
            "action": pa.array(ac.tolist(), type=pa.list_(pa.float32(), ac.shape[1])),
            "observation.environment_state": pa.array(en.tolist(), type=pa.list_(pa.float32(), en.shape[1])),
            "timestamp": pa.array(ts, pa.float32()),
            "frame_index": pa.array(np.arange(n), pa.int64()),
            "episode_index": pa.array(np.full(n, ei), pa.int64()),
            "index": pa.array(np.arange(index, index + n), pa.int64()),
            "task_index": pa.array(np.zeros(n, np.int64), pa.int64()),
        })
        pq.write_table(table, data_dir / f"episode_{ei:06d}.parquet")
        stats = {"observation.state": _stats(st), "action": _stats(ac),
                 "observation.environment_state": _stats(en), "timestamp": _stats(ts),
                 "frame_index": _stats(np.arange(n)), "episode_index": _stats(np.full(n, ei)),
                 "index": _stats(np.arange(index, index + n)), "task_index": _stats(np.zeros(n))}
        if images and col.frames:
            iio.mimwrite(vid_dir / f"episode_{ei:06d}.mp4", col.frames, fps=FPS, codec="libx264", macro_block_size=8)
            px = np.stack(col.frames[:: max(1, n // 20)]).astype(np.float64) / 255.0  # sampled frames
            stats["observation.images.front"] = {k: [[[v]] for v in vals] for k, vals in {
                "min": px.min((0, 1, 2)), "max": px.max((0, 1, 2)), "mean": px.mean((0, 1, 2)),
                "std": px.std((0, 1, 2))}.items()} | {"count": [len(px)]}
        ep_lines.append({"episode_index": ei, "tasks": [prompt], "length": n})
        stat_lines.append({"episode_index": ei, "stats": stats})
        index += n

    env_names = [f"{o}.{k}" for o in (names or []) for k in ("x", "y", "z", "qw", "qx", "qy", "qz")]
    features = {
        "observation.state": {"dtype": "float32", "shape": [9], "names": [f"joint{i}" for i in range(1, 8)] + ["finger_left", "finger_right"]},
        "action": {"dtype": "float32", "shape": [8], "names": [f"joint{i}_target" for i in range(1, 8)] + ["gripper_open"]},
        "observation.environment_state": {"dtype": "float32", "shape": [len(env_names)], "names": env_names},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    if images:
        features["observation.images.front"] = {
            "dtype": "video", "shape": [IMG_H, IMG_W, 3], "names": ["height", "width", "channels"],
            "info": {"video.height": IMG_H, "video.width": IMG_W, "video.codec": "h264", "video.pix_fmt": "yuv420p",
                     "video.is_depth_map": False, "video.fps": FPS, "video.channels": 3, "has_audio": False}}
    info = {"codebase_version": CODEBASE, "robot_type": "franka_panda", "total_episodes": len(kept),
            "total_frames": index, "total_tasks": 1, "total_videos": len(kept) if images else 0, "total_chunks": 1,
            "chunks_size": 1000, "fps": FPS, "splits": {"train": f"0:{len(kept)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4" if images else None,
            "features": features,
            "spec2sim": {"task": prompt, "facts": provenance or {},
                         "generator": "Spec-to-Sim: re-runs of a verified plan with randomized starts, masses and friction; successes only"}}
    (root / "meta/info.json").write_text(json.dumps(info, indent=2))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": prompt}) + "\n")
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in ep_lines))
    (root / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(x) + "\n" for x in stat_lines))
    (root / "README.md").write_text(
        f"# Spec-to-Sim generated dataset\n\nTask: \"{prompt}\"\n\n{len(kept)} successful episodes out of {tried} valid "
        f"perturbed variants (objects nudged up to {STRESS['pos_jitter'] * 100:g} cm and "
        f"{np.degrees(STRESS['yaw_jitter']):.0f} deg, mass +-20%, friction +-30%), {FPS} fps, "
        f"Franka Panda (MuJoCo Menagerie) in MuJoCo.\nLayout: LeRobot dataset format {CODEBASE} (parquet + mp4).\n"
        "Grasping uses Spec-to-Sim's grasp assist: once both finger pads touch an object and it holds the fingers "
        "apart, a weld keeps it in the hand (no slip). Motions are scripted top-down waypoints, not human demos.\n")
    zpath = root.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(root.parent))
    return {"episodes": len(kept), "tried": tried, "skipped_invalid": invalid, "frames": index,
            "seconds": round(time.time() - t0, 1),
            "zip": str(zpath), "fps": FPS, "images": images}
