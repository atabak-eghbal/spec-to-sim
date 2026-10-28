"""Offscreen rendering: MP4 for the UI, sampled frames + one labeled contact sheet for the Critic.

The Critic gets a single grid image (8 timestamped frames) instead of 8 separate images -
one image request is much faster on the VL endpoint (see docs/research/03-models.md).
Video frames stream straight to the MP4 encoder; only small thumbnails stay in memory, so a 30 s
multi-object task fits comfortably in a small CPU container.

Every frame's body poses are also recorded for the browser 3D replay (webviz.py). With
SPEC2SIM_VIDEO=0 (the CPU server) no MP4 is made and only ~3 frames per second are rendered,
for the Critic's contact sheet: the browser draws the replay instead.
"""
from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from spec2sim.sim.scene import RENDER_H, RENDER_W
from spec2sim.sim.webviz import Trajectory

FPS = 30
# Shadows cost ~100 ms/frame in software GL (OSMesa/llvmpipe), 4x the rest of the frame. The CPU
# container sets SPEC2SIM_SHADOWS=0; GPU/desktop rendering keeps them.
SHADOWS = os.environ.get("SPEC2SIM_SHADOWS", "1") == "1"
VIDEO = os.environ.get("SPEC2SIM_VIDEO", "1") == "1"
THUMB_EVERY = 3 if VIDEO else 10  # frames between thumbnails for the Critic (10 fps / 3 fps)


class Recorder:
    def __init__(self, model: mujoco.MjModel, out_dir: Path):
        self.r = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = (0.38, 0.0, 0.3)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.5, 145, -17
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.video = out_dir / "run.mp4" if VIDEO else None
        self.writer = iio.get_writer(self.video, fps=FPS, codec="libx264", macro_block_size=8,
                                     output_params=["-movflags", "+faststart"]) if VIDEO else None
        self.traj = Trajectory(model)
        self.n = 0
        self.thumbs: list[Image.Image] = []
        self.times: list[float] = []

    def capture(self, data: mujoco.MjData) -> None:
        self.traj.capture(data)
        keep = self.n % THUMB_EVERY == 0
        self.n += 1
        if not (VIDEO or keep):
            return
        self.r.update_scene(data, self.cam)
        # Floor reflections cost ~half the frame time on CPU (OSMesa) and add little; shadows stay
        # because they show how high an object is above a surface.
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = SHADOWS
        frame = self.r.render()
        if self.writer:
            self.writer.append_data(frame)
        if keep:
            self.thumbs.append(Image.fromarray(frame).resize((RENDER_W // 2, RENDER_H // 2)))
            self.times.append(data.time)

    def save(self, events: list[str], goals: list[dict], n_samples: int = 8) -> tuple[str | None, list[str], str, str]:
        """Returns (video path or None, frame paths, contact sheet path, viz.json path)."""
        if self.writer:
            self.writer.close()
        out_dir = self.out_dir
        idx = np.linspace(0, len(self.thumbs) - 1, n_samples).round().astype(int)
        paths, tiles = [], []
        for k, i in enumerate(idx):
            p = out_dir / f"frame_{k}.jpg"
            self.thumbs[i].save(p, quality=85)
            paths.append(str(p))
            tile = self.thumbs[i].copy()
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, 118, 26), fill=(0, 0, 0))
            draw.text((6, 6), f"#{k + 1}  t={self.times[i]:.1f}s", fill=(255, 255, 255))
            tiles.append(tile)
        cols, rows = 4, (n_samples + 3) // 4
        sheet = Image.new("RGB", (cols * RENDER_W // 2, rows * RENDER_H // 2))
        for k, t in enumerate(tiles):
            sheet.paste(t, ((k % cols) * RENDER_W // 2, (k // cols) * RENDER_H // 2))
        sheet_path = out_dir / "contact_sheet.jpg"
        sheet.save(sheet_path, quality=85)
        viz = self.traj.save(out_dir, FPS, events, goals)
        return (str(self.video) if self.video else None), paths, str(sheet_path), viz

    def close(self) -> None:
        self.r.close()
        if self.writer and not self.writer.closed:
            self.writer.close()
