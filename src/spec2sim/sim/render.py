"""Offscreen rendering: MP4 for the UI, sampled frames + one labeled contact sheet for the Critic.

The Critic gets a single grid image (8 timestamped frames) instead of 8 separate images -
one image request is much faster on the VL endpoint (see docs/research/03-models.md).
"""
from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from spec2sim.sim.scene import RENDER_H, RENDER_W

FPS = 30


class Recorder:
    def __init__(self, model: mujoco.MjModel):
        self.r = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = (0.38, 0.0, 0.3)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.5, 145, -17
        self.frames: list[np.ndarray] = []
        self.times: list[float] = []

    def capture(self, data: mujoco.MjData) -> None:
        self.r.update_scene(data, self.cam)
        self.frames.append(self.r.render().copy())
        self.times.append(data.time)

    def save(self, out_dir: Path, n_samples: int = 8) -> tuple[str, list[str], str]:
        out_dir.mkdir(parents=True, exist_ok=True)
        video = out_dir / "run.mp4"
        iio.imwrite(video, np.stack(self.frames), fps=FPS, codec="libx264",
                    output_params=["-movflags", "+faststart"])
        idx = np.linspace(0, len(self.frames) - 1, n_samples).round().astype(int)
        paths, tiles = [], []
        for k, i in enumerate(idx):
            p = out_dir / f"frame_{k}.jpg"
            Image.fromarray(self.frames[i]).save(p, quality=85)
            paths.append(str(p))
            tile = Image.fromarray(self.frames[i]).resize((RENDER_W // 2, RENDER_H // 2))
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
        return str(video), paths, str(sheet_path)

    def close(self) -> None:
        self.r.close()
