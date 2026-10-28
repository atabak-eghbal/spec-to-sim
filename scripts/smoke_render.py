"""Smoke test: load the Franka Panda from MuJoCo Menagerie, move it, render a short MP4 headless.

Proves the Runner's core path (load Menagerie scene + objects, step physics, offscreen render,
encode video) works on this machine before any LLM is involved.

Run: uv run python scripts/smoke_render.py   ->  runs/smoke/panda_smoke.mp4
"""
import os
import sys
import time
from pathlib import Path

# Pick a headless GL backend before mujoco is imported. EGL/OSMesa are Linux-only;
# macOS uses CGL. An explicit MUJOCO_GL in the shell always wins.
if sys.platform == "darwin":
    os.environ["MUJOCO_GL"] = "cgl"
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v3 as iio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "third_party/menagerie/franka_emika_panda/scene.xml"
OUT = ROOT / "runs/smoke/panda_smoke.mp4"

# Two cubes on the floor, as a standalone objects fragment - the same shape the Builder emits.
OBJECTS = """
<mujoco>
  <worldbody>
    <body name="red_cube" pos="0.5 0.1 0.025"><freejoint/>
      <geom type="box" size="0.025 0.025 0.025" rgba="0.85 0.15 0.15 1" mass="0.1"/></body>
    <body name="blue_cube" pos="0.5 -0.1 0.025"><freejoint/>
      <geom type="box" size="0.025 0.025 0.025" rgba="0.15 0.3 0.85 1" mass="0.1"/></body>
  </worldbody>
</mujoco>
"""

SECONDS, FPS, W, H = 3.0, 30, 848, 480  # width divisible by 16 for H.264


def main() -> None:
    if not SCENE.exists():
        sys.exit("Menagerie missing - run scripts/fetch_menagerie.sh first")
    spec = mujoco.MjSpec.from_file(str(SCENE))  # Menagerie robot; mesh paths resolve correctly
    frag = mujoco.MjSpec.from_string(OBJECTS)  # objects only, never robot XML
    for body in frag.worldbody.bodies:
        spec.worldbody.add_frame().attach_body(body, "obj/", "")  # namespace LLM objects
    spec.visual.global_.offwidth, spec.visual.global_.offheight = W, H
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    home = data.ctrl.copy()

    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.3, 0.0, 0.3)
    cam.distance, cam.azimuth, cam.elevation = 1.8, 135, -25

    frames, t0 = [], time.time()
    steps_per_frame = round(1 / (FPS * model.opt.timestep))
    for i in range(int(SECONDS * FPS)):
        t = i / FPS
        data.ctrl[:] = home
        data.ctrl[0] = 0.6 * np.sin(2 * np.pi * t / SECONDS)   # sweep base joint
        data.ctrl[3] = home[3] + 0.4 * np.sin(4 * np.pi * t / SECONDS)  # bob elbow
        data.ctrl[7] = 255 if (t % 1.5) < 0.75 else 0  # open/close gripper
        mujoco.mj_step(model, data, nstep=steps_per_frame)
        renderer.update_scene(data, cam)
        frames.append(renderer.render())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(OUT, np.stack(frames), fps=FPS, codec="libx264")
    print(f"MUJOCO_GL={os.environ['MUJOCO_GL']}  mujoco {mujoco.__version__}")
    print(f"{len(frames)} frames {W}x{H} in {time.time() - t0:.1f}s -> {OUT.relative_to(ROOT)}")
    for name in ("obj/red_cube", "obj/blue_cube"):
        print(name, "pos =", np.round(data.body(name).xpos, 3), "(expect x=0.5, z~0.025)")
    iio.imwrite(OUT.with_suffix(".png"), frames[len(frames) // 2])


if __name__ == "__main__":
    main()
