"""Scene assembly: Menagerie robot (fixed, hand-checked XML) + LLM-described objects (data only).

The Builder never writes XML. It returns `SceneObject` records; this module turns them into
MuJoCo bodies through MjSpec, adds a tool-center-point site to the gripper, and gives every
movable object an inactive weld to the hand (the grasp-assist fallback used by the controller).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Headless GL must be chosen before mujoco is imported. macOS only supports CGL.
if sys.platform == "darwin":
    os.environ["MUJOCO_GL"] = "cgl"
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from spec2sim.schemas import SceneObject

ROOT = Path(__file__).resolve().parents[3]
PANDA_SCENE = ROOT / "third_party/menagerie/franka_emika_panda/scene.xml"
OBJ_PREFIX = "obj/"
TCP_OFFSET = 0.1034  # Franka flange -> fingertip center, meters
RENDER_W, RENDER_H = 848, 480

# Workspace the Builder is told about; objects outside it are rejected before simulation.
WORKSPACE = {"x": (0.25, 0.75), "y": (-0.4, 0.4), "max_size": 0.25, "max_mass": 3.0}


@dataclass
class Scene:
    model: mujoco.MjModel
    objects: dict[str, SceneObject]  # by plain name (no prefix)
    weld_ids: dict[str, int]  # object name -> equality id
    arm_qpos: np.ndarray  # qpos addresses of joint1..7
    arm_dof: np.ndarray  # dof addresses of joint1..7


class SceneError(ValueError):
    """The described scene is invalid (bad sizes, out of reach, overlapping, failed compile)."""


def _geom_size(o: SceneObject) -> list[float]:
    sx, sy, sz = o.size
    if o.shape == "box":
        return [sx / 2, sy / 2, sz / 2]
    if o.shape == "cylinder":
        return [max(sx, sy) / 2, sz / 2, 0]
    return [max(o.size) / 2, 0, 0]


def validate_objects(objects: list[SceneObject]) -> None:
    names = [o.name for o in objects]
    if len(set(names)) != len(names):
        raise SceneError(f"duplicate object names: {names}")
    for o in objects:
        if min(o.size) <= 0.005 or max(o.size) > WORKSPACE["max_size"]:
            raise SceneError(f"{o.name}: size {o.size} m outside 0.005-{WORKSPACE['max_size']} m")
        if not 0 < o.mass <= WORKSPACE["max_mass"]:
            raise SceneError(f"{o.name}: mass {o.mass} kg outside 0-{WORKSPACE['max_mass']} kg")
        x, y, z = o.pos
        if not (WORKSPACE["x"][0] <= x <= WORKSPACE["x"][1] and WORKSPACE["y"][0] <= y <= WORKSPACE["y"][1]):
            raise SceneError(f"{o.name}: pos {o.pos} outside reachable area x{WORKSPACE['x']} y{WORKSPACE['y']}")
        if z < o.size[2] / 2 - 0.01:
            raise SceneError(f"{o.name}: z={z} puts it below the floor (half height {o.size[2] / 2})")
    # Footprint overlap check (axis-aligned, conservative) for objects at the same height band.
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            dx, dy = abs(a.pos[0] - b.pos[0]), abs(a.pos[1] - b.pos[1])
            dz = abs(a.pos[2] - b.pos[2])
            if (dx < (a.size[0] + b.size[0]) / 2 - 0.002 and dy < (a.size[1] + b.size[1]) / 2 - 0.002
                    and dz < (a.size[2] + b.size[2]) / 2 - 0.002):
                raise SceneError(f"{a.name} and {b.name} overlap at their start positions")


def build_scene(objects: list[SceneObject]) -> Scene:
    validate_objects(objects)
    spec = mujoco.MjSpec.from_file(str(PANDA_SCENE))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = RENDER_W, RENDER_H
    spec.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.005, 0, 0])

    shapes = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
              "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}
    for o in objects:
        quat = [np.cos(o.yaw / 2), 0, 0, np.sin(o.yaw / 2)]
        body = spec.worldbody.add_body(name=OBJ_PREFIX + o.name, pos=list(o.pos), quat=quat)
        if not o.fixed:
            body.add_freejoint()
        body.add_geom(type=shapes[o.shape], size=_geom_size(o), rgba=list(o.rgba), mass=o.mass,
                      friction=[1.5, 0.05, 0.001], condim=4, solref=[0.01, 1])
        if not o.fixed:
            spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                              name1="hand", name2=OBJ_PREFIX + o.name, active=False,
                              name=f"grip/{o.name}")
    try:
        model = spec.compile()
    except Exception as e:  # MuJoCo compile errors are plain exceptions
        raise SceneError(f"MuJoCo failed to compile the scene: {e}") from e

    weld_ids = {o.name: model.equality(f"grip/{o.name}").id for o in objects if not o.fixed}
    joints = [model.joint(f"joint{i}") for i in range(1, 8)]
    return Scene(
        model=model,
        objects={o.name: o for o in objects},
        weld_ids=weld_ids,
        arm_qpos=np.array([j.qposadr[0] for j in joints]),
        arm_dof=np.array([j.dofadr[0] for j in joints]),
    )
