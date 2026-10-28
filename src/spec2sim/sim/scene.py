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

from spec2sim.schemas import REACH_R, REACH_X, REACH_Y, SceneObject, reachable_xy

ROOT = Path(__file__).resolve().parents[3]
PANDA_SCENE = ROOT / "third_party/menagerie/franka_emika_panda/scene.xml"
OBJ_PREFIX = "obj/"
TCP_OFFSET = 0.1034  # Franka flange -> fingertip center, meters
RENDER_W, RENDER_H = 848, 480

# Workspace the Builder is told about; objects outside it are rejected before simulation.
WORKSPACE = {"x": REACH_X, "y": REACH_Y, "max_size": 0.25, "max_mass": 3.0}
BIN_WALL = 0.008  # wall/floor thickness of open-top bins, m
BIN_MAX_SIZE = 0.30  # bins wider than this crowd the 50 cm-deep workspace
BIN_MAX_HEIGHT = 0.15  # taller walls block the arm's approach


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


def validate_objects(objects: list[SceneObject], allowed: set[str] | None = None) -> None:
    """Raise SceneError listing EVERY problem at once, so one repair can fix them all.

    `allowed`: the task's object names; anything else (e.g. an invented table) is rejected.
    """
    problems: list[str] = []
    names = [o.name for o in objects]
    if len(set(names)) != len(names):
        problems.append(f"duplicate object names: {names}")
    if allowed is not None:
        extra = [n for n in names if n not in allowed]
        missing = sorted(allowed - set(names))
        if extra:
            problems.append(f"objects {extra} are not in the task; use only {sorted(allowed)} (the floor is the table)")
        if missing:
            problems.append(f"task objects {missing} are missing from the scene")
    lo_x, hi_x = WORKSPACE["x"]
    lo_y, hi_y = WORKSPACE["y"]
    for o in objects:
        x, y, z = o.pos
        max_size = BIN_MAX_SIZE if o.shape == "bin" else WORKSPACE["max_size"]
        if min(o.size) <= 0.005 or max(o.size) > max_size:
            problems.append(f"{o.name}: size {o.size} m outside 0.005-{max_size} m")
        if not 0 < o.mass <= WORKSPACE["max_mass"]:
            problems.append(f"{o.name}: mass {o.mass} kg outside 0-{WORKSPACE['max_mass']} kg")
        if z < o.size[2] / 2 - 0.01:
            problems.append(f"{o.name}: z={z} puts it below the floor (half height {o.size[2] / 2})")
        if o.shape == "bin":
            if min(o.size[:2]) < 0.08 or o.size[2] < 0.02:
                problems.append(f"{o.name}: bin {o.size} m is too small to hold anything (min 8 cm wide, 2 cm tall)")
            if o.size[2] > BIN_MAX_HEIGHT:
                problems.append(f"{o.name}: bin is {o.size[2] * 100:.0f} cm tall; use <= {BIN_MAX_HEIGHT * 100:.0f} cm "
                                "so the arm can reach over its walls")
            hx, hy = o.size[0] / 2, o.size[1] / 2
            if not (lo_x <= x - hx and x + hx <= hi_x and lo_y <= y - hy and y + hy <= hi_y):
                problems.append(f"{o.name}: bin at {o.pos} sticks out of the reachable area x{WORKSPACE['x']} "
                                f"y{WORKSPACE['y']}; its whole footprint must be inside: center x between "
                                f"{lo_x + hx:.2f} and {hi_x - hx:.2f}, center y between {lo_y + hy:.2f} and {hi_y - hy:.2f}")
        elif not reachable_xy(x, y):
            problems.append(f"{o.name}: pos {o.pos} outside reachable area x{WORKSPACE['x']} y{WORKSPACE['y']}, "
                            f"within {REACH_R} m of the base")
    # Footprint overlap check (axis-aligned, conservative) for objects at the same height band.
    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            dx, dy = abs(a.pos[0] - b.pos[0]), abs(a.pos[1] - b.pos[1])
            dz = abs(a.pos[2] - b.pos[2])
            if (dx < (a.size[0] + b.size[0]) / 2 - 0.002 and dy < (a.size[1] + b.size[1]) / 2 - 0.002
                    and dz < (a.size[2] + b.size[2]) / 2 - 0.002):
                problems.append(f"{a.name} and {b.name} overlap at their start positions")
    if problems:
        raise SceneError("; ".join(problems) if len(problems) == 1
                         else f"{len(problems)} scene problems: " + " | ".join(problems))


def _add_bin(body: mujoco.MjsBody, o: SceneObject) -> None:
    """Open-top container: a floor plate plus four walls, all on one fixed body."""
    hx, hy, hz = (s / 2 for s in o.size)
    t = BIN_WALL / 2
    parts = [([0, 0, -hz + t], [hx, hy, t]),  # floor
             ([hx - t, 0, 0], [t, hy, hz]), ([-hx + t, 0, 0], [t, hy, hz]),
             ([0, hy - t, 0], [hx - 2 * t, t, hz]), ([0, -hy + t, 0], [hx - 2 * t, t, hz])]
    for pos, size in parts:
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, rgba=list(o.rgba),
                      mass=o.mass / 5, friction=[1.0, 0.05, 0.001], condim=4, solref=[0.01, 1])


_SHAPES = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
           "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}


def _add_objects(spec: mujoco.MjSpec, objects: list[SceneObject], grasp_welds: bool) -> None:
    for o in objects:
        quat = [np.cos(o.yaw / 2), 0, 0, np.sin(o.yaw / 2)]
        body = spec.worldbody.add_body(name=OBJ_PREFIX + o.name, pos=list(o.pos), quat=quat)
        if not o.fixed:
            body.add_freejoint()
        if o.shape == "bin":
            _add_bin(body, o)
            continue
        body.add_geom(type=_SHAPES[o.shape], size=_geom_size(o), rgba=list(o.rgba), mass=o.mass,
                      friction=[1.5, 0.05, 0.001], condim=4, solref=[0.01, 1])
        if grasp_welds and not o.fixed:
            spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                              name1="hand", name2=OBJ_PREFIX + o.name, active=False,
                              name=f"grip/{o.name}")


def objects_mjcf(objects: list[SceneObject]) -> str:
    """The generated objects as a standalone MJCF file (no robot), for export. Load it next to any
    robot, e.g. MuJoCo Menagerie's franka_emika_panda/scene.xml, via <include> or MjSpec.attach."""
    objects = [o.model_copy(update={"fixed": True}) if o.shape == "bin" else o for o in objects]
    spec = mujoco.MjSpec()
    spec.modelname = "spec2sim_objects"
    _add_objects(spec, objects, grasp_welds=False)
    return spec.to_xml()


def build_scene(objects: list[SceneObject], allowed: set[str] | None = None) -> Scene:
    validate_objects(objects, allowed)
    # Bins are furniture: never simulated as free bodies, never grasped.
    objects = [o.model_copy(update={"fixed": True}) if o.shape == "bin" else o for o in objects]
    spec = mujoco.MjSpec.from_file(str(PANDA_SCENE))
    spec.visual.global_.offwidth, spec.visual.global_.offheight = RENDER_W, RENDER_H
    spec.body("hand").add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.005, 0, 0])

    _add_objects(spec, objects, grasp_welds=True)
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
