"""Data contracts passed between pipeline steps (Planner -> Grounder -> Builder -> Runner -> Critic).

LLMs only ever produce these validated structures - never code or robot XML. All units are SI:
meters, kilograms, seconds. World frame: robot base at origin, +x forward, +z up, floor at z=0.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Vec3 = tuple[float, float, float]
RGBA = tuple[float, float, float, float]


# ---------- 1. Planner ----------
class ObjectSpec(BaseModel):
    name: str = Field(description="snake_case id, e.g. red_cube")
    description: str = Field(description="what the real object is, e.g. 'wooden toy block'")
    shape: Literal["box", "cylinder", "sphere"]
    color: str = Field(description="plain color name, e.g. red")


class Goal(BaseModel):
    """A success predicate checked in code at the end of the run."""

    type: Literal["on", "at", "lifted"] = Field(
        description="on: obj rests on top of target; at: obj near xyz; lifted: obj held above floor"
    )
    obj: str
    target: str | None = Field(None, description="object name, for 'on'")
    xyz: Vec3 | None = Field(None, description="world position, for 'at'")


class TaskSpec(BaseModel):
    summary: str = Field(min_length=5, description="one-line restatement of the task (required, non-empty)")
    robot: Literal["franka_panda"] = "franka_panda"
    objects: list[ObjectSpec]
    goals: list[Goal]
    search_queries: list[str] = Field(
        description="1-3 web searches that find real dimensions/mass of the objects", max_length=3
    )


# ---------- 2. Grounder ----------
class ObjectFact(BaseModel):
    name: str = Field(description="matches ObjectSpec.name")
    size: Vec3 = Field(description="full extents in meters (x, y, z); cylinder/sphere: (d, d, h)")
    mass: float = Field(description="kg")
    source_url: str | None = Field(None, description="URL the numbers came from, if any")
    note: str = Field("", description="short justification, e.g. 'standard block is 5 cm'")


class GroundedFacts(BaseModel):
    facts: list[ObjectFact]


class Source(BaseModel):
    title: str
    url: str


# ---------- 3. Builder ----------
class SceneObject(BaseModel):
    name: str
    shape: Literal["box", "cylinder", "sphere"]
    size: Vec3 = Field(description="full extents in meters (x, y, z); cylinder/sphere: (d, d, h)")
    mass: float
    rgba: RGBA
    pos: Vec3 = Field(description="center position in world; z = height/2 to rest on the floor")
    yaw: float = Field(0.0, description="rotation about z, radians")
    fixed: bool = Field(False, description="true for static things like a table or tray")


class Action(BaseModel):
    """One step of the motion plan, executed by our fixed controller.

    move_to: move the gripper's tool point to `xyz`, or to an object's *current* center plus
    `offset` when `object` is given. Gripper always points down; `yaw` rotates it about z.
    """

    op: Literal["move_to", "open", "close", "wait"]
    object: str | None = None
    offset: Vec3 = (0.0, 0.0, 0.0)
    xyz: Vec3 | None = None
    yaw: float = 0.0
    duration: float = Field(1.0, description="seconds for this step", ge=0.1, le=3.0)


class MotionPlan(BaseModel):
    actions: list[Action]


class Build(BaseModel):
    """Builder output: the scene plus the plan to execute in it."""

    objects: list[SceneObject]
    plan: MotionPlan
    rationale: str = Field(min_length=5, description="one or two sentences on key choices (required)")


# ---------- 4. Runner ----------
class GoalResult(BaseModel):
    goal: Goal
    ok: bool
    error_m: float = Field(description="distance from goal pose, meters")
    detail: str


class RunResult(BaseModel):
    success: bool
    goals: list[GoalResult]
    events: list[str] = Field(description="notable sim events, e.g. 'grasped red_cube at t=1.4s'")
    error: str | None = Field(None, description="scene failed to load, IK failed, etc.")
    video_path: str | None = None
    frame_paths: list[str] = []
    contact_sheet_path: str | None = None
    duration_s: float = 0.0


# ---------- 5. Critic ----------
class Critique(BaseModel):
    # Field order matters: the model writes evidence before deciding the cause.
    evidence: str = Field(description="quote the failed goal details and the sim events that show what went wrong")
    category: Literal[
        "missed_grasp", "dropped_object", "wrong_placement", "collision", "unreachable",
        "unstable_scene", "scene_error", "other",
    ]
    what_happened: str = Field(description="what the frames show, in plain words")
    root_cause: str
    suggested_fix: str = Field(description="concrete change to scene or plan, with numbers")
