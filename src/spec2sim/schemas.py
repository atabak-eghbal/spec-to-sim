"""Data contracts passed between pipeline steps (Planner -> Grounder -> Builder -> Runner -> Critic).

LLMs only ever produce these validated structures - never code or robot XML. All units are SI:
meters, kilograms, seconds. World frame: robot base at origin, +x forward, +z up, floor at z=0.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Vec3 = tuple[float, float, float]

# Where the Franka can reach with the gripper pointing down, mapped by solving IK on a grid
# (x 0.30-0.70 m in front of the base, |y| <= 0.40 m, at most 0.78 m from the base).
REACH_X, REACH_Y, REACH_R = (0.30, 0.70), (-0.40, 0.40), 0.78


def reachable_xy(x: float, y: float) -> bool:
    return REACH_X[0] <= x <= REACH_X[1] and REACH_Y[0] <= y <= REACH_Y[1] and (x * x + y * y) ** 0.5 <= REACH_R
RGBA = tuple[float, float, float, float]


# ---------- 1. Planner ----------
class ObjectSpec(BaseModel):
    name: str = Field(description="snake_case id, e.g. red_cube")
    description: str = Field(description="what the real object is, e.g. 'wooden toy block'")
    shape: Literal["box", "cylinder", "sphere", "bin", "drawer"] = Field(
        description="bin = open-top container (tray, basket, storage bin) other objects go into; "
                    "drawer = small cabinet with one sliding drawer that has a handle")
    color: str = Field(description="plain color name, e.g. red")
    material: Literal["rubber", "wood", "plastic", "metal", "glass", "ceramic", "cardboard", "paper",
                      "fabric", "foam", "fruit"] = Field(
        "plastic", description="what its surface is mostly made of (sets friction and bounciness from the web)")
    approx_size_cm: Vec3 = Field(description="your own rough guess of its typical real size in cm "
                                             "(x, y, z); cylinder/sphere: (d, d, h). Used to sanity-check web data")


GoalType = Literal["on", "inside", "at", "lifted", "upright", "toppled", "opened", "closed",
                   "left_of", "right_of", "in_front_of", "behind"]


class Goal(BaseModel):
    """A success predicate checked in code at the end of the run.

    The vocabulary follows NVIDIA RoboLab's task conditionals (object_on_top, object_in_container,
    object_left_of, ...): every goal is a simple geometric test on final simulator state.
    """

    type: GoalType = Field(
        description="on: obj rests on top of target; inside: obj is in target bin or drawer; at: obj near xyz; "
                    "lifted: obj held above floor; upright: obj not tipped over; toppled: obj knocked over or "
                    "down from where it started; opened/closed: obj is a drawer pulled open / pushed shut; "
                    "left_of/right_of/in_front_of/behind: obj relative to target, robot's view")
    obj: str
    target: str | None = Field(None, description="object name, for on/inside/left_of/right_of/in_front_of/behind")
    xyz: Vec3 | None = Field(None, description="world position, for 'at'")


class TaskSpec(BaseModel):
    summary: str = Field(min_length=5, description="one-line restatement of the task (required, non-empty)")
    robot: Literal["franka_panda"] = "franka_panda"
    objects: list[ObjectSpec]
    goals: list[Goal]
    search_queries: list[str] = Field(
        description="1-3 web searches that find real dimensions/mass of the objects", max_length=3
    )

    @model_validator(mode="after")
    def _goals_reference_objects(self) -> TaskSpec:
        """Every goal must name declared objects; a bad reference goes back to the model to fix."""
        for o in self.objects:  # a "rubiks_cube" once came back as a black cylinder
            if any(w in o.name.lower() for w in ("cube", "block", "brick", "box")) and o.shape not in ("box", "bin", "drawer"):
                raise ValueError(f"'{o.name}' sounds box-shaped but has shape '{o.shape}'; use shape 'box'")
        names = {o.name for o in self.objects}
        bins = {o.name for o in self.objects if o.shape in ("bin", "drawer")}
        drawers = {o.name for o in self.objects if o.shape == "drawer"}
        for g in self.goals:
            for ref in (g.obj, g.target):
                if ref is not None and ref not in names:
                    raise ValueError(f"goal {g.type} refers to '{ref}', which is not one of the objects {sorted(names)}")
            if g.type == "inside" and g.target not in bins:
                raise ValueError(f"'inside' goal target '{g.target}' must be an object with shape 'bin' or 'drawer'")
            if g.type in ("opened", "closed") and g.obj not in drawers:
                raise ValueError(f"'{g.type}' goal needs obj to be a 'drawer' object, got '{g.obj}'")
            if g.type == "toppled" and any(h.type in ("on", "upright") and h.obj == g.obj for h in self.goals):
                raise ValueError(f"'{g.obj}' must end both toppled and on/upright; goals describe only the FINAL "
                                 "state (stack-then-knock-over ends with the upper objects toppled, no 'on' goals)")
            if g.type == "opened" and any(h.type == "closed" and h.obj == g.obj for h in self.goals):
                raise ValueError(f"'{g.obj}' has both 'opened' and 'closed' goals; goals describe only the FINAL "
                                 "state (for 'open it, put X in, close it' use inside + closed)")
            if g.obj in bins and g.type not in ("opened", "closed"):
                raise ValueError(f"'{g.obj}' is fixed in place; only 'opened'/'closed' goals can name it")
            if g.type == "at" and (g.xyz is None or not reachable_xy(g.xyz[0], g.xyz[1])):
                raise ValueError(f"'at' goal xyz {g.xyz} is outside the robot's reach: x {REACH_X}, "
                                 f"y {REACH_Y} m (the robot base is at 0,0)")
        return self


# ---------- 2. Grounder ----------
class ObjectFact(BaseModel):
    name: str = Field(description="matches ObjectSpec.name")
    size: Vec3 = Field(description="full extents in meters (x, y, z); cylinder/sphere: (d, d, h)")
    mass: float = Field(description="kg")
    source_url: str | None = Field(None, description="URL the numbers came from, if any")
    note: str = Field("", description="short justification, e.g. 'standard block is 5 cm'")
    material: str | None = None
    image_url: str | None = Field(None, description="product image from the search results")
    sources_checked: int = Field(0, description="independent web sources that gave a size")
    sources_agree: int = Field(0, description="of those, how many agree with the chosen size (within 20%)")


class MaterialFact(BaseModel):
    """Contact physics for one material, from the web (cached and shared across runs).
    A value counts as sourced only if its quote was found verbatim in the fetched text."""
    material: str
    friction: float = Field(description="sliding friction coefficient against a table")
    restitution: float = Field(description="coefficient of restitution, 0 (dead) to 1 (perfect bounce)")
    friction_source: str | None = None
    friction_quote: str = ""
    restitution_source: str | None = None
    restitution_quote: str = ""
    note: str = ""


class RobotFact(BaseModel):
    """Franka Panda limits from its spec sheet (cached); quotes verified against the fetched text."""
    gripper_opening_mm: float
    payload_kg: float
    reach_mm: float
    gripper_source: str | None = None
    gripper_quote: str = ""
    arm_source: str | None = None
    arm_quote: str = ""
    note: str = ""


class GroundedFacts(BaseModel):
    facts: list[ObjectFact]
    materials: list[MaterialFact] = []
    robot: RobotFact | None = None

    def physics(self) -> dict[str, tuple[float, float]]:
        """object name -> (friction, restitution), for the simulator."""
        by_mat = {m.material: m for m in self.materials}
        return {f.name: (by_mat[f.material].friction, by_mat[f.material].restitution)
                for f in self.facts if f.material in by_mat}


class Source(BaseModel):
    title: str
    url: str


# ---------- 3. Builder ----------
class SceneObject(BaseModel):
    name: str
    shape: Literal["box", "cylinder", "sphere", "bin", "drawer"]
    size: Vec3 = Field(description="full extents in meters (x, y, z); cylinder/sphere: (d, d, h); "
                                   "bin: outer extents, open at the top")
    mass: float
    rgba: RGBA
    pos: Vec3 = Field(description="center position in world; z = height/2 to rest on the floor")
    yaw: float = Field(0.0, description="rotation about z, radians")
    fixed: bool = Field(False, description="true for static things like a table or tray; bins are always fixed")


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
    viz_path: str | None = Field(None, description="viz.json for the browser 3D replay")
    duration_s: float = 0.0


# ---------- 5. Critic ----------
class Critique(BaseModel):
    # Field order matters: the model first says what it SEES, then quotes the measurements, then decides.
    what_happened: str = Field(description="what the frames show, frame by frame where it matters: where the gripper "
                                           "is, whether the object moves with it, where it ends up")
    evidence: str = Field(description="quote the failed goal details and the sim events that show what went wrong")
    category: Literal[
        "missed_grasp", "dropped_object", "wrong_placement", "collision", "unreachable",
        "unstable_scene", "scene_error", "other",
    ]
    root_cause: str
    suggested_fix: str = Field(description="concrete change to scene or plan, with numbers")
