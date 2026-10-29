"""Step 3 - Builder and repair (Nemotron 3 Super).

Produces a `Build`: the objects to place (data, not XML) and the motion plan our fixed
controller executes. `repair()` takes the failed Build plus the Runner's metrics and the Critic's
diagnosis, and returns a corrected Build - the same contract, so the loop can swap models freely.
"""
from __future__ import annotations

from langsmith import traceable

from spec2sim import config
from spec2sim.llm import CallStat, chat_json
from spec2sim.schemas import REACH_R, Build, Critique, GroundedFacts, RunResult, TaskSpec
from spec2sim.sim.scene import WORKSPACE

SYSTEM = f"""You design a MuJoCo scene and a motion plan for a Franka Panda arm. You output DATA
only; a fixed controller executes it and the result is checked numerically.

WORLD: meters, kg, radians. Robot base at origin. +x forward, +y left, +z up. Floor at z=0.
The floor IS the table: never add a table, floor or any object that is not in OBJECTS, and use
the exact object names given.
Place objects within x in [{WORKSPACE['x'][0]}, {WORKSPACE['x'][1]}], y in [{WORKSPACE['y'][0]}, {WORKSPACE['y'][1]}], at most
{REACH_R} m from the base. Every move_to target must also be inside that area.
Objects resting on the floor need pos z = size_z / 2. Keep >= 10 cm between objects so the
gripper fits. Use the given real sizes and masses exactly (bins are already scaled to tabletop
size). rgba values are 0-1.
BINS (shape "bin"): open-top containers, fixed in place; size = outer extents. Give them their
own area and put the objects to pick elsewhere, e.g. pick objects in a row at y = +0.2 and bins
centered at y = -0.22. A bin's footprint must not touch any other object's footprint (10 cm gap), and its WHOLE
footprint must be inside the workspace: center x >= {WORKSPACE['x'][0]} + size_x/2,
center x <= {WORKSPACE['x'][1]} - size_x/2, |center y| <= {WORKSPACE['y'][1]} - size_y/2.

GRIPPER: always points straight down. The controller tracks the tool point (TCP) = the center
between the fingertips. Fingers open 8 cm total (7.8 cm usable): an object can only be grasped if its width
across the fingers is <= 7.8 cm. `yaw` (radians) rotates the gripper about z; yaw=0 closes the
fingers along the world x axis (toward/away from the robot), yaw=1.5708 along y.

DRAWERS (shape "drawer"): a fixed cabinet whose open front faces the robot; its drawer slides out
toward the robot (-x). yaw must be 0. Put the cabinet center at x = 0.48 + size_x/2 + 0.02 (so the
pulled-out handle stays in reach) and keep 15 cm free in front of it. Anchors usable as move_to "object", written as the drawer's real name + anchor (for a
drawer named desk_drawer: "desk_drawer.handle"): ".handle" (the handle now), ".open" (where the
handle is when the drawer is pulled 12 cm out), ".closed" (where it is when shut), ".slot" (the
floor of the part of the drawer that is out of the cabinet: drop objects there). Below, D stands
for the drawer's name.
OBJECTS MAY START STACKED (e.g. a tower to knock over): pos z = (height of what is below) + size_z/2.

ACTIONS (executed in order). Prefer SKILLS: code turns them into waypoints from the real sizes and
the objects' current positions, so you never compute heights:
- {{"op":"pick","object":A}}                      grasp A at its center and lift it clear of everything
- {{"op":"place_on","target":B,"offset":[dx,dy,0]}} set the held object down on top of B (dx,dy = spot on B's top, usually 0) and let go
- {{"op":"drop_into","target":BIN,"offset":[dx,dy,0]}} lower the held object into the bin and let go; give each object its own
  spot (dx,dy a few cm apart, inside the bin) so they don't land on each other
- {{"op":"push","object":A,"direction":"+x"|"-x"|"+y"|"-y"}} knock A over (or slide it) toward that direction; for a stack, push the TOP object
Primitives, only for drawers or anything the skills can't do:
- {{"op":"move_to","object":NAME,"offset":[dx,dy,dz]}} move the tool point to NAME's CURRENT center + offset
- {{"op":"move_to","xyz":[x,y,z]}}  move the tool point to a world point
- {{"op":"close"}} close the gripper; an object is grasped only if it is BETWEEN the fingers (tool point at
  its center height, centered on it)
- {{"op":"open"}}  open the gripper; the held object is released
- {{"op":"wait"}}
Primitives take "duration" in seconds (0.3-1.5 typical). Skills ignore duration. `yaw` (radians)
turns the gripper (skills too); set unused fields to null (object, target, direction, xyz) or
defaults (offset [0,0,0], yaw 0). Total plan under 30 s (a pick + place is about 5 s).

EXAMPLES: A onto B: pick A, place_on B. Into a bin: pick A, drop_into BIN [-0.05,0,0], pick C,
drop_into BIN [0.05,0,0]. A tower red-green-blue: pick green, place_on red, pick blue, place_on green.
Knock over: push A toward open space.

OPEN A DRAWER D: move_to D.handle offset [0,0,0.10] (1.0 s); move_to D.handle offset [0,0,0] (0.6 s);
close (0.4 s); move_to D.open (1.5 s); open (0.3 s); move_to D.handle offset [0,0,0.12] (0.6 s).
CLOSE IT: the same with D.closed instead of D.open.
PUT A INTO AN OPEN DRAWER: pick A, then move_to D.slot offset
[0,0,h_A/2 + 0.15] with yaw 1.5708 (1.0 s), move_to D.slot offset [0,0,h_A/2 + 0.02] yaw 1.5708
(0.6 s), open, move_to D.slot offset [0,0,0.2] yaw 1.5708. (yaw 1.5708 keeps the wide hand clear
of the cabinet top.) Two objects go side by side across the drawer's width: D.slot offsets
[0,+0.045,...] and [0,-0.045,...]. Then close the drawer if the task asks.

Return JSON only."""


def _context(task: TaskSpec, facts: GroundedFacts) -> str:
    return (f"TASK: {task.summary}\nOBJECTS: {[o.model_dump() for o in task.objects]}\n"
            f"GOALS (checked in code): {[g.model_dump(exclude_none=True) for g in task.goals]}\n"
            f"REAL FACTS (use these sizes/masses): {[f.model_dump() for f in facts.facts]}")


@traceable(name="builder")
def build(task: TaskSpec, facts: GroundedFacts) -> tuple[Build, list[CallStat]]:
    return chat_json("builder", config.SUPER, SYSTEM, _context(task, facts), Build, max_tokens=16000, thinking=config.SUPER_THINKING)


REPAIR_ADDENDUM = """

You are REPAIRING a plan that failed. You get every earlier attempt with its measured results
(ground truth) and sim events, and a vision critic's reading of the last run's video. Fix the root
cause with the smallest change that makes every goal pass. Do NOT repeat a scene or plan that
already failed: if a change did not help before, try a different one. Do not change object sizes or
masses - they are real-world facts. Remember the frame: +y is the robot's LEFT (left_of = larger y),
+x is away from the robot. Explain the fix in `rationale`."""


def _plan_brief(b: Build) -> str:
    out = []
    for a in b.plan.actions:
        bits = [a.op] + [x for x in (a.object, a.target, a.direction) if x]
        if any(a.offset):
            bits.append(f"offset {[round(v, 3) for v in a.offset]}")
        if a.xyz:
            bits.append(f"xyz {[round(v, 3) for v in a.xyz]}")
        if a.yaw:
            bits.append(f"yaw {a.yaw:g}")
        out.append(" ".join(bits))
    return "; ".join(out)


def _result_brief(r: RunResult) -> str:
    return ((f"error: {r.error}\n" if r.error else "")
            + "\n".join(f"- {'PASS' if g.ok else 'FAIL'}: {g.detail}" for g in r.goals)
            + ("\nevents: " + " | ".join(r.events[-8:]) if r.events else ""))


@traceable(name="repair")
def repair(task: TaskSpec, facts: GroundedFacts, history: list[tuple[Build, RunResult]],
           critique: Critique, model: config.ModelInfo, attempt: int,
           temperature: float = 0.3) -> tuple[Build, list[CallStat]]:
    """history: every attempt so far as (build, result), oldest first; the last one is shown in full."""
    *older, (prev, result) = history
    past = "".join(f"\n\nATTEMPT {i + 1} (failed): objects "
                   + "; ".join(f"{o.name}@{[round(v, 3) for v in o.pos]}" for o in b.objects)
                   + f"\nplan: {_plan_brief(b)}\n{_result_brief(r)}" for i, (b, r) in enumerate(older))
    user = (_context(task, facts) + past
            + f"\n\nLAST ATTEMPT {attempt} (failed), in full:\n{prev.model_dump_json()}"
            + f"\n\nMEASURED RESULTS:\n{_result_brief(result)}"
            + f"\n\nCRITIC DIAGNOSIS:\n{critique.model_dump_json()}")
    return chat_json(f"repair#{attempt}", model, SYSTEM + REPAIR_ADDENDUM, user, Build, max_tokens=16000,
                     temperature=temperature, thinking=config.SUPER_THINKING)
