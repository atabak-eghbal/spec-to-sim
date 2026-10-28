"""Step 3 - Builder (Nemotron 3 Super) and repair (Super, or Ultra when escalated).

Produces a `Build`: the objects to place (data, not XML) and the motion plan our fixed
controller executes. `repair()` takes the failed Build plus the Runner's metrics and the Critic's
diagnosis, and returns a corrected Build - the same contract, so the loop can swap models freely.
"""
from __future__ import annotations

from langsmith import traceable

from spec2sim import config
from spec2sim.llm import CallStat, chat_json
from spec2sim.schemas import Build, Critique, GroundedFacts, RunResult, TaskSpec
from spec2sim.sim.scene import WORKSPACE

SYSTEM = f"""You design a MuJoCo scene and a motion plan for a Franka Panda arm. You output DATA
only; a fixed controller executes it and the result is checked numerically.

WORLD: meters, kg, radians. Robot base at origin. +x forward, +y left, +z up. Floor at z=0.
Place objects within x in [{WORKSPACE['x'][0]}, {WORKSPACE['x'][1]}], y in [{WORKSPACE['y'][0]}, {WORKSPACE['y'][1]}].
Objects resting on the floor need pos z = size_z / 2. Keep >= 10 cm between objects so the
gripper fits. Use the given real sizes and masses exactly. rgba values are 0-1.

GRIPPER: always points straight down. The controller tracks the tool point (TCP) = the center
between the fingertips. Fingers open to 8 cm total: an object can only be grasped if its width
across the fingers is <= 7.8 cm. `yaw` (radians) rotates the gripper about z; yaw=0 closes the
fingers along the world y axis.

ACTIONS (executed in order):
- {{"op":"move_to","object":NAME,"offset":[dx,dy,dz]}} move TCP to NAME's CURRENT center + offset
- {{"op":"move_to","xyz":[x,y,z]}}  move TCP to a world point
- {{"op":"close"}} close gripper; if the TCP is inside/within 2 cm of an object that fits, it is grasped
- {{"op":"open"}}  open gripper; the held object is released
- {{"op":"wait"}}
Each action has "duration" in seconds (0.3-1.5 typical; total plan under 10 s). Set unused fields
to null (object, xyz) or defaults (offset [0,0,0], yaw 0).

RELIABLE PICK-AND-PLACE PATTERN (A onto B, both resting on the floor):
1. move_to A offset [0,0,0.12]          (approach from above, 1.0 s)
2. move_to A offset [0,0,0]             (TCP at A's center, 0.6 s)
3. close                                (0.4 s)
4. move_to A offset [0,0,0.15]          (lift, 0.7 s)
5. move_to B offset [0,0,h_B/2 + h_A/2 + 0.15]   (carry above B, 1.0 s)
6. move_to B offset [0,0,h_B/2 + h_A/2 + 0.003]  (lower until A's bottom touches B's top, 0.6 s)
7. open                                 (0.3 s)
8. move_to B offset [0,0,0.2]           (retreat, 0.6 s)
Here h_A, h_B are the objects' heights (size z). Because A was grasped at its center, the TCP
height above B's center when A is resting on B is h_B/2 + h_A/2. Compute these numbers.
Return JSON only."""


def _context(task: TaskSpec, facts: GroundedFacts) -> str:
    return (f"TASK: {task.summary}\nOBJECTS: {[o.model_dump() for o in task.objects]}\n"
            f"GOALS (checked in code): {[g.model_dump(exclude_none=True) for g in task.goals]}\n"
            f"REAL FACTS (use these sizes/masses): {[f.model_dump() for f in facts.facts]}")


@traceable(name="builder")
def build(task: TaskSpec, facts: GroundedFacts) -> tuple[Build, list[CallStat]]:
    return chat_json("builder", config.SUPER, SYSTEM, _context(task, facts), Build, max_tokens=16000)


REPAIR_ADDENDUM = """

You are REPAIRING a previous attempt that failed. You get the previous scene+plan, the measured
results (ground truth), sim events, and a vision critic's diagnosis of the video. Fix the root cause
with the smallest change that makes every goal pass. Do not change object sizes or masses - they
are real-world facts. Explain the fix in `rationale`."""


@traceable(name="repair")
def repair(task: TaskSpec, facts: GroundedFacts, prev: Build, result: RunResult,
           critique: Critique, model: config.ModelInfo, attempt: int) -> tuple[Build, list[CallStat]]:
    user = (_context(task, facts)
            + f"\n\nPREVIOUS ATTEMPT {attempt} (failed):\n{prev.model_dump_json()}"
            + f"\n\nMEASURED RESULTS:\nerror: {result.error}\n"
            + "\n".join(f"- {'PASS' if g.ok else 'FAIL'}: {g.detail}" for g in result.goals)
            + "\nSIM EVENTS:\n" + "\n".join(f"- {e}" for e in result.events)
            + f"\n\nCRITIC DIAGNOSIS:\n{critique.model_dump_json()}")
    return chat_json(f"repair#{attempt}", model, SYSTEM + REPAIR_ADDENDUM, user, Build, max_tokens=16000)
