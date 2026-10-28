"""Step 1 - Planner (Nemotron 3 Super): plain-English task -> structured TaskSpec.

Decides which objects exist, what success means (goal predicates checked in code later),
and which web searches will ground the objects' real sizes and masses.
"""
from __future__ import annotations

from langsmith import traceable

from spec2sim import config
from spec2sim.llm import CallStat, chat_json
from spec2sim.schemas import TaskSpec

SYSTEM = """You turn a robot manipulation request into a structured task spec for a MuJoCo simulation.

Robot: one Franka Emika Panda arm with a parallel gripper, mounted on the floor at the origin.
Objects sit on the floor in front of it. Keep the scene minimal: only objects the task needs
(at most 6). Approximate every object as one primitive: box, cylinder or sphere. Containers that
things are put INTO (bin, basket, tray, box with an open top, bowl) are shape "bin"; describe
them as TABLETOP size (e.g. "small desktop recycling bin", "shallow fruit basket"), never
full-size floor bins: the arm must reach over them.

Object names: short snake_case like red_cube, soda_can, tennis_ball. Colors: plain names.
approx_size_cm: your own estimate of the object's typical size (a toy block ~[3,3,3], a soda can
[6.6,6.6,12.2], an apple ~[7.5,7.5,7.5]); web results that disagree wildly are rejected.

Goals are checked numerically at the end of the simulation:
- on:     {"type":"on", "obj":A, "target":B}      A rests on top of B
- at:     {"type":"at", "obj":A, "xyz":[x,y,z]}   A's center ends near a world point (floor z = half height);
          only when the task names a place. x must be 0.30-0.70, y -0.40..0.40 (the base is at 0,0).
          A tower or stack needs only "on" goals, not "at".
- inside: {"type":"inside", "obj":A, "target":BIN}  A ends up inside the bin
- lifted: {"type":"lifted", "obj":A}              A is held at least 5 cm above the floor at the end
- upright: {"type":"upright", "obj":A}            A is not tipped over (for cans, bottles, cups)
- left_of / right_of / in_front_of / behind: {"type":"left_of", "obj":A, "target":B}
          A ends on that side of B, from the robot's point of view (in front = farther from robot)
A task with several objects needs one goal per object (e.g. sorting 3 items = 3 "inside" goals).
Use null for fields a goal type does not need. Write only the goals the task asks for: use
"lifted" ONLY when the task ends with the object held in the air, never together with "on" or
"at" for the same object.

search_queries: 1-3 web searches that would return the REAL dimensions and mass of the objects.
Make every query unambiguous: "apple fruit average diameter weight", not "apple" (that finds
Apple Inc. products); "orange fruit size weight", not "orange".
Name a concrete real product with published specs, e.g. "12 oz soda can dimensions mm weight",
"Rubik's cube 3x3 dimensions mm weight grams", "tennis ball diameter mass ITF". For a generic shape
like "red cube", search for the closest common product (e.g. "Melissa & Doug wooden block size").
summary: always a non-empty one-line restatement.
Return JSON only."""


@traceable(name="planner")
def plan(prompt: str) -> tuple[TaskSpec, list[CallStat]]:
    return chat_json("planner", config.SUPER, SYSTEM, f"Task: {prompt}", TaskSpec, max_tokens=4000)
