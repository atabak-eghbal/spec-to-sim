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
(at most 4). Approximate every object as one primitive: box, cylinder or sphere.

Object names: short snake_case like red_cube, soda_can, tennis_ball. Colors: plain names.

Goals are checked numerically at the end of the simulation:
- on:     {"type":"on", "obj":A, "target":B}      A rests on top of B
- at:     {"type":"at", "obj":A, "xyz":[x,y,z]}   A's center ends near a world point (floor z = half height)
- lifted: {"type":"lifted", "obj":A}              A is held at least 5 cm above the floor at the end
Use null for fields a goal type does not need.

search_queries: 1-3 web searches that would return the REAL dimensions and mass of the objects.
Name a concrete real product with published specs, e.g. "12 oz soda can dimensions mm weight",
"Rubik's cube 3x3 dimensions mm weight grams", "tennis ball diameter mass ITF". For a generic shape
like "red cube", search for the closest common product (e.g. "Melissa & Doug wooden block size").
summary: always a non-empty one-line restatement.
Return JSON only."""


@traceable(name="planner")
def plan(prompt: str) -> tuple[TaskSpec, list[CallStat]]:
    return chat_json("planner", config.SUPER, SYSTEM, f"Task: {prompt}", TaskSpec, max_tokens=4000)
