"""Build the sourced physics knowledge base once: friction and bounciness for every material the
Planner can choose, plus the Franka's limits, from Tavily searches (cached in .cache/physics).

Each value is kept only if its quote is found verbatim in the fetched text and it is physically
plausible for that material; otherwise a labelled typical value is used. Later runs read the cache.

    uv run python scripts/build_physics_kb.py
"""
from __future__ import annotations

import json
import typing

from spec2sim.grounder import PHYS_CACHE, physics_facts
from spec2sim.schemas import ObjectSpec

MATERIALS = list(typing.get_args(ObjectSpec.model_fields["material"].annotation))

if __name__ == "__main__":
    for i in range(0, len(MATERIALS), 2):  # 2 materials per batch = 4 searches (+2 for the robot first time)
        mats, robot, stats, info = physics_facts(MATERIALS[i:i + 2], budget=6)
        for m in mats:
            print(f"{m.material:10s} friction {m.friction:<5} {'sourced' if m.friction_source else 'typical':8s} "
                  f"restitution {m.restitution:<5} {'sourced' if m.restitution_source else 'typical':8s} {m.note}")
    print("robot:", json.loads((PHYS_CACHE / "robot_franka_panda.json").read_text()))
