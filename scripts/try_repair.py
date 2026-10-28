"""Dev check: force a failure (release 10 cm too high) and exercise Critic -> repair -> rerun.

Run: uv run python scripts/try_repair.py
"""
import time
from pathlib import Path

from spec2sim import builder, config, critic
from spec2sim.schemas import (
    Action,
    Build,
    Goal,
    GroundedFacts,
    MotionPlan,
    ObjectFact,
    ObjectSpec,
    SceneObject,
    TaskSpec,
)
from spec2sim.sim.runner import run

C = 0.05
task = TaskSpec(summary="Stack the red cube on the blue cube.",
                objects=[ObjectSpec(name="red_cube", description="toy block", shape="box", color="red"),
                         ObjectSpec(name="blue_cube", description="toy block", shape="box", color="blue")],
                goals=[Goal(type="on", obj="red_cube", target="blue_cube")], search_queries=[])
facts = GroundedFacts(facts=[ObjectFact(name=n, size=(C, C, C), mass=0.1) for n in ("red_cube", "blue_cube")])
bad = Build(
    objects=[SceneObject(name="red_cube", shape="box", size=(C, C, C), mass=0.1, rgba=(0.85, .15, .15, 1), pos=(0.5, 0.15, C / 2)),
             SceneObject(name="blue_cube", shape="box", size=(C, C, C), mass=0.1, rgba=(.15, .3, .85, 1), pos=(0.5, -0.15, C / 2))],
    plan=MotionPlan(actions=[
        Action(op="move_to", object="red_cube", offset=(0, 0, 0.12)),
        Action(op="move_to", object="red_cube", offset=(0, 0, 0), duration=0.6),
        Action(op="close", duration=0.4),
        Action(op="move_to", object="red_cube", offset=(0, 0, 0.2), duration=0.7),
        Action(op="move_to", object="blue_cube", offset=(0.03, 0, 0.16), duration=1.0),
        Action(op="open", duration=0.3),
        Action(op="move_to", object="blue_cube", offset=(0, 0, 0.25), duration=0.6)]),
    rationale="deliberately broken: releases 10 cm above and 3 cm off-center")

r1 = run(bad, task.goals, Path("runs/try_repair/attempt1"))
print("attempt1 success:", r1.success, [g.detail for g in r1.goals], r1.events)
t = time.time()
crit, st = critic.critique(task, bad, r1)
print(f"\nCRITIC {time.time() - t:.1f}s ${sum(s.cost_usd for s in st):.4f}:", crit.model_dump_json(indent=1))
fixed, st = builder.repair(task, facts, bad, r1, crit, config.SUPER, 1)
print(f"\nREPAIR ${sum(s.cost_usd for s in st):.4f}: {fixed.rationale}")
for a in fixed.plan.actions:
    print("  ", a.op, a.object, a.offset)
r2 = run(fixed, task.goals, Path("runs/try_repair/attempt2"))
print("\nattempt2 success:", r2.success, [g.detail for g in r2.goals], r2.events)
