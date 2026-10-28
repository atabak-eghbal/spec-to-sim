"""Dev check: run a hand-written stacking Build through the Runner (no LLM).

Run: uv run python scripts/try_runner.py  ->  runs/try_runner/run.mp4
"""
from pathlib import Path

from spec2sim.schemas import Action, Build, Goal, MotionPlan, SceneObject
from spec2sim.sim.runner import run

C = 0.05  # cube edge, m
build = Build(
    objects=[
        SceneObject(name="red_cube", shape="box", size=(C, C, C), mass=0.1,
                    rgba=(0.85, 0.15, 0.15, 1), pos=(0.5, 0.15, C / 2)),
        SceneObject(name="blue_cube", shape="box", size=(C, C, C), mass=0.1,
                    rgba=(0.15, 0.3, 0.85, 1), pos=(0.5, -0.15, C / 2)),
    ],
    plan=MotionPlan(actions=[
        Action(op="move_to", object="red_cube", offset=(0, 0, 0.12), duration=1.0),
        Action(op="move_to", object="red_cube", offset=(0, 0, 0.0), duration=0.6),
        Action(op="close", duration=0.4),
        Action(op="move_to", object="red_cube", offset=(0, 0, 0.15), duration=0.7),
        Action(op="move_to", object="blue_cube", offset=(0, 0, 0.15), duration=1.0),
        Action(op="move_to", object="blue_cube", offset=(0, 0, C + 0.003), duration=0.6),
        Action(op="open", duration=0.3),
        Action(op="move_to", object="blue_cube", offset=(0, 0, 0.2), duration=0.6),
    ]),
    rationale="hand-written reference plan",
)
res = run(build, [Goal(type="on", obj="red_cube", target="blue_cube")], Path("runs/try_runner"))
print("success:", res.success, "| error:", res.error, "| wall:", res.duration_s, "s")
for g in res.goals:
    print(" ", "PASS" if g.ok else "FAIL", g.detail)
for e in res.events:
    print("  event:", e)
print(res.video_path, res.contact_sheet_path)
