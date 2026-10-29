"""Behaviour-cloning policy: trains on a tiny generated dataset and rolls out without errors."""
from pathlib import Path

from spec2sim.policy import evaluate, load_dataset, make_fk, train
from spec2sim.schemas import Action, Build, Goal, MotionPlan, SceneObject
from spec2sim.sim.dataset import generate
from spec2sim.sim.runner import _Sim

C = 0.05
ON = [Goal(type="on", obj="a", target="b")]


def _build() -> Build:
    objs = [SceneObject(name=n, shape="box", size=(C, C, C), mass=0.1, rgba=(1, 0, 0, 1), pos=p)
            for n, p in (("a", (0.5, 0.15, C / 2)), ("b", (0.5, -0.15, C / 2)))]
    plan = MotionPlan(actions=[
        Action(op="move_to", object="a", offset=(0, 0, 0.12)),
        Action(op="move_to", object="a", offset=(0, 0, 0), duration=0.6),
        Action(op="close", duration=0.4),
        Action(op="move_to", object="a", offset=(0, 0, 0.15), duration=0.7),
        Action(op="move_to", object="b", offset=(0, 0, 0.2), duration=1.0),
        Action(op="move_to", object="b", offset=(0, 0, C + 0.003), duration=0.6),
        Action(op="open", duration=0.3),
        Action(op="move_to", object="b", offset=(0, 0, 0.2), duration=0.6),
    ])
    return Build(objects=objs, plan=plan, rationale="two-cube stack")


def test_train_and_rollout(tmp_path: Path):
    build = _build()
    ds = tmp_path / "ds"
    res = generate(build, ON, "stack a on b", ds, episodes=3, images=False)
    assert res["episodes"] >= 2
    assert len(load_dataset(ds)["episodes"]) == res["episodes"]
    fk = make_fk(_Sim(build, None))
    for kind in ("mlp", "knn"):
        policy, _ = train(ds, kind=kind, epochs=20, fk=fk)
        r = evaluate(build, ON, trials=2, seed=5, policy=policy)
        assert r["trials"] == 2 and 0 <= r["passed"] <= 2  # runs to completion; success is not asserted
