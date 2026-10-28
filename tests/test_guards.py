"""Guards around the LLM steps (no API calls): JSON recovery, task validation, feasibility, anti-cheat."""
from pathlib import Path

import pytest
from pydantic import ValidationError

from spec2sim.llm import _json_part
from spec2sim.loop import infeasible
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


def test_json_part_skips_leading_reasoning():
    text = 'We need to output JSON only. Something like {"a": 1}. Final:\n{"objects": [], "x": {"y": 2}}'
    assert _json_part(text) == '{"objects": [], "x": {"y": 2}}'
    assert _json_part('{"a": 1}') == '{"a": 1}'


def _spec(goals, shapes=None):
    shapes = shapes or {"apple": "sphere", "basket": "bin"}
    objs = [ObjectSpec(name=n, description=n, shape=s, color="red", approx_size_cm=(7, 7, 7)) for n, s in shapes.items()]
    return TaskSpec(summary="sort fruit", objects=objs, goals=goals, search_queries=["q"])


def test_taskspec_rejects_unknown_target():
    with pytest.raises(ValidationError, match="not one of the objects"):
        _spec([Goal(type="inside", obj="apple", target="target': ")])
    with pytest.raises(ValidationError, match="must be an object with shape 'bin'"):
        _spec([Goal(type="inside", obj="basket", target="apple")])
    _spec([Goal(type="inside", obj="apple", target="basket")])  # valid


def test_too_wide_object_is_infeasible_before_building():
    task = _spec([Goal(type="inside", obj="apple", target="basket")])
    facts = lambda d: GroundedFacts(facts=[ObjectFact(name="apple", size=(d, d, d), mass=0.2),
                                           ObjectFact(name="basket", size=(0.3, 0.2, 0.1), mass=0.3)])
    assert "opens only" in infeasible(task, facts(0.083))
    assert infeasible(task, facts(0.07)) is None


def test_goal_already_true_at_start_is_rejected(tmp_path: Path):
    cube = lambda n, z: SceneObject(name=n, shape="box", size=(0.05, 0.05, 0.05), mass=0.1,
                                    rgba=(1, 0, 0, 1), pos=(0.5, 0.0, z))
    b = Build(objects=[cube("b", 0.025), cube("a", 0.076)],  # a already sits on b
              plan=MotionPlan(actions=[Action(op="wait", duration=0.2)]), rationale="cheat test")
    r = run(b, [Goal(type="on", obj="a", target="b")], tmp_path)
    assert not r.success and "already starts on b" in r.error


def test_grasp_width_uses_real_shape():
    import numpy as np

    from spec2sim.sim.controller import _width_across
    ball = SceneObject(name="ball", shape="sphere", size=(0.068,) * 3, mass=0.06, rgba=(1, 1, 0, 1), pos=(0.5, 0, 0.034))
    tilted = np.array([[0.6, -0.8, 0], [0.48, 0.36, -0.8], [0.64, 0.48, 0.6]])  # any rotation
    assert _width_across(ball, tilted, np.array([0, 1.0, 0])) == 0.068
    can = SceneObject(name="can", shape="cylinder", size=(0.066, 0.066, 0.123), mass=0.37, rgba=(1, 1, 1, 1), pos=(0.5, 0, 0.06))
    assert abs(_width_across(can, np.eye(3), np.array([0, 1.0, 0])) - 0.066) < 1e-9  # upright can


def test_grounder_rejects_implausible_web_sizes():
    from spec2sim.grounder import _Extracted, _to_fact
    ex = lambda size, g=1000: _Extracted(name="x", quote="q", source_url="http://s", size_mm=size, mass_g=g, note="from quote")
    block = _to_fact(ex((440, 100, 215)), "box", prior_cm=(3, 3, 3))  # furniture-sized "block"
    assert block.size == (0.03, 0.03, 0.03) and block.source_url is None and "rejected" in block.note
    can = _to_fact(ex((122, 66, 122), 375), "cylinder", prior_cm=(6.6, 6.6, 12.2))  # scrambled order
    assert can.size == (0.066, 0.066, 0.122) and can.source_url == "http://s"
    apple = _to_fact(ex((89, 89, 89), 200), "sphere", prior_cm=(7.5, 7.5, 7.5))  # big but plausible
    assert apple.size == (0.089, 0.089, 0.089)


def test_scene_reports_all_problems_and_extra_objects():
    from spec2sim.sim.scene import SceneError, validate_objects
    table = SceneObject(name="table", shape="box", size=(1.0, 1.0, 0.02), mass=0.0, rgba=(1, 1, 1, 1), pos=(0.5, 0, -0.01))
    a = SceneObject(name="a", shape="box", size=(0.05,) * 3, mass=0.1, rgba=(1, 0, 0, 1), pos=(0.2, 0, 0.025))
    with pytest.raises(SceneError) as e:
        validate_objects([table, a], allowed={"a"})
    msg = str(e.value)
    assert "not in the task" in msg and "mass 0.0" in msg and "a: pos" in msg and "scene problems" in msg


def test_daily_budget_cap(tmp_path, monkeypatch):
    import json
    import time

    from spec2sim import config, llm
    log = tmp_path / "spend.jsonl"
    monkeypatch.setattr(llm, "SPEND_LOG", log)
    monkeypatch.setattr(config, "DAILY_SPEND_LIMIT_USD", 1.0)
    log.write_text(json.dumps({"cost_usd": 0.6, "ts": time.time()}) + "\n"
                   + json.dumps({"cost_usd": 5.0, "ts": time.time() - 3 * 86400}) + "\n")
    llm.check_budget()  # 0.60 today: allowed (old spend doesn't count toward today)
    with log.open("a") as f:
        f.write(json.dumps({"cost_usd": 0.5, "ts": time.time()}) + "\n")
    with pytest.raises(llm.BudgetError, match="today's model budget"):
        llm.check_budget()
