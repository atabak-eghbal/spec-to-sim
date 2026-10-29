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
    facts = lambda d: GroundedFacts(facts=[ObjectFact(name="apple", size=(d, d, d), mass=0.2, source_url="http://s", sources_agree=2),
                                           ObjectFact(name="basket", size=(0.3, 0.2, 0.1), mass=0.3)])
    assert "7.8 cm usable" in infeasible(task, facts(0.083))[0]
    assert infeasible(task, facts(0.07)) is None
    # One web source says 8.3 cm but the typical size (7 cm) fits and no second source agrees: try it.
    one = GroundedFacts(facts=[ObjectFact(name="apple", size=(0.083,) * 3, mass=0.2, source_url="http://s", sources_agree=1),
                               ObjectFact(name="basket", size=(0.3, 0.2, 0.1), mass=0.3)])
    assert infeasible(task, one) is None
    # A model's guess with no source can't refuse a task on its own either.
    guess = GroundedFacts(facts=[ObjectFact(name="apple", size=(0.089,) * 3, mass=0.2),
                                 ObjectFact(name="basket", size=(0.3, 0.2, 0.1), mass=0.3)])
    assert infeasible(task, guess) is None


def test_planner_table_dropped_and_names_checked():
    objs = lambda names: [ObjectSpec(name=n, description=n, shape="box", color="red", approx_size_cm=(5, 5, 5)) for n in names]
    t = TaskSpec(summary="knock it over", objects=objs(["soda_can", "table"]),
                 goals=[Goal(type="toppled", obj="soda_can"), Goal(type="on", obj="soda_can", target="table")][:1],
                 search_queries=["q"])
    assert [o.name for o in t.objects] == ["soda_can"]
    with pytest.raises(ValidationError, match="snake_case"):
        TaskSpec(summary="bad names", objects=objs(["", "apple"]), goals=[Goal(type="lifted", obj="apple")], search_queries=["q"])


def test_web_size_oriented_like_the_estimate():
    from spec2sim.grounder import _Extracted, _to_fact
    e = _Extracted(name="soda_can", quote="123 x 123 x 66 mm", source_url="http://s", size_mm=(123, 123, 66), mass_g=380, note="")
    f = _to_fact(e, "cylinder", prior_cm=(6.6, 6.6, 12.2), material="metal")
    assert abs(f.size[0] - 0.066) < 1e-6 and abs(f.size[2] - 0.123) < 1e-6


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


def test_quote_verification_rejects_invented_citations():
    from spec2sim.grounder import _norm, verified
    page = _norm("Technical data: the Franka Hand has a maximum opening stroke of 80 mm and 70 N continuous force.")
    assert verified("a maximum opening stroke of 80 mm", page)
    assert not verified("Maximum gripper opening (stroke) in mm: 100 mm", page)  # what Nano once invented
    assert not verified("stroke of 100 mm", page)  # right words, wrong number


def test_web_restitution_is_calibrated_in_the_simulator():
    import mujoco
    import numpy as np

    from spec2sim.sim.scene import OBJ_PREFIX, build_scene
    ball = SceneObject(name="ball", shape="sphere", size=(0.066,) * 3, mass=0.058, rgba=(1, 1, 0, 1), pos=(0.5, 0.0, 0.4))
    for e_web in (0.6, 0.3):
        sc = build_scene([ball], physics={"ball": (0.9, e_web)})
        m, d = sc.model, mujoco.MjData(sc.model)
        d.qpos[sc.arm_qpos] = [0, -0.8, 0, -2.5, 0, 1.7, 0.8]
        mujoco.mj_forward(m, d)
        z = []
        for _ in range(1200):
            mujoco.mj_step(m, d)
            z.append(d.body(OBJ_PREFIX + "ball").xpos[2])
        z = np.array(z)
        first = int(np.argmax(z < 0.035))
        e_sim = np.sqrt((z[first:first + 500].max() - 0.033) / 0.367)
        assert abs(e_sim - e_web) < 0.06, (e_web, e_sim)


def test_size_mass_two_out_of_three():
    from spec2sim.grounder import _Extracted, _to_fact
    e = _Extracted(name="grapefruit", quote="7.5 inches, 246 g", source_url="http://s", size_mm=(190.5, 190.5, 190.5),
                   mass_g=246, note="from quote")
    f = _to_fact(e, "sphere", prior_cm=(11, 11, 11), material="fruit")
    assert 0.07 < f.size[0] < 0.095 and abs(f.mass - 0.246) < 1e-9 and "size corrected" in f.note


def test_incomplete_web_size_falls_back_to_estimate():
    from spec2sim.grounder import _Extracted, _to_fact
    e = _Extracted(name="bin", quote="q", source_url="http://s", size_mm=(300, 0, 150), mass_g=500, note="x")
    f = _to_fact(e, "box", prior_cm=(20, 15, 10), material="plastic")
    assert min(f.size) > 0 and "incomplete" in f.note


def test_evidence_must_be_checkable():
    """An agreement vote counts only if its page was returned by Tavily, once per page, with the number in it."""
    from spec2sim.grounder import _Evidence, _Extracted, _verified_evidence
    ev = [_Evidence(source="http://a", size_mm=(66, 66, 123), mass_g=None),
          _Evidence(source="http://a", size_mm=(66, 66, 123), mass_g=None),   # same page twice
          _Evidence(source="http://b", size_mm=(66, 66, 123), mass_g=None),   # number not on the page
          _Evidence(source="summary", size_mm=(66, 66, 123), mass_g=None),    # the search summary
          _Evidence(source="http://c", size_mm=(66, 66, 123), mass_g=None)]   # 4.84 in = 12.3 cm
    e = _Extracted(name="can", quote="q", source_url="http://main", size_mm=(66, 66, 123), mass_g=380, note="", evidence=ev)
    pages = {"http://a": "a standard can is 12.3 cm tall", "http://b": "cans come in many sizes", "http://c": "height 4.84 in"}
    kept = _verified_evidence(e, pages)
    assert [k.source for k in kept] == ["http://a", "http://c"]
