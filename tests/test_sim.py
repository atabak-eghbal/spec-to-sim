"""Deterministic simulator tests (no LLM calls): scene validation, grasping, success metrics."""
from pathlib import Path

import pytest

from spec2sim.schemas import Action, Build, Goal, MotionPlan, SceneObject
from spec2sim.sim.runner import run
from spec2sim.sim.scene import SceneError, build_scene

C = 0.05
ON = [Goal(type="on", obj="a", target="b")]


def _obj(name, pos, size=(C, C, C), shape="box"):
    return SceneObject(name=name, shape=shape, size=size, mass=0.1, rgba=(1, 0, 0, 1), pos=pos)


def _stack_plan(place_dz: float, a_h: float = C, place_dx: float = 0.0) -> MotionPlan:
    return MotionPlan(actions=[
        Action(op="move_to", object="a", offset=(0, 0, 0.12)),
        Action(op="move_to", object="a", offset=(0, 0, 0), duration=0.6),
        Action(op="close", duration=0.4),
        Action(op="move_to", object="a", offset=(0, 0, 0.15), duration=0.7),
        Action(op="move_to", object="b", offset=(0, 0, 0.15 + a_h), duration=1.0),
        Action(op="move_to", object="b", offset=(place_dx, 0, place_dz), duration=0.6),
        Action(op="open", duration=0.3),
        Action(op="move_to", object="b", offset=(0, 0, 0.2), duration=0.6),
    ])


def _cubes():
    return [_obj("a", (0.5, 0.15, C / 2)), _obj("b", (0.5, -0.15, C / 2))]


def test_reference_stack_succeeds(tmp_path: Path):
    r = run(Build(objects=_cubes(), plan=_stack_plan(C + 0.003), rationale="reference"), ON, tmp_path)
    assert r.success, (r.goals, r.events)
    assert any("grasped a" in e for e in r.events)
    assert Path(r.video_path).exists() and Path(r.contact_sheet_path).exists()


def test_offcenter_drop_fails(tmp_path: Path):
    plan = _stack_plan(0.18, place_dx=0.03)  # released 13 cm up and 3 cm off-center -> falls off
    r = run(Build(objects=_cubes(), plan=plan, rationale="dropped from height"), ON, tmp_path)
    assert not r.success
    assert any("released a" in e and "cm above" in e for e in r.events)


def test_too_wide_object_is_not_grasped(tmp_path: Path):
    wide = [_obj("a", (0.5, 0.15, 0.05), size=(0.09, 0.09, 0.1), shape="cylinder"),
            _obj("b", (0.5, -0.15, C / 2))]
    r = run(Build(objects=wide, plan=_stack_plan(C / 2 + 0.05), rationale="too wide to grasp"), ON, tmp_path)
    assert not r.success
    assert any("gripper opens only" in e for e in r.events)


@pytest.mark.parametrize("bad", [
    [_obj("a", (1.2, 0.0, C / 2))],  # out of reach
    [_obj("a", (0.5, 0.0, C / 2)), _obj("a", (0.5, 0.2, C / 2))],  # duplicate names
    [_obj("a", (0.5, 0.0, C / 2)), _obj("b", (0.51, 0.0, C / 2))],  # overlapping
])
def test_invalid_scenes_rejected(bad):
    with pytest.raises(SceneError):
        build_scene(bad)
