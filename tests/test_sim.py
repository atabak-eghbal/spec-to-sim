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


def _bin(name="bin", pos=(0.45, -0.2, 0.04), size=(0.2, 0.2, 0.08)):
    return SceneObject(name=name, shape="bin", size=size, mass=0.5, rgba=(0.3, 0.3, 0.8, 1), pos=pos)


def _drop_in_bin(obj: str, dx: float, bin_h: float = 0.08) -> list[Action]:
    """Pick obj, carry it over the bin, release it 2 cm above the rim."""
    return [
        Action(op="move_to", object=obj, offset=(0, 0, 0.12)),
        Action(op="move_to", object=obj, offset=(0, 0, 0), duration=0.6),
        Action(op="close", duration=0.4),
        Action(op="move_to", object=obj, offset=(0, 0, 0.18), duration=0.7),
        Action(op="move_to", object="bin", offset=(dx, 0, bin_h / 2 + C / 2 + 0.15), duration=1.0),
        Action(op="move_to", object="bin", offset=(dx, 0, bin_h / 2 + C / 2 + 0.02), duration=0.6),
        Action(op="open", duration=0.3),
        Action(op="move_to", object="bin", offset=(0, 0, 0.25), duration=0.6),
    ]


def test_two_objects_into_bin(tmp_path: Path):
    objs = [_obj("a", (0.4, 0.15, C / 2)), _obj("b", (0.6, 0.15, C / 2)), _bin()]
    plan = MotionPlan(actions=_drop_in_bin("a", -0.04) + _drop_in_bin("b", 0.04))
    goals = [Goal(type="inside", obj="a", target="bin"), Goal(type="inside", obj="b", target="bin"),
             Goal(type="upright", obj="a"), Goal(type="right_of", obj="a", target="b")]
    r = run(Build(objects=objs, plan=plan, rationale="sort into bin"), goals, tmp_path)
    by = {g.goal.type: g for g in r.goals}
    assert r.goals[0].ok and r.goals[1].ok, (r.goals, r.events)
    assert by["upright"].ok
    # a starts left of b... after sorting both sit in the bin with a at dx=-0.04 (x, not y): not right_of
    assert not by["right_of"].ok
    assert any("bottom" in e and "above bin" in e for e in r.events), r.events


def test_object_left_on_floor_is_not_inside(tmp_path: Path):
    objs = [_obj("a", (0.4, 0.15, C / 2)), _bin()]
    r = run(Build(objects=objs, plan=MotionPlan(actions=[Action(op="wait", duration=0.5)]), rationale="idle run"),
            [Goal(type="inside", obj="a", target="bin"), Goal(type="left_of", obj="a", target="bin")], tmp_path)
    assert not r.goals[0].ok and "outside" in r.goals[0].detail
    assert r.goals[1].ok  # a is at y=+0.15, bin at y=-0.2 -> a is on the robot's left


def test_browser_replay_files(tmp_path: Path):
    import json

    from spec2sim.sim.webviz import export_meshes
    r = run(Build(objects=_cubes(), plan=_stack_plan(C + 0.003), rationale="reference"), ON, tmp_path)
    meta = json.loads(Path(r.viz_path).read_text())
    n_bodies = len(meta["bodies"])
    traj = (tmp_path / "traj.bin").read_bytes()
    assert len(traj) == meta["frames"] * n_bodies * 7 * 4 and meta["frames"] == len(meta["times"])
    assert {g["object"] for g in meta["geoms"] if g["object"]} == {"a", "b"}
    assert meta["goals"][0]["ok"] and any("grasped a" in e for e in meta["events"])
    export_meshes(build_scene(_cubes()).model, tmp_path / "viz")
    manifest = json.loads((tmp_path / "viz/meshes.json").read_text())
    assert {g["mesh"] for g in meta["geoms"] if g["mesh"]} <= set(manifest)
