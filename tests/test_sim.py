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


def _drawer_scene():
    cab = SceneObject(name="cab", shape="drawer", size=(0.2, 0.24, 0.12), mass=1, rgba=(0.55, 0.42, 0.3, 1),
                      pos=(0.6, -0.05, 0.06))
    cube = SceneObject(name="cube", shape="box", size=(C,) * 3, mass=0.1, rgba=(0.9, 0.2, 0.2, 1), pos=(0.45, 0.25, C / 2))
    return [cab, cube]


def _drawer_plan() -> MotionPlan:
    Y = 1.5708
    grab = lambda: [Action(op="move_to", object="cab.handle", offset=(0, 0, 0.1)),
                    Action(op="move_to", object="cab.handle", duration=0.6), Action(op="close", duration=0.4)]
    leave = lambda: [Action(op="open", duration=0.3), Action(op="move_to", object="cab.handle", offset=(0, 0, 0.12), duration=0.6)]
    return MotionPlan(actions=[
        *grab(), Action(op="move_to", object="cab.open", duration=1.5), *leave(),
        Action(op="move_to", object="cube", offset=(0, 0, 0.12)), Action(op="move_to", object="cube", duration=0.6),
        Action(op="close", duration=0.4), Action(op="move_to", object="cube", offset=(0, 0, 0.18), duration=0.7),
        Action(op="move_to", object="cab.slot", offset=(0, 0, C / 2 + 0.15), yaw=Y),
        Action(op="move_to", object="cab.slot", offset=(0, 0, C / 2 + 0.02), yaw=Y, duration=0.6),
        Action(op="open", duration=0.3), Action(op="move_to", object="cab.slot", offset=(0, 0, 0.2), yaw=Y, duration=0.6),
        *grab(), Action(op="move_to", object="cab.closed", duration=1.5), *leave(),
    ])


def test_drawer_open_put_inside_close(tmp_path: Path):
    goals = [Goal(type="inside", obj="cube", target="cab"), Goal(type="closed", obj="cab")]
    r = run(Build(objects=_drawer_scene(), plan=_drawer_plan(), rationale="drawer task"), goals, tmp_path)
    assert r.success, (r.goals, r.events)
    assert any("grasped cab handle" in e for e in r.events)
    assert any("drawer is 1" in e and "cm open" in e for e in r.events)  # pulled ~10-12 cm


def test_knock_over_tower(tmp_path: Path):
    tower = [_obj(n, (0.5, 0.0, C / 2 + i * C)) for i, n in enumerate(("b1", "b2", "b3"))]
    plan = MotionPlan(actions=[Action(op="close", duration=0.3),
                               Action(op="move_to", xyz=(0.5, 0.11, 2.5 * C), duration=1.2),
                               Action(op="move_to", xyz=(0.5, -0.15, 2.5 * C), duration=0.45)])
    r = run(Build(objects=tower, plan=plan, rationale="knock it over"), [Goal(type="toppled", obj="b3")], tmp_path)
    assert r.success, (r.goals, r.events)
    idle = run(Build(objects=tower, plan=MotionPlan(actions=[Action(op="wait", duration=0.5)]), rationale="do nothing"),
               [Goal(type="toppled", obj="b3")], tmp_path / "idle")
    assert not idle.success


def test_stress_test_separates_good_and_bad_plans():
    from spec2sim.sim.runner import stress_test
    good = stress_test(Build(objects=_cubes(), plan=_stack_plan(C + 0.003), rationale="reference"), ON, trials=6)
    bad = stress_test(Build(objects=_cubes(), plan=_stack_plan(0.18, place_dx=0.03), rationale="drops"), ON, trials=6)
    assert good["trials"] == 6 and good["passed"] == 6
    assert bad["passed"] == 0


def test_dataset_lerobot_layout(tmp_path: Path):
    import json

    import pyarrow.parquet as pq

    from spec2sim.sim.dataset import FPS, generate
    res = generate(Build(objects=_cubes(), plan=_stack_plan(C + 0.003), rationale="reference"), ON,
                   "stack a on b", tmp_path / "ds", episodes=2, images=False)
    assert res["episodes"] == 2 and Path(res["zip"]).exists()
    info = json.loads((tmp_path / "ds/meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1" and info["total_episodes"] == 2 and info["fps"] == FPS
    t = pq.read_table(tmp_path / "ds/data/chunk-000/episode_000000.parquet")
    assert {"observation.state", "action", "observation.environment_state", "timestamp", "frame_index"} <= set(t.column_names)
    ts = t.column("timestamp").to_pylist()
    assert abs(ts[1] - ts[0] - 1 / FPS) < 1e-6  # LeRobot checks exact 1/fps spacing
    assert len(t.column("observation.state")[0]) == 9 and len(t.column("action")[0]) == 8


def test_toppled_counts_falls_after_stacking(tmp_path: Path):
    # a is stacked on b (it rests up there), then swept off: 'toppled' must see that fall,
    # even though a ends no lower than where it started on the floor.
    z = C + C / 2  # a's center when stacked
    knock = [Action(op="close", duration=0.3),
             Action(op="move_to", xyz=(0.5, -0.03, 0.2), duration=0.8),
             Action(op="move_to", xyz=(0.5, -0.03, z), duration=0.6),
             Action(op="move_to", xyz=(0.5, -0.32, z), duration=0.45)]
    plan = MotionPlan(actions=[*_stack_plan(C + 0.003).actions, *knock])
    r = run(Build(objects=_cubes(), plan=plan, rationale="stack then knock"), [Goal(type="toppled", obj="a")], tmp_path)
    assert r.success, (r.goals, r.events)
    stacked = run(Build(objects=_cubes(), plan=_stack_plan(C + 0.003), rationale="only stack"),
                  [Goal(type="toppled", obj="a")], tmp_path / "s")
    assert not stacked.success
