"""The exported policy harness (spec2sim_env.py) must judge goals exactly like the repo's metrics.py."""
import importlib.util
import json

import mujoco
import numpy as np
import pytest

from spec2sim.harness import harness_files
from spec2sim.schemas import Goal, SceneObject
from spec2sim.sim.metrics import check_goal
from spec2sim.sim.scene import OBJ_PREFIX, PANDA_SCENE, build_scene, objects_mjcf

C = 0.05
OBJS = [SceneObject(name="a", shape="box", size=(C, C, C), mass=0.1, rgba=(1, 0, 0, 1), pos=(0.5, 0.15, C / 2)),
        SceneObject(name="b", shape="box", size=(C, C, C), mass=0.1, rgba=(0, 0, 1, 1), pos=(0.5, -0.15, C / 2)),
        SceneObject(name="bin", shape="bin", size=(0.16, 0.16, 0.08), mass=0.5, rgba=(.5, .5, .5, 1),
                    pos=(0.4, 0.0, 0.04))]
GOALS = [Goal(type="on", obj="a", target="b"), Goal(type="inside", obj="a", target="bin"),
         Goal(type="left_of", obj="a", target="b"), Goal(type="lifted", obj="a"), Goal(type="upright", obj="a"),
         Goal(type="at", obj="a", xyz=(0.5, -0.15, 0.075)), Goal(type="toppled", obj="a")]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """Write the harness files exactly as the export does, then load them like a user would."""
    d = tmp_path_factory.mktemp("export")
    (d / "scene_objects.xml").write_text(objects_mjcf(OBJS))
    (d / "task.json").write_text(json.dumps({
        "objects": [{"name": o.name, "shape": o.shape, "size_m": o.size, "start_pos_m": o.pos,
                     "fixed": o.shape == "bin"} for o in OBJS],
        "goals": [g.model_dump(exclude_none=True) for g in GOALS]}))
    for name, text in harness_files().items():
        (d / name).write_text(text)
    spec = importlib.util.spec_from_file_location("exported_env", d / "spec2sim_env.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Spec2SimEnv(str(d), menagerie=str(PANDA_SCENE))


def _put(env, ref_scene, ref_data, name, xyz):
    """Place object `name` at xyz (upright) in both the harness env and the repo's scene."""
    a = env.free[name]
    env.d.qpos[a:a + 7] = [*xyz, 1, 0, 0, 0]
    mujoco.mj_forward(env.m, env.d)
    j = ref_scene.model.body(OBJ_PREFIX + name).jntadr[0]
    q = ref_scene.model.jnt_qposadr[j]
    ref_data.qpos[q:q + 7] = [*xyz, 1, 0, 0, 0]
    mujoco.mj_forward(ref_scene.model, ref_data)


@pytest.mark.parametrize("pos,on_ok", [((0.5, -0.15, 1.5 * C), True),   # a sits on b
                                       ((0.5, 0.15, C / 2), False)])    # a stays where it started
def test_env_agrees_with_metrics(env, pos, on_ok):
    ref = build_scene(OBJS)
    rd = mujoco.MjData(ref.model)
    env.reset(seed=0)
    _put(env, ref, rd, "b", (0.5, -0.15, C / 2))  # reset() nudged b; put it back so both sides agree
    _put(env, ref, rd, "a", pos)
    env.peaks["a"] = ref_peak = C / 2  # both sides judge 'toppled' against the same resting height
    for g, r in zip(GOALS, env.success()):
        want = check_goal(ref, rd, g, None, {"a": ref_peak})
        assert r["ok"] == want.ok, (g.type, r["detail"], want.detail)
    assert env.success()[0]["ok"] == on_ok


def test_inside_bin_and_reset_is_seeded(env):
    ref = build_scene(OBJS)
    rd = mujoco.MjData(ref.model)
    env.reset(seed=3)
    _put(env, ref, rd, "a", (0.4, 0.0, 0.03))
    got = {r["goal"]["type"]: r["ok"] for r in env.success()}
    assert got["inside"] and check_goal(ref, rd, GOALS[1], None).ok
    o1 = env.reset(seed=7)["objects"]["a"]["pos"].copy()
    o2 = env.reset(seed=7)["objects"]["a"]["pos"].copy()
    assert np.allclose(o1, o2) and np.abs(o1[:2] - [0.5, 0.15]).max() <= 0.04 + 1e-9
    assert np.allclose(env.d.body("task/" + OBJ_PREFIX + "bin").xpos, [0.4, 0.0, 0.04])  # furniture never moves
