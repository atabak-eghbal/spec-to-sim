"""Policy test harness for the scene export: the files a robotics team needs to run its own policy.

The export zip gets `spec2sim_env.py` (a gym-like env + the goal predicates, depending only on
mujoco and numpy), `evaluate.py` (N seeded episodes -> per-goal pass table + results.json) and
`variations.json` (the randomization ranges). The two scripts live in `harness_assets/` as real,
runnable files; this module copies them and rewrites the env's constants block from the repo's own
constants (metrics.py, scene.py, controller.py, runner.STRESS), so the tolerances can never drift.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from spec2sim.schemas import REACH_R, REACH_X, REACH_Y
from spec2sim.sim import metrics, scene
from spec2sim.sim.controller import HOME_Q, MAX_GRASP_WIDTH, MIN_FINGER_GAP, ONE_PAD_SLACK
from spec2sim.sim.runner import STRESS

ASSETS = Path(__file__).parent / "harness_assets"
_BLOCK = re.compile(r"(# BEGIN GENERATED CONSTANTS[^\n]*\n).*?(# END GENERATED CONSTANTS)", re.DOTALL)


def variations() -> dict:
    """The stress test's randomization ranges (what Spec2SimEnv.reset applies)."""
    return {"pos_jitter_m": STRESS["pos_jitter"], "yaw_jitter_rad": STRESS["yaw_jitter"],
            "mass_scale": STRESS["mass_scale"], "friction_scale": STRESS["friction_scale"]}


def _constants() -> str:
    c = {"ON_Z_TOL": metrics.ON_Z_TOL, "AT_TOL": metrics.AT_TOL, "LIFT_MIN": metrics.LIFT_MIN,
         "SIDE_MARGIN": metrics.SIDE_MARGIN, "UPRIGHT_MAX_TILT": metrics.UPRIGHT_MAX_TILT,
         "TOPPLE_TILT": metrics.TOPPLE_TILT, "BIN_WALL": scene.BIN_WALL, "OPENED_MIN": scene.OPENED_MIN,
         "CLOSED_MAX": scene.CLOSED_MAX, "MAX_GRASP_WIDTH": MAX_GRASP_WIDTH, "ONE_PAD_SLACK": ONE_PAD_SLACK,
         "MIN_FINGER_GAP": MIN_FINGER_GAP, "REACH_X": REACH_X, "REACH_Y": REACH_Y, "REACH_R": REACH_R,
         "HOME_Q": tuple(float(x) for x in HOME_Q), "TCP_OFFSET": scene.TCP_OFFSET, "VARIATIONS": variations()}
    return "".join(f"{k} = {v!r}\n" for k, v in c.items())


def env_source() -> str:
    """spec2sim_env.py with its constants block regenerated from the repo."""
    src = (ASSETS / "spec2sim_env.py").read_text()
    return _BLOCK.sub(lambda m: m.group(1) + _constants() + m.group(2), src)


def harness_files() -> dict[str, str]:
    """Zip member name -> text, for the harness part of an export."""
    return {"spec2sim_env.py": env_source(), "evaluate.py": (ASSETS / "evaluate.py").read_text(),
            "variations.json": json.dumps(variations(), indent=2)}
