"""Simulation package: scene assembly, controller, success metrics, rendering, runner.

The headless GL backend must be chosen before mujoco is first imported, so it is set here (the
package init runs before any submodule). macOS only supports CGL, whatever .env says.
"""
import os
import sys

if sys.platform == "darwin":
    os.environ["MUJOCO_GL"] = "cgl"
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
