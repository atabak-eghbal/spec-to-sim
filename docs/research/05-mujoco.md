# MuJoCo stack

## MuJoCo
- Docs: https://mujoco.readthedocs.io/
- Python bindings: `pip install mujoco`; `mujoco.Renderer(model, height, width)` for offscreen frames.
- Headless: set `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU, slower) before importing mujoco.

## MuJoCo Menagerie (robot models)
- https://github.com/google-deepmind/mujoco_menagerie
- 📌 Use `franka_emika_panda` first (`scene.xml` + `panda.xml`); `universal_robots_ur5e` optional.
- 📌 LLM never writes robot XML; we merge an LLM-written *objects-only* fragment into the Menagerie scene.

## mink (differential IK for MuJoCo)
- https://github.com/kevinzakka/mink
- 📌 Converts our waypoint targets (xyz + orientation) into joint commands.

## Known pitfalls
- Physical grasping in sim is fiddly (friction, contact params). Fallback: weld/attach when gripper is closed within ~2 cm of the object; state it in the README.
- Keep objects simple primitives (box, cylinder, sphere) with real dimensions from Tavily.
- Validate the merged XML by actually loading it (`mujoco.MjModel.from_xml_string`) and feed load errors back to the Builder.
