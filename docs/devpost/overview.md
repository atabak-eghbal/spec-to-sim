# Devpost: overview fields

## Project name (≤ 60 chars)
Spec-to-Sim: one sentence to a checked robot simulation

## Elevator pitch (≤ 200 chars)
Type a robot task in plain English. Get a MuJoCo scene grounded in real web facts, goals checked in code, and when a plan fails, Nemotron proposes 3 fixes and the simulator picks one.

(183 characters)

## Links
- Try it out: https://austintransformer-spec-to-sim.hf.space
- Code: https://github.com/atabak-eghbal/spec-to-sim
- Video: https://youtu.be/HfrEUVrve4M

## Built with (tags)
nvidia-nemotron, nebius-token-factory, tavily, mujoco, mujoco-menagerie, langsmith, python, fastapi, pydantic, three.js, numpy, lerobot, hugging-face-spaces

## Gallery images (3:2, in docs/devpost/images/)
1. `1-fail-then-fixed.jpg`: the app: a soda can falls over on attempt 1 and stands on attempt 2, with the real-world facts beside it.
2. `2-repair-loop.jpg`: three repair candidates tested in the simulator; the sturdiest runs.
3. `3-tavily-facts.jpg`: real sizes from the web, the model's own guess under each, 4/4 sources agreeing.
4. `4-stress-and-policy.jpg`: 16 perturbed re-runs as tiles, and a learned policy's report card.
5. `5-refused.jpg`: a grapefruit refused before building: 8.2 cm against the gripper's 8 cm.
