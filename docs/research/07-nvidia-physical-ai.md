# NVIDIA physical-AI work we align with (2026)

Spec-to-Sim is a small MuJoCo cousin of what NVIDIA's Isaac teams are building for Isaac Lab.
These sources shaped the October 28 "harder tasks" work (bins, `inside`, spatial goals, guards).

## Isaac Lab-Arena: Agentic Environment Generation (NVIDIA, release 0.3.x)
- Overview: https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.1/pages/concepts/agentic_environment_generation/system_overview.html
- Models + spec quality: https://isaac-sim.github.io/IsaacLab-Arena/release/0.3.1/pages/concepts/agentic_environment_generation/model_selection.html
- Natural-language task → a JSON `ArenaEnvGraphSpec` through strict `response_format` json_schema,
  validated with Pydantic against live asset/relation/task catalogs. Rejected output plus its
  errors go back to the model ("critic feedback"), max 3 iterations.
- Their model table includes `nvidia/nemotron-3-ultra-550b-a55b`: 9/15 specs accepted.
- Their own caveat: "Validation proves admissibility only, not that the result matches the request."
- 📌 Same contract as ours (LLM emits schema-validated data, never code; errors are fed back).
- 📌 **Our step beyond it:** we *run* the spec, check the request's goals in code, and a VLM watches the
  rollout video to explain failures. That checks the request is done, not just that the spec is valid.
- 📌 Same failure modes we hit: invented names (our Planner wrote a goal target `"target': "`),
  non-determinism at low temperature (the same prompt passed once and failed 4×). Our fix mirrors
  theirs: a cross-reference validator on `TaskSpec` (goals must name declared objects) with
  errors fed back for one retry.

## RoboLab (NVlabs, RSS 2026, built on Isaac Lab)
- Paper: https://arxiv.org/abs/2604.09860 · Code (Apache-2.0): https://github.com/NVlabs/RoboLab
- Conditionals doc: https://github.com/NVlabs/RoboLab/blob/main/docs/task_conditionals.md
- 120 tabletop tasks; success = "automated success/failure detection via composable predicates";
  scenes/tasks can be generated from language (`/robolab-scenegen`, `/robolab-taskgen` skills).
- Predicates include `object_on_top` (contact-force cone), `object_in_container` (object
  centroid inside the container's convex hull, tested in the container's own frame, open top),
  `object_left_of / right_of / in_front_of / behind` (robot frame by default), plus `any/all/choose`.
- 📌 We adopted its vocabulary: goal types `on`, `inside`, `left_of`, `right_of`, `in_front_of`,
  `behind`, `upright`. `inside` uses RoboLab's centroid-in-cavity test (in the bin's frame,
  open top). Implemented in `src/spec2sim/sim/metrics.py`.
- 📌 Not adopted (yet): contact-force-cone stability for `on` (we use height + footprint tolerance),
  and `choose`-K logicals (all goals must pass).

## Cosmos Reason (NVIDIA): VLM as physical-plausibility critic
- https://docs.nvidia.com/cosmos/latest/reason1/video_critic.html · https://github.com/nvidia-cosmos/cosmos-reason1
- NVIDIA uses a reasoning VLM to judge whether generated videos obey physics (gravity, object
  permanence, collisions) and rejection-samples bad ones (best-of-N with Cosmos-Predict).
- 📌 Same idea as our Critic (a VLM judges a rollout), except ours watches a *simulated* rollout and
  writes a repair. Cosmos Reason is not on Token Factory, so we use Nemotron 3.5 Super VL. The
  VLM never decides pass/fail; code does.

## Related academic work (2026)
- V-CAGE: Vision-Closed-Loop Agentic Generation Engine for Robotic Manipulation, https://arxiv.org/abs/2604.09036.
  A VLM visual critic filters "silent failures" in generated manipulation data. 📌 Supports our
  "a VLM watches the video" step.
- Robot Critics that Sweat the Small Stuff, https://arxiv.org/abs/2606.21572. A fine-tuned VLM
  progress critic scores predicted outcomes and selects actions (+11% real, +5.9% sim success).
  📌 Evidence that general VLMs miss fine-grained failures. This is why our Critic gets the
  measured goal results as ground truth and must quote evidence first.

## Pitch line
> "NVIDIA's Isaac Lab-Arena turns language into a *valid* environment spec. Spec-to-Sim turns
> language into a *working* simulation: it runs the spec, checks the goals in code using
> RoboLab-style predicates, and a Nemotron vision model watches the video and repairs the plan."
