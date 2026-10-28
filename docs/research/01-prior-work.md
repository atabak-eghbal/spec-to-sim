# Prior Work — LLMs that build robot simulations

How we differ from all of these: **a vision-language model watches the rendered rollout and
drives the repair loop**, the scene is **grounded in real-world facts via web search**, and the
whole thing is a **product with a UI**, not a research pipeline.

## Eureka (NVIDIA, 2023)
- https://arxiv.org/abs/2310.12931 · https://eureka-research.github.io/
- LLM writes reward-function code, trains in simulation, reads training stats, rewrites the reward.
- 📌 We borrow the *generate → run → reflect → refine* loop, but reflect on **video** instead of scalar stats.

## RoboGen (2023)
- https://arxiv.org/abs/2311.01455
- LLM proposes tasks, generates scenes (assets + layout), then learns skills to solve them.
- 📌 Validates "LLM populates scenes, robot assets come from a library" — same split we use.

## GenSim (2023)
- https://arxiv.org/abs/2310.01361
- LLM writes new simulation tasks as code for a tabletop benchmark.
- 📌 Shows code-generated tasks often fail to run → we generate **validated data (JSON/MJCF fragment)**, not code.

## Text2Reward (2023)
- https://arxiv.org/abs/2309.11489
- Language goal → dense reward code, iterated with human feedback.
- 📌 Supports measuring success with explicit, checkable predicates.

## Code as Policies (Google, 2022)
- https://arxiv.org/abs/2209.07753
- LLM writes robot policy code calling perception/control primitives.
- 📌 Our action vocabulary (`move_to`, `open_gripper`, `close_gripper`, `wait`) is a tiny, safe version of this.

## Pitch line for Devpost
> "Eureka showed LLMs can iterate on simulation rewards from numbers. Spec-to-Sim lets a
> vision model iterate on the simulation itself — by watching it."
