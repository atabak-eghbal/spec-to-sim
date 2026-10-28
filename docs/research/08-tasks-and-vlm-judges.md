# Harder tasks and VLM judges: what Isaac Lab and recent papers do, and what we took

Collected Oct 28, 2026 to choose tasks beyond pick-and-place and to back our design choices.

## 1. NVIDIA Isaac Lab: the Franka task family
- Environment catalog: https://isaac-sim.github.io/IsaacLab/ (Environments page; `docs/source/isaaclab/environments.rst` in the repo)
- Isaac Lab's standard Franka manipulation tasks: **reach**, **lift a cube**, **stack cubes** (also
  used by Isaac Lab Mimic for synthetic demos), **open a cabinet drawer** (Isaac-Open-Drawer-Franka),
  and the **Factory/AutoMate** contact-rich assembly tasks (peg insertion, gear meshing, nut threading).
  Humanoid and dexterous-hand tasks (in-hand cube reorientation, nut pouring with GR1) are out of scope for one arm.
- 📌 We now cover lift, stack, place-in-container, **drawer opening/closing with objects inside**
  (articulated object, slide joint, handle grasp) and **knock-over/push** (contact-rich, no grasp).
  Assembly with tight tolerances is out of scope: it needs real contact physics, not grasp assist.

## 2. Isaac Lab-Arena: sensitivity analysis
- https://isaac-sim.github.io/IsaacLab-Arena/ ("Metrics", "Sensitivity Analysis" pages)
- Arena evaluates policies under controlled variations of the environment.
- 📌 Our **stress test** does the same for a generated plan. After a success, the plan is re-run
  16 times with each object up to 1.5 cm off and ~9° turned, ±20 % mass and ±30 % friction.
  Physics only, no model calls, a few seconds. It separates luck from robustness: the tennis-ball
  plan that failed on attempt 1 passes only 11/20 perturbed runs, and the repaired plan passes 20/20.

## 3. Papers on generating tasks and verifying them with a VLM
| Work | What it does | 📌 What it means for us |
|---|---|---|
| AnyTask (arXiv 2512.17853, Dec 2025) https://arxiv.org/abs/2512.17853 | Foundation models design tasks (incl. **drawer opening, contact-rich pushing, long-horizon**) and synthesize demos in parallel GPU sim; VLM-in-the-loop refinement | Validates our task picks. Their generator targets data for policy training; ours targets one working, checkable scene quickly |
| V-CAGE (arXiv 2601.15164, Jan 2026) https://arxiv.org/abs/2601.15164 | LLM decomposes long-horizon goals; a VLM visual critic checks each subtask and rejects "silent failures" | Same critic idea; we also give the critic the code's measurements as ground truth |
| RoboTwin 2.0 (arXiv 2506.18088) https://arxiv.org/abs/2506.18088 | Multimodal LLM + **simulation-in-the-loop verification** to generate bimanual task code | Ours generates data, not code (D01), and verifies with predicates |
| GenManip (arXiv 2506.10966) https://arxiv.org/abs/2506.10966 | LLM-driven, task-oriented scene graphs over ~10K assets, incl. articulated objects | Asset-library approach; we use real-world sizes from web search instead of an asset library |
| RoboPearls (ICCV 2025) | VLM diagnoses failures and proposes targeted simulation fixes | Closest to our Critic → repair loop |
| GRS (CVPR 2025 workshop, arXiv 2410.15536) | Generates sim tasks from real images with a VLM + LLM refinement loop | Possible extension: start from a photo of a real table |

## 4. How far can a VLM be trusted as a judge?
| Work | Finding | 📌 Our design |
|---|---|---|
| **FailBench** (arXiv 2609.03611, Sep 2026) https://arxiv.org/abs/2609.03611 | 13 VLM success detectors on 2,197 real attempts: best **0.77** balanced accuracy, **below 0.60 on contact-heavy tasks**, and a **bias toward saying "success" when unsure**. Cropping the relevant region helps (+2.4 pts) | Exactly why pass/fail is decided in code (D02) and the VLM only explains. The Critic gets the measured failures first ("evidence first", D08) |
| PhysBench (ICLR 2025) https://arxiv.org/abs/2501.16411 | 75 VLMs: good at common sense, weak at physical-world understanding (GPT-4o ~49 % vs humans ~96 %) | The Critic is told the measured numbers instead of being asked to judge physics from pixels |
| OSCAR world model (arXiv 2606.04463) https://arxiv.org/abs/2606.04463 | A frontier VLM judge matches human success labels on 78/100 clips | ~1 in 5 wrong is acceptable for explanations, not for verdicts |
| Robot Critics that Sweat the Small Stuff (arXiv 2606.21572) | Fine-tuned progress critic for subtle failures | Same conclusion: general VLMs miss fine-grained failures |

## 5. Do we need GPUs? (Oct 28 decision)
- **Original plan:** a GPU Serverless Endpoint so MuJoCo could render video with EGL.
- **Now not needed:** the browser draws the 3D replay (D28), and the server only simulates
  (2.9 s for the test scene on 2 CPUs). The models run on Token Factory, not on our hardware.
- **Where a GPU would add something** (not planned, list for the README's "future work"):
  1. **MuJoCo Warp / NVIDIA Newton**: thousands of parallel stress-test variants instead of 16.
     Our scenes are plain MJCF, so they would load unchanged.
  2. **Self-hosting Cosmos Reason 2** on a Nebius GPU as a second, physics-tuned critic
     (it is not on Token Factory).
  3. **Learning a policy** (RL / imitation) in the generated scene, as AnyTask does.
  Each costs GPU hours and days of work. For this hackathon, a CPU endpoint is the right call.
