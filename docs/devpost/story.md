# Devpost: About the project

## Inspiration
Testing a robot skill in simulation starts with a scene, and someone has to build it by hand: measure the objects, write the scene file, place everything where the arm can reach, write the success checks, then debug why the cube fell off. That's days of work per scene, before a single policy is tested. We wanted to type the task and get a working, checked scene, one that tells you whether a plan really works or just got lucky.

## What it does
You type a task, for example *"Put the soda can to the left of the Rubik's cube and keep it upright."*

1. **Plan:** NVIDIA Nemotron 3 Super turns it into objects and goals that code can check.
2. **Ground:** Tavily finds each object's real size and mass, cross-checked across web pages (and read in full with `/extract` when search snippets aren't enough), plus friction and bounciness from engineering tables and the Franka gripper's limits from its datasheet. Tasks the gripper physically can't do are refused before anything is built, with the sources.
3. **Build:** Super places the objects and writes the plan as skills (pick, place, push). Our controller turns them into motion; the model never writes code or robot XML.
4. **Simulate and check:** MuJoCo runs a Franka Panda (from MuJoCo Menagerie), and code checks every goal. A model never decides pass or fail.
5. **Repair:** if a goal fails, Nemotron 3.5 Super VL explains what went wrong from the frames, Super writes three repair candidates in parallel, the simulator tests each one in seconds, and the sturdiest runs next.
6. **Stress-test:** every success is re-run 16 times with objects moved, heavier, more slippery, and the robot's view slightly off.
7. **Use it:** export the scene with a policy test harness (`evaluate.py` scores your own policy on randomized versions of the scene), or generate LeRobot-format training data.

Everything streams live in the web app: which model ran each step, its time, tokens and cost, and a 3D replay of every attempt in the browser, with a before/after view of the failure and the fix. You can drag objects or edit their sizes and masses and re-run the same plan, to try to break it.

## How we built it
- **Models on Nebius Token Factory:** Nemotron 3 Super for planning, building and repair; Nemotron 3 Nano for extracting numbers from web pages; Nemotron 3.5 Super VL for looking at the scene and at failed runs. One OpenAI-compatible client, strict JSON-schema output validated with Pydantic, reasoning switched on or off per step, and the cost of every call logged and shown in the UI.
- **Tavily** for every real-world number: advanced search, `/extract` for full pages, quotes that must appear word for word in what Tavily returned, a context check (the friction quote must be about this material on a dry wood table), cross-source agreement, and size-vs-weight sanity checks.
- **MuJoCo** with the Franka Panda from MuJoCo Menagerie, our own IK controller and skill library, goal predicates modelled on NVIDIA RoboLab's vocabulary, and a three.js viewer that replays the physics in the browser.
- **LangSmith** traces every call. A FastAPI app streams the run over server-sent events. The public demo runs on a Hugging Face Space.
- **Evaluation:** a frozen set of 15 held-out prompts plus 11 development prompts, 3 runs each, 104 runs per round, with 95 % confidence intervals.

## Challenges we ran into
- **Honest grasping.** Physical grasping in simulation is fragile, so we use a weld once the object is truly between the fingers. Reviewers found an early version would "grasp" a cube the fingers only touched from above; we fixed it, added the probe as a test, and now report every stress test with the assist off too.
- **Bad web data.** One farm guide's 8.9 cm apple could refuse a whole task, and a soda can came back measured lying down. We added median voting across sources, verified agreement votes, orientation checks and a size-vs-weight test.
- **Measuring what helps.** Our own evaluation showed the vision Critic made no measurable difference, and escalating to the 550B Ultra model fixed only 4 of the 10 runs sent to it. We replaced Ultra with three cheap Super repairs checked by the simulator, which did as well for less, and we say plainly that the vision model is there to explain, not to win.
- **Deployment.** Our Nebius AI Cloud account couldn't create registries or endpoints (PermissionDenied), so the demo runs on a free Hugging Face Space, while every model call still goes through Token Factory.

## Accomplishments that we're proud of
- **42 of 45 runs pass on 15 held-out prompts**, at a median model cost of about half a cent per task.
- **Test-time compute plus a verifier:** three parallel Super repairs, judged by physics, matched a model four times larger.
- **Reasoning pays for itself:** with Super's reasoning off, success fell from 11/11 to 6/11, and tasks cost twice as much because of the extra repairs.
- **A policy test you can take home:** the exported harness needs only MuJoCo and NumPy. A small policy we trained on our generated demos matches the scripted plan on a simple stack but drops to 4/16 placing a tennis ball, which is exactly the kind of weakness the test is meant to find.
- **Every claim checked:** we ran five rounds of independent reviews and corrected every number they could not verify.

## What we learned
- A cheap model plus a fast, exact verifier beats a bigger model guessing alone.
- Never let a model decide pass or fail; let it explain.
- Web grounding is only as good as its checks: quotes must exist, sources must agree, and the context must match.
- Measure every component. Two of ours (Ultra escalation and the vision Critic as a repair engine) didn't earn their place, and the data said so.

## What's next for Spec-to-Sim
- Physical grasping without the weld, using stronger fingers and contact tuning.
- More robots from MuJoCo Menagerie (UR5e first) and side grasps.
- Mesh objects with sizes from the web, instead of boxes, cylinders and spheres.
- Export to Isaac Lab and USD, and a regression suite that scores a policy on every saved task in CI.
- A Nebius Serverless Job that generates thousands of verified demonstrations overnight.
