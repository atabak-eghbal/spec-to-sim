# Spec-to-Sim — Project Brief & Working Rules

> Read this file at the start of every session. It is the source of truth for scope,
> rules, and priorities. If a request conflicts with it, flag the conflict before acting.

## 1. The project in one paragraph
**Spec-to-Sim** turns a plain-English robot task ("a Franka arm stacks the red cube on the blue one")
into a running MuJoCo simulation. A cascade of NVIDIA Nemotron models on **Nebius Token Factory**
plans the task, grounds real-world object/robot facts with **Tavily**, builds the scene + motion plan,
runs it, then a **vision model watches the rendered video**, diagnoses failures, and the loop repairs
itself until the task succeeds. Every step is traced in **LangSmith** and shown live in the UI.

- **Hackathon:** Nebius x NVIDIA Global AI Hackathon (Devpost)
- **Track:** Physical AI (no hardware → video must show ≥1 min of "key application modules in action")
- **Deadline:** **Fri Oct 30, 2026, 10:00 AM PT**. Internal target: submitted by **08:00 AM PT** (2h buffer).
- **Entrant:** solo, Vancouver, BC resident.

## 2. Non-negotiables (from the official rules)
- [ ] Runtime calls to Token Factory API (`https://api.tokenfactory.nebius.com/v1/`, OpenAI-compatible).
- [ ] At least one NVIDIA open model (we use several Nemotron models).
- [ ] Public GitHub repo, OSI license file (MIT) visible in the About section.
- [ ] README: setup instructions + **explicit section on Nemotron usage and where Token Factory accelerated the workflow** + other Nebius services used.
- [ ] Public working demo URL (free access for judges until Dec 15, 2026).
- [ ] YouTube demo video < 3 min, public/unlisted, no copyrighted music, no third-party trademarks beyond what's needed.
- [ ] Track selected: Physical AI. Tavily question answered "Yes".
- [ ] Feedback questions answered thoughtfully (eligible for "Most Valuable Feedback").

## 3. Prize strategy (what every decision should serve)
1. **Physical AI Track** (or an Overall award) + **Best Use of Tavily** bonus ($3K) + **Most Valuable Feedback**.
2. Judging criteria are equally weighted — optimise all four:
   - **Technological Implementation** → real cascade, Serverless Endpoint + Serverless Job, LangSmith traces, deterministic success checks.
   - **Design** → a complete product with a polished UI, not a script.
   - **Potential Impact** → robotics teams spend days hand-building sim scenes; we make that minutes. Be specific.
   - **Quality of Idea** → a VLM that *watches* its own simulation and fixes it. Visual "wow".
3. **Model cascade is a feature, show it.** Every step in the UI shows which model ran it, latency, tokens, and $ cost.
4. Competition is large (>24k registrants). Polish beats breadth.

## 4. Architecture (see docs/architecture.md for the diagram)
| Step | Role | Model / service | Why this one |
|---|---|---|---|
| 1 | **Planner**: English → structured task spec (JSON) | Nemotron 3 Super | Good tool-calling/JSON, cheap |
| 2 | **Grounder**: real object/robot facts with source links | **Tavily** search + Nemotron Nano to extract numbers | Real dimensions/masses; runtime Tavily call |
| 3 | **Builder**: scene objects (MJCF fragment) + motion plan (waypoint JSON) | Nemotron 3 Super | Structured output |
| 4 | **Runner**: run MuJoCo headless, render MP4, compute success metrics | Our code (inside Nebius Serverless Endpoint container) | Deterministic |
| 5 | **Critic**: watch frames + metrics, diagnose failure | **Nemotron 3.5 Super VL** | Image/video understanding |
| 6 | **Fixer** (only after a failed retry): deep repair | **Nemotron 3 Ultra** | Escalate only when needed → saves credits |
| — | Batch eval of N prompts → success-rate table | **Nebius Serverless Job** | Shows rigor + second Nebius service |
| — | Tracing | **LangSmith** (`wrap_openai` + `@traceable`) | Architecture visible to judges |

Loop: max **4 iterations**. Attempt 1–2 repaired by Super; attempt 3+ escalates to Ultra.

### Key design decisions (do not change without discussing)
- **Robots come from MuJoCo Menagerie** (Franka Panda first; UR5e optional). The LLM never writes robot XML — too error-prone.
- **The LLM outputs data, not code**: an MJCF *objects fragment* + a *waypoint/action JSON* validated with Pydantic, executed by our own fixed controller. No arbitrary LLM-generated Python is executed.
- **Success is measured in code** (object positions vs goal, tolerances), the VLM explains *why* it failed. Never trust the VLM alone for pass/fail.
- **Grasp fallback**: if physical grasping is flaky, use a weld/attach constraint when the gripper closes within a threshold of the object. Document it honestly in the README.
- **Headless rendering**: `MUJOCO_GL=egl` (GPU) or `osmesa` (CPU). Short clips: ≤6 s, 480p, 30 fps.

## 5. Budget guardrails
| Credit | Amount | Rule |
|---|---|---|
| Token Factory | $50 | Use Nano/Super by default; Ultra only on escalation. Log cost per call. Hard stop if dev spend > $20. |
| Tavily | $25 | Cache results on disk by query hash. ≤ 3 searches per run. |
| LangSmith | $100 | Trace everything in dev and prod. |
| Nebius AI Cloud | (check) | Serverless is billed per second while running. **Stop dev endpoints when idle.** Smallest CPU preset that works. |
| Toloka / Tandem | $50 / $50 | Not required. Ignore unless a natural use appears. |

Before any Ultra call or any Nebius resource creation that costs money, say so in one line.

## 6. Verify before assuming (things that change)
- Exact Token Factory model IDs: run `GET /v1/models` and record them in `.env` and `src/spec2sim/config.py`. Expected IDs (from third-party listings, **unverified**):
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`, `nvidia/nemotron-3-super-120b-a12b`, `nvidia/Nemotron-3-Ultra-550b-a55b`, `nvidia/Nemotron-3.5-Super-VL`.
- Whether the VL model accepts video directly or only image frames on Token Factory → default to sending 6–8 sampled frames as images.
- Nebius Serverless CPU presets and whether endpoints can scale to zero.

## 7. Repo conventions
- Python 3.11, `uv` for env, `ruff` for lint, `pytest` for tests.
- Secrets only in `.env` (never committed). `.env.example` lists every variable.
- Small commits with clear messages. `main` is always demo-able.
- Every module has a docstring explaining its role in the pipeline.
- `docs/research/` holds sources; cite them in README where relevant.
- `docs/devpost/` holds drafts of every Devpost field.

## 8. Working agreement with Claude
- Claude Code (VS Code) builds and runs the code. The claude.ai Project chat handles research, writing, and Devpost drafts.
- Claude does not create accounts, type API keys into websites, or submit the Devpost form; Austin does those.
- Prefer working end-to-end slices over perfect components. Demo-ability first.
- When something will take longer than planned, say so immediately and propose a cut.
