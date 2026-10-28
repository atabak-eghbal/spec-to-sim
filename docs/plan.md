# Build Plan (all times Pacific)

Deadline: **Fri Oct 30, 10:00 AM PT**. Target submit: **08:00 AM PT**.

## Tue Oct 27 (evening) — Setup  ⏱ ~2h
- [x] Create accounts / API keys: Token Factory, Tavily, LangSmith, Nebius AI Cloud (CLI profile `spec2sim`, project `project-e00yz5abpr00fd1bg6p5he`)
- [x] GitHub repo `spec-to-sim` created, MIT detected — kept **private** until submission day; flip to public before Devpost
- [x] `uv sync`, fill `.env`
- [x] Smoke test: list Token Factory models, record exact Nemotron IDs (all 4 confirmed; `src/spec2sim/config.py`)
- [x] Live API smoke test: Nano, Super, Super VL (image), Tavily, LangSmith trace (`scripts/smoke_apis.py`)
- [x] Smoke test: load Franka Panda from MuJoCo Menagerie, render 3 s MP4 headless (`scripts/smoke_render.py`)

## Wed Oct 28 (morning) — Core loop on CLI  ⏱ ~5h
- [x] Pydantic schemas: TaskSpec, GroundedFacts, Build (objects + MotionPlan), RunResult, Critique
- [x] Planner (Super) → TaskSpec
- [x] Grounder (Tavily + Nano) → object facts with sources (+ disk cache)
- [x] Builder (Super) → objects (data → MJCF in code) + action plan
- [x] Runner: own DLS IK (mink not needed), weld grasp assist w/ 8 cm width limit, success metrics, MP4 + contact sheet
- [x] Critic (Super VL, thinking off ≈3 s) on one 8-frame contact sheet → Critique
- [x] Loop with escalation to Ultra; LangSmith tracing on every call (Ultra path not yet exercised live)

## Wed Oct 28 (afternoon/evening) — Product  ⏱ ~5h
- [ ] FastAPI backend + Server-Sent Events streaming each step
- [ ] Single-page UI: prompt box, live step timeline with model badges (Nano/Super/Ultra/VL), cost/tokens, per-iteration video, Tavily sources, LangSmith trace link
- [ ] 4–5 curated example prompts that reliably succeed (one that needs a repair loop!)

## Wed Oct 28 (night) — Nebius deployment  ⏱ ~3h
- [ ] Dockerfile (osmesa/egl), push image
- [ ] Deploy as **Nebius Serverless Endpoint** (public URL = demo link)
- [ ] Batch eval as **Nebius Serverless Job** → results table committed to `docs/eval.md`

## Thu Oct 29 — Polish, docs, video  ⏱ ~6h
- [ ] README final (Nemotron + Token Factory section, architecture diagram, eval table)
- [ ] Screenshots / GIF into `assets/`
- [ ] Record demo video (≤ 2:45), upload to YouTube
- [ ] Draft all Devpost fields in `docs/devpost/`

## Fri Oct 30 (morning) — Submit
- [ ] **Make GitHub repo public** (`gh repo edit --visibility public --accept-visibility-change-consequences`)
- [ ] Fill Devpost, double-check every required field, submit by 08:00 PT
- [ ] Leave endpoint running (judging Dec 1–15) — see cost note in docs/research/02-nebius-platform.md

## Cut list (if behind schedule, cut in this order)
1. Batch-eval Serverless Job → run eval locally, keep Job for one small run
2. UR5e second robot
3. Ultra escalation → keep only if one example needs it
