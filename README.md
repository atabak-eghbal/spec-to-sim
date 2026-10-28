# Spec-to-Sim 🤖➡️🎬

> **Describe a robot task in plain English. Watch it become a working simulation — and watch the AI fix its own mistakes.**

<!-- TODO: hero GIF of the UI (assets/hero.gif) -->

**Track:** Physical AI · **Built for:** Nebius x NVIDIA Global AI Hackathon 2026
**Live demo:** _TBD (Nebius Serverless Endpoint)_ · **Video:** _TBD (YouTube)_

---

## What it does
You type: *"A Franka arm picks up a coffee mug and places it on a coaster."*

Spec-to-Sim then:
1. **Plans** the task into a structured spec (Nemotron 3 Super)
2. **Grounds** it in reality — looks up real object dimensions and masses on the web (**Tavily** + Nemotron Nano)
3. **Builds** a MuJoCo scene and a motion plan (Nemotron 3 Super)
4. **Runs** the simulation and renders a video
5. **Watches** the video with a vision-language model and diagnoses what went wrong (Nemotron 3.5 Super VL)
6. **Repairs** the plan and retries — escalating to Nemotron 3 Ultra only when the easy fixes fail

Every step streams live into the UI with the model that ran it, its latency, tokens and cost.

## Why it matters
<!-- TODO: concrete problem statement: hours spent hand-authoring sim scenes; who benefits -->

## Architecture
See [docs/architecture.md](docs/architecture.md).

## How we use NVIDIA Nemotron + Nebius Token Factory
| Step | Model | Why this size |
|---|---|---|
| Planner / Builder | Nemotron 3 Super | <!-- TODO --> |
| Fact extraction | Nemotron 3 Nano | <!-- TODO --> |
| Critic | Nemotron 3.5 Super VL | <!-- TODO --> |
| Fixer (escalation) | Nemotron 3 Ultra | <!-- TODO --> |

**Where Token Factory accelerated the workflow:** <!-- TODO: one OpenAI-compatible endpoint for 4 model sizes, swap per step, cost data -->

**Other Nebius services:** Serverless Endpoint (hosts the app + simulator), Serverless Job (batch evaluation).

## Results
<!-- TODO: eval table from docs/eval.md -->

## Quickstart
```bash
git clone https://github.com/atabak-eghbal/spec-to-sim && cd spec-to-sim
cp .env.example .env        # fill in your keys
uv sync
./scripts/fetch_menagerie.sh                 # Franka Panda model (MuJoCo Menagerie, pinned commit)
uv run python scripts/smoke_render.py        # sanity check -> runs/smoke/panda_smoke.mp4
uv run python scripts/list_models.py         # sanity check Token Factory key + model IDs
uv run python -m spec2sim.cli "a Franka arm stacks the red cube on the blue cube"
uv run uvicorn app.main:app --reload   # UI at http://localhost:8000
```

## Honest limitations
<!-- TODO: e.g. grasp attach fallback, Menagerie robots only -->

## License
MIT — see [LICENSE](LICENSE). Robot models come from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Apache-2.0), fetched at setup time.
