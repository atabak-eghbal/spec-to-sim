# Demo video plan (target 2:50, hard limit < 3:00)

## Rules this video must meet
- Under 3 minutes, on YouTube (public or unlisted).
- Physical AI track: **at least 1 minute of the key modules in action** (no hardware, so this is the live app).
- No copyrighted music. Use a royalty-free track (YouTube Audio Library) or none.
- No third-party trademarks beyond what's needed. Say "soda can", not a brand; NVIDIA, Nebius, Tavily and LangSmith are named because they are the required or used tools.

## Style
- **Voice:** an AI narrator reading the script below, so the pacing is calm and consistent. Options:
  1. NVIDIA's TTS models on build.nvidia.com (Magpie / Riva). This fits the NVIDIA theme; check availability and terms first.
  2. ElevenLabs (free tier allows commercial use with attribution; check current terms).
  3. Your own voice. Judges often like a real person. A hybrid works too: you for the intro and outro, AI for the walkthrough.
- **Picture:** screen capture of the web app at 1440×900, browser zoom 110%, cursor visible. Waiting time is
  sped up with an on-screen "4× speed" tag, so it's honest about latency.
- **One idea per shot.** Keep captions short and large. End every section on a visual payoff (a green check, a red ✗, a robustness bar).
- Recorded runs replay instantly, so most shots can use them. Only shot 3 needs a live run.

## Shot list and narration (~150 words per minute)

| # | Time | Shot (exact URL or action) | Narration |
|---|---|---|---|
| 1 | 0:00–0:12 | Drawer run replaying in 3D, slow orbit: `/?run=20261028-142918-490a` | "Put the tennis ball and the Rubik's cube in the drawer, and close it. One sentence. Sixty seconds later, a robot does it in a physics simulation, and the result is checked." |
| 2 | 0:12–0:27 | Title card "Spec-to-Sim", then the empty app | "Robotics teams spend days building simulation scenes by hand: measuring objects, writing XML, writing success checks. Spec-to-Sim does it from plain English, using NVIDIA Nemotron models on Nebius Token Factory." |
| 3 | 0:27–1:25 | **Live run** (4× during waits): type "Put the apple and the lime into the fruit basket, and the soda can into the recycling bin", click Simulate. Show the Planner card, then the Grounder table with Tavily source links, the Builder, then the 3D replay. Orbit the camera, then the green goal labels at the end, then the stress-test bar. | "Nemotron 3 Super writes the task as checkable goals. Tavily finds each object's real size and weight, with sources, and Nemotron Nano reads off the numbers: an apple is 6 centimetres, a soda can 378 grams. Super places the objects and plans the motion, as data, never code. MuJoCo runs it, and code, not a model, checks every goal. Then the plan is re-run sixteen times with the objects nudged, heavier, more slippery. Sixteen out of sixteen: it's robust, not lucky." |
| 4 | 1:25–2:00 | Tennis-ball run: `/?run=20261028-093443-14f0&attempt=1&t=99` (red ✗ label), the Critic card, Attempt 2 tab, green ✓ | "When it fails, it fixes itself. Here the ball was released a centimetre and a half too high, and rolled off. The code measures the miss. Nemotron Super VL watches the frames and explains why. Super repairs the plan, and attempt two succeeds. If a repair fails again, it escalates to Nemotron Ultra. The first plan survived only eleven of twenty perturbed re-runs; the repaired one, all twenty." |
| 5 | 2:00–2:22 | "Beyond pick & place" tab: drawer run, knock-over run; click Export scene, show the zip contents (scene XML, task.json) | "It goes beyond pick-and-place: opening drawers like NVIDIA's Isaac Lab tasks, knocking things over, sorting. And every scene exports as a MuJoCo file with its success checks, ready to test your own robot policy." |
| 6 | 2:22–2:42 | Model-cost bar, then the LangSmith trace (click the link), then docs/eval.md table | "Each step shows which model ran it, how long it took and what it cost. A typical task costs about a cent. Every call is traced in LangSmith. On our evaluation suite, a batch job on Nebius, every task passes." |
| 7 | 2:42–2:52 | Back to the drawer replay, title card with the URL | "Spec-to-Sim. Describe it, simulate it, and trust the result." |

Segment 3 alone is ~60 s of the modules in action. Segments 3–5 together are ~95 s.

## Production steps
1. Restart the app with all recorded runs: `uv run uvicorn app.main:app`.
2. Generate the narration (one audio file per shot) from the table above.
3. Record each shot separately (QuickTime: File → New Screen Recording, or OBS). Shots 1, 4 and 5 use recorded runs, so no API cost.
4. Edit in iMovie, DaVinci Resolve or CapCut: put the narration on each shot, add the "4× speed" tag in shot 3, captions, and the title cards.
5. Export 1080p, upload to YouTube as unlisted, paste the link into README and Devpost.

## Could Claude make parts of it?
- **Yes:** scripted screen captures of recorded runs (headless Chrome with fixed camera times, `?t=` links), title cards as images, and captions (.srt) from the script.
- **Needs you:** choosing and generating the voice (account on a TTS service), the live-run recording, final edit, and the YouTube upload.
