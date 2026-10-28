# Demo video (v3, 2:48)

Built automatically by `scripts/video/` (Playwright capture of the running app, Kokoro narration,
ffmpeg assembly with burned-in captions and an .srt for YouTube). Every claim is backed by a recorded
run or a test. At least 1 minute of the key modules in action (0:22 to 1:50 is continuous app footage).

| Time | Beat | Shot | Why it's there |
|---|---|---|---|
| 0:00 | Hook | Two-object drawer task, full-screen 3D, slow orbit | Show the result first |
| 0:08 | Problem | Animated card: days by hand vs one sentence | Potential Impact |
| 0:23 | Live run | Typed prompt, pipeline strip; jump cuts at 1x (labelled "N s skipped"); a real self-repair (bin out of reach → Critic → fix), 15/16 robust | Technological Implementation, modules in action |
| 0:45 | **Tavily is the reality check** | Grounder card: product photos, "4/4 agree", friction/bounciness from the web, Franka limits from its datasheet; a "18 cm, 246 g" grapefruit corrected to 8.2 cm by its weight, then stopped as too wide for the gripper | Best Use of Tavily |
| 1:12 | Self-repair | Before / after side by side; Critic close-up | Quality of Idea |
| 1:25 | Robust, not lucky | Stress card 12/20 → 20/20; live drag-and-re-run | Rigor, Design |
| 1:42 | Beyond pick & place | Drawer (Isaac Lab's Franka drawer task), knock-over | Physical AI depth |
| 1:50 | Training data | One click → 20 demos (8× speed) → LeRobot card (loader output) | Track text: "generate synthetic data"; GR00T |
| 2:07 | Where it fits at NVIDIA | Animated comparison: Isaac Lab-Arena · Mimic/GR00T blueprint · Spec-to-Sim | Impact for NVIDIA |
| 2:25 | Rigor | Pipeline strip with costs; 11/11 eval card | Technological Implementation |
| 2:36 | Close | Title over the final 3D frame | |

v3 fixes (from review of v2): no frozen frames (no shot stretched past its footage; static shots push in slowly),
no rushed footage (3D at most 1.3x; waiting removed by labelled jump cuts instead of 8-10x fast-forward).

Rules check: under 3 minutes; no music; no third-party logos (product names only where needed);
narration voice is Kokoro (Apache-2.0) running locally.

To update after a change: re-run only the affected `capture2.py` step, then `assemble2.py`.
If Nebius billing is activated, add a 3-5 s clip of the Serverless Job logs to the training-data beat
and change its line to "...runs as a Nebius Serverless Job".
