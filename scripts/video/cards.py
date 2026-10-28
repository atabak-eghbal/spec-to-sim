"""Title cards and overlay tags as 1920x1080 PNGs, rendered with Playwright in the app's fonts."""
import base64
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
OUT = HERE / "cards"
OUT.mkdir(exist_ok=True)

HEAD = """<link href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@75..100,500..800&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
* { box-sizing: border-box; margin: 0; }
body { width: 1920px; height: 1080px; font-family: "IBM Plex Sans", sans-serif; color: #e3eaef;
  background: radial-gradient(ellipse at 30% 20%, #23323f 0%, #141c23 60%, #0d1318 100%); overflow: hidden; }
.pad { padding: 110px 140px; height: 100%; display: flex; flex-direction: column; justify-content: center; gap: 36px; }
h1 { font: 800 132px/0.95 "Archivo"; font-stretch: 78%; letter-spacing: -1px; color: #fff; }
h2 { font: 700 64px/1.05 "Archivo"; font-stretch: 82%; color: #fff; }
.lede { font-size: 40px; line-height: 1.35; color: #b9c7d2; max-width: 1500px; }
.label { font: 500 24px/1 "IBM Plex Mono"; letter-spacing: 3px; text-transform: uppercase; color: #7fb0e0; }
.badges { display: flex; gap: 14px; flex-wrap: wrap; }
.badge { font: 500 26px/1 "IBM Plex Mono"; padding: 12px 18px; border-radius: 6px; color: #0d1318; }
.mono { font-family: "IBM Plex Mono"; }
.row { display: flex; gap: 60px; align-items: stretch; }
.col { flex: 1; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 14px; padding: 40px 44px; }
.col p { font-size: 32px; line-height: 1.45; color: #c9d5de; }
.bar { height: 46px; border-radius: 6px; background: #3a211b; overflow: hidden; margin: 14px 0 8px; }
.bar span { display: block; height: 100%; }
.big { font: 600 64px/1 "IBM Plex Mono"; }
table { border-collapse: collapse; width: 100%; font-size: 27px; }
td { padding: 11px 14px; border-bottom: 1px solid rgba(255,255,255,0.08); }
td.ok { color: #6fc38f; font-family: "IBM Plex Mono"; text-align: right; white-space: nowrap; }
pre { font: 400 28px/1.5 "IBM Plex Mono"; background: rgba(0,0,0,0.35); padding: 30px 36px; border-radius: 12px; color: #d8e3ea; }
</style>"""

BADGES = """<div class="badges"><span class="badge" style="background:#7fb0e0">Nemotron 3 Super</span>
<span class="badge" style="background:#4fc2b8">Nemotron Nano</span><span class="badge" style="background:#b996e6">Nemotron 3.5 Super VL</span>
<span class="badge" style="background:#ec8a72">Nemotron Ultra</span><span class="badge" style="background:#e3eaef">Nebius Token Factory</span></div>"""

CARDS = {
    "b_problem": f"""<div class="pad"><span class="label">Physical AI · Nebius x NVIDIA Hackathon</span><h1>Spec-to-Sim</h1>
<p class="lede">Describe a robot task in plain English. Get a working, checked simulation — and an AI that fixes its own mistakes.</p>
<div class="row" style="margin-top:20px"><div class="col"><span class="label" style="color:#ec8a72">Today, by hand</span><p style="margin-top:18px">Measure the objects · write scene files · place everything · write success checks · debug. <b>Days per scene.</b></p></div>
<div class="col"><span class="label" style="color:#6fc38f">Spec-to-Sim</span><p style="margin-top:18px">One sentence → real-world sizes from the web → scene, motion plan, checked result. <b>About a minute, about a cent.</b></p></div></div>
{BADGES}</div>""",
    "e4_stress": """<div class="pad"><span class="label">Stress test · 20 perturbed re-runs</span><h2>Repaired, not lucky</h2>
<p class="lede">Each object nudged up to 1.5 cm, turned up to 9°, ±20% mass, ±30% friction. Physics only — no model calls.</p>
<div style="margin-top:10px"><div class="mono" style="font-size:32px;color:#b9c7d2">Attempt 1 plan (ball rolled off)</div>
<div class="bar"><span style="width:60%;background:#e2b55c"></span></div><div class="big" style="color:#e2b55c">12 / 20</div></div>
<div><div class="mono" style="font-size:32px;color:#b9c7d2">Repaired plan</div>
<div class="bar"><span style="width:100%;background:#6fc38f"></span></div><div class="big" style="color:#6fc38f">20 / 20</div></div></div>""",
    "f4_export": """<div class="pad"><span class="label">Export scene</span><h2>Every scene is reusable</h2>
<div class="row"><pre style="flex:0 0 640px">spec2sim-run.zip
├── scene_objects.xml   MuJoCo MJCF
├── task.json           sizes + sources,
│                       success checks
├── plan.json           motion plan
└── README.md</pre>
<pre style="flex:1">robot = mujoco.MjSpec.from_file(
    "menagerie/franka_emika_panda/scene.xml")
objects = mujoco.MjSpec.from_file("scene_objects.xml")
robot.attach(objects, frame=robot.worldbody.add_frame(),
             prefix="task/")
model = robot.compile()   # test your own policy</pre></div>
<p class="lede" style="font-size:32px">Success checks use NVIDIA RoboLab's vocabulary: object_on_top, object_in_container, object_left_of…</p></div>""",
    "g2_eval": """<div class="pad" style="gap:26px"><span class="label">Batch evaluation · scripts/eval.py · Nebius Serverless Job</span><h2>11 of 11 tasks pass</h2>
<table>
<tr><td>Stack the red cube on the blue cube</td><td class="ok">✓ pass</td></tr>
<tr><td>Put the Rubik's cube on top of the soda can</td><td class="ok">✓ pass</td></tr>
<tr><td>Place the tennis ball on the Rubik's cube</td><td class="ok">✓ pass</td></tr>
<tr><td>Three-block tower: red, green, blue</td><td class="ok">✓ pass</td></tr>
<tr><td>Clear the table: three blocks into the storage bin</td><td class="ok">✓ pass</td></tr>
<tr><td>Apple and lime into the basket, soda can into the recycling bin</td><td class="ok">✓ pass</td></tr>
<tr><td>Stack red on blue, then green block and tennis ball into the tray</td><td class="ok">✓ pass</td></tr>
<tr><td>Soda can to the left of the Rubik's cube, kept upright</td><td class="ok">✓ pass</td></tr>
<tr><td>Open the drawer, put the Rubik's cube in, close it</td><td class="ok">✓ pass</td></tr>
<tr><td>Tennis ball and Rubik's cube into the drawer, close it</td><td class="ok">✓ pass</td></tr>
<tr><td>Knock over the tower of three blocks</td><td class="ok">✓ pass</td></tr>
</table><p class="mono" style="font-size:24px;color:#9aabb8">Pass/fail decided by code, never by a model. Whole development so far: under $2 of model calls.</p></div>""",
}


def close_card(bg: Path) -> str:
    b64 = base64.b64encode(bg.read_bytes()).decode()
    return f"""<div style="position:absolute;inset:0;background:url(data:image/jpeg;base64,{b64}) center/cover;filter:brightness(0.45)"></div>
<div class="pad" style="position:relative"><h1>Spec-to-Sim</h1><p class="lede" style="color:#fff;font-size:48px">Describe it. Simulate it. Trust it.</p>
{BADGES}<p class="mono" style="font-size:30px;color:#b9c7d2">MuJoCo · Tavily · LangSmith · github.com/atabak-eghbal/spec-to-sim</p></div>"""


def tag(text: str) -> str:
    return f"""<style>body{{background:transparent!important}}</style><div style="position:absolute;right:48px;top:40px;font:600 30px/1 'IBM Plex Mono';
color:#fff;background:rgba(13,19,24,0.78);padding:14px 22px;border-radius:40px;border:1px solid rgba(255,255,255,0.25)">{text}</div>"""


def main(tags=()):
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        page = b.new_page(viewport={"width": 1920, "height": 1080})
        for name, body in CARDS.items():
            page.set_content(HEAD + body); page.wait_for_timeout(1200)
            page.screenshot(path=str(OUT / f"{name}.png"))
        bg = HERE / "cards/close_bg.jpg"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-sseof", "-1.5", "-i", str(HERE / "clips2/c3_result.mp4"),
                        "-frames:v", "1", str(bg)], check=True)
        page.set_content(HEAD + close_card(bg)); page.wait_for_timeout(1200)
        page.screenshot(path=str(OUT / "h_close.png"))
        for t in tags:
            page.set_content(HEAD + tag(t)); page.wait_for_timeout(600)
            page.screenshot(path=str(OUT / f"tag_{t.split()[0].replace('×', 'x')}.png"), omit_background=True)
        b.close()
    print(sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    import sys
    main(tags=sys.argv[1:])
