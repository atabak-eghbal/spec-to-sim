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
td.bad { color: #ec8a72; font-family: "IBM Plex Mono"; text-align: right; white-space: nowrap; }
pre { font: 400 28px/1.5 "IBM Plex Mono"; background: rgba(0,0,0,0.35); padding: 30px 36px; border-radius: 12px; color: #d8e3ea; }
</style>"""

BADGES = """<div class="badges"><span class="badge" style="background:#7fb0e0">Nemotron 3 Super</span>
<span class="badge" style="background:#4fc2b8">Nemotron Nano</span><span class="badge" style="background:#b996e6">Nemotron 3.5 Super VL</span>
<span class="badge" style="background:#e3eaef">Nebius Token Factory</span><span class="badge" style="background:#f2c14e">Tavily</span></div>"""

CARDS = {
    "b_problem": f"""<div class="pad"><span class="label">Physical AI · Nebius x NVIDIA Hackathon</span><h1>Spec-to-Sim</h1>
<p class="lede">Describe a robot task in plain English. Get a working, checked simulation — and an AI that fixes its own mistakes.</p>
<div class="row" style="margin-top:20px"><div class="col"><span class="label" style="color:#ec8a72">Today, by hand</span><p style="margin-top:18px">Measure the objects · write scene files · place everything · write success checks · debug. <b>Days per scene.</b></p></div>
<div class="col"><span class="label" style="color:#6fc38f">Spec-to-Sim</span><p style="margin-top:18px">One sentence → real-world sizes from the web → scene, motion plan, checked result. <b>About a minute, about a cent.</b></p></div></div>
{BADGES}</div>""",
    "e4_stress": """<div class="pad"><span class="label">Stress test · 16 perturbed re-runs, physics only</span><h2>Robust, or lucky?</h2>
<p class="lede">Objects moved up to 4 cm and turned up to 30°, ±20% mass, ±30% friction, and the robot sees each object up to 0.5 cm off.</p>
<div style="margin-top:6px"><div class="mono" style="font-size:30px;color:#b9c7d2">Tennis ball on the cube, repaired plan</div>
<div class="bar"><span style="width:100%;background:#6fc38f"></span></div><div class="big" style="color:#6fc38f">16 / 16</div></div>
<div><div class="mono" style="font-size:30px;color:#b9c7d2">Soda can left of the cube, the live run</div>
<div class="bar"><span style="width:81.25%;background:#6fc38f"></span></div><div class="big" style="color:#6fc38f">13 / 16</div></div>
<div><div class="mono" style="font-size:30px;color:#b9c7d2">Fruit sorting: passed once, but the soda can barely stays standing</div>
<div class="bar"><span style="width:6.25%;background:#ec8a72"></span></div><div class="big" style="color:#ec8a72">1 / 16</div></div></div>""",
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
    "g2_eval": """<div class="pad" style="gap:24px"><span class="label">Batch evaluation · scripts/eval.py · 104 runs, one commit, 3 per prompt</span><h2>93 % on prompts it had never seen</h2>
<table>
<tr><td><b>15 held-out prompts</b>, written before any was run</td><td class="ok">42 / 45</td></tr>
<tr><td>11 development prompts</td><td class="ok">29 / 33</td></tr>
<tr><td>Model cost per task (median)</td><td class="ok">$0.005</td></tr>
<tr><td>Cut: Ultra escalation (fixed 4 of 18 runs)</td><td class="ok" style="color:#e2b55c">→ 3 Super repairs, tested in sim</td></tr>
<tr><td>Vision model: Critic and scene check</td><td class="ok" style="color:#e2b55c">no measurable gain yet</td></tr>
<tr><td>Weak spot: pushing things over</td><td class="bad">4 / 6</td></tr>
</table><p class="mono" style="font-size:24px;color:#9aabb8">Pass/fail decided by code, never by a model. Everything above is in docs/eval.md.</p></div>""",
    "i_harness": """<div class="pad" style="gap:30px"><span class="label">Every export is a policy test</span><h2>Bring your policy. Get its score.</h2>
<div class="row"><pre style="flex:0 0 720px">export.zip
├── scene_objects.xml   MuJoCo MJCF
├── task.json           goals + sources
├── spec2sim_env.py     reset · step · success
├── variations.json     the stress-test ranges
└── evaluate.py         mujoco + numpy only</pre>
<pre style="flex:1">$ python evaluate.py --policy my_policy.py \
      --episodes 20

goal                           passed
left_of soda_can -> rubiks_cube  0/20
upright soda_can                20/20

Overall success: 0/20 = 0.0 %
# baseline policy: holds still</pre></div>
<p class="lede" style="font-size:32px">Same randomization, same goal checks as the stress test, in one file.</p></div>""",
}


def close_card(bg: Path) -> str:
    b64 = base64.b64encode(bg.read_bytes()).decode()
    return f"""<div style="position:absolute;inset:0;background:url(data:image/jpeg;base64,{b64}) center/cover;filter:brightness(0.45)"></div>
<div class="pad" style="position:relative"><h1>Spec-to-Sim</h1><p class="lede" style="color:#fff;font-size:48px">Describe it. Simulate it. Test it.</p>
{BADGES}<p class="mono" style="font-size:30px;color:#b9c7d2">MuJoCo · Tavily · LangSmith · github.com/atabak-eghbal/spec-to-sim</p></div>"""


def tag(text: str) -> str:
    return f"""<style>body{{background:transparent!important}}</style><div style="position:absolute;right:48px;top:40px;font:600 30px/1 'IBM Plex Mono';
color:#fff;background:rgba(13,19,24,0.78);padding:14px 22px;border-radius:40px;border:1px solid rgba(255,255,255,0.25)">{text}</div>"""


def title_card(bg: Path) -> str:
    b64 = base64.b64encode(bg.read_bytes()).decode()
    return f"""<div style="position:absolute;inset:0;background:url(data:image/jpeg;base64,{b64}) center/cover;filter:blur(3px) brightness(0.35)"></div>
<div class="pad" style="position:relative"><span class="label">Physical AI · Nebius × NVIDIA Global AI Hackathon</span><h1 style="font-size:168px">Spec-to-Sim</h1>
<p class="lede" style="font-size:46px;max-width:1550px;color:#dbe5ec">Type a robot task in plain English. Get a checked MuJoCo simulation that repairs itself when it fails, and training data.</p>
{BADGES}</div>"""


CHAPTERS = {"b_problem": ("", "The problem"), "c_live": ("1", "A live run"), "d_tavily": ("2", "Grounded in the real world · Tavily"),
            "e_repair": ("3", "When it fails, it repairs itself"), "f_robust": ("4", "Robust, or lucky? · stress test"),
            "g_beyond": ("5", "Beyond pick and place"), "h_data": ("6", "Training data and a learned policy"),
            "i_fit": ("7", "Test your own policy"), "j_rigor": ("", "Evaluation")}


def chapter(num: str, title: str) -> str:
    n = f'<span style="font:600 30px/1 IBM Plex Mono;color:#0d1318;background:#7fb0e0;border-radius:6px;padding:8px 13px;margin-right:18px">{num}</span>' if num else ""
    return f"""<style>body{{background:transparent!important}}</style>
<div style="position:absolute;left:56px;top:48px;display:flex;align-items:center;background:rgba(10,15,20,0.82);
border:1px solid rgba(127,176,224,0.45);border-left:6px solid #7fb0e0;border-radius:10px;padding:16px 26px">{n}
<span style="font:700 40px/1 Archivo;font-stretch:85%;color:#fff">{title}</span></div>"""


def main(tags=()):
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        page = b.new_page(viewport={"width": 1920, "height": 1080})
        for name, body in CARDS.items():
            page.set_content(HEAD + body); page.wait_for_timeout(1200)
            page.screenshot(path=str(OUT / f"{name}.png"))
        bg = HERE / "cards/close_bg.jpg"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-sseof", "-1.5", "-i", str(HERE / "clips7/c3_result.mp4"),
                        "-frames:v", "1", str(bg)], check=True)
        page.set_content(HEAD + close_card(bg)); page.wait_for_timeout(1200)
        page.screenshot(path=str(OUT / "h_close.png"))
        tbg = HERE / "cards/title_bg.jpg"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "2", "-i", str(HERE / "clips7/a_hook.mp4"),
                        "-frames:v", "1", str(tbg)], check=True)
        page.set_content(HEAD + title_card(tbg)); page.wait_for_timeout(1200)
        page.screenshot(path=str(OUT / "a_title.png"))
        for key, (num, title) in CHAPTERS.items():
            page.set_content(HEAD + chapter(num, title)); page.wait_for_timeout(300)
            page.screenshot(path=str(OUT / f"chapter_{key}.png"), omit_background=True)
        for t in tags:
            page.set_content(HEAD + tag(t)); page.wait_for_timeout(600)
            page.screenshot(path=str(OUT / f"tag_{t.split()[0].replace('×', 'x')}.png"), omit_background=True)
        b.close()
    print(sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    import sys
    main(tags=sys.argv[1:])
