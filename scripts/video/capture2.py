"""Shots for the v2 demo video (dark UI, compare, try-it, training data, NVIDIA fit).

    .venv/bin/python capture2.py live      # live run + shots that need it (calls the models, a few cents)
    .venv/bin/python capture2.py rest      # everything from recorded runs
    .venv/bin/python capture2.py cards     # animated title/info cards
"""
import json
import shutil
import sys
import time
from pathlib import Path

from cards import BADGES, HEAD
from playwright.sync_api import sync_playwright
from rec import BASE, Shot, to_mp4

HERE = Path(__file__).parent
CLIPS, FRAMES = HERE / "clips2", HERE / "shots"
CLIPS.mkdir(exist_ok=True)
TENNIS, DRAWER1, KNOCK, FRUIT, APPLE_NO = ("20261028-151115-e705", "20261028-142206-5e4c", "20261028-165935-e1e5",
                                         "20261028-123520-54e9", "20261028-151454-c73a")
LIVE_PROMPT = "Put the lime and the tennis ball into the fruit basket, and the soda can into the recycling bin"
TH = "&theater=1&clean=1"


def contexts(b):
    wide = b.new_context(viewport={"width": 1280, "height": 720}, device_scale_factor=1.5)
    narrow = b.new_context(viewport={"width": 760, "height": 428}, device_scale_factor=1920 / 760)
    card = b.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    return wide, narrow, card


def shoot(ctx, name, url, seconds, before=None, during=None, settle=4.0, nth=2):
    page = ctx.new_page()
    page.goto(BASE + url)
    page.wait_for_timeout(int(settle * 1000))
    if before:
        before(page)
    shot = Shot(page, FRAMES / name, every_nth=nth)
    shot.start()
    if during:
        during(page, shot)
    else:
        shot.wait(seconds)
    shot.stop()
    dur = to_mp4(FRAMES / name, CLIPS / f"{name}.mp4")
    shutil.rmtree(FRAMES / name)
    page.close()
    print(f"{name}: {dur:.1f} s", flush=True)
    return dur


def to_step(text, offset=12):
    def f(page):
        page.evaluate("""([t, off]) => {
          const el = [...document.querySelectorAll('.step-name')].find(e => e.textContent.trim() === t);
          if (el) window.scrollTo({top: el.getBoundingClientRect().top + window.scrollY - off});
        }""", [text, offset])
        page.wait_for_timeout(700)
    return f


def live(wide, narrow):
    page = wide.new_page()
    page.goto(BASE + "/?hero=0")  # empty stage: no autoplaying recorded run
    page.wait_for_timeout(2500)
    shot = Shot(page, FRAMES / "c_live", every_nth=3)
    shot.start()
    shot.wait(0.8)
    page.click("#prompt")
    page.keyboard.type(LIVE_PROMPT, delay=26)
    shot.wait(0.5)
    page.click("#runbtn")
    t0, last_h = time.time(), 0
    while time.time() - t0 < 400:
        shot.wait(1.0)
        if page.evaluate("state.done !== null || state.error !== null"):
            break
        h = page.evaluate("document.body.scrollHeight")
        if h != last_h:
            page.evaluate("window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})")
            last_h = h
    shot.wait(2.0)
    run_id = page.evaluate("state.runId")
    shot.stop()
    dur = to_mp4(FRAMES / "c_live", CLIPS / "c_live.mp4")
    shutil.rmtree(FRAMES / "c_live")
    page.close()
    (HERE / "live_run2.json").write_text(json.dumps({"run_id": run_id, "seconds": dur}))
    print(f"c_live: {dur:.1f} s run {run_id}", flush=True)
    after_live(wide, narrow, run_id)


def after_live(wide, narrow, run_id):
    shoot(narrow, "c2_ground", f"/?run={run_id}", 3.6, before=to_step("Grounder"))
    shoot(wide, "c3_result", f"/?run={run_id}&t=99{TH}&orbit=1", 4.6)
    shoot(wide, "j1_cascade", f"/?run={run_id}", 5.5)

    def data(page, shot):
        page.evaluate("document.getElementById('dataset').scrollIntoView({block: 'center'})")
        shot.wait(1.2)
        page.click("#dataset-btn")
        for _ in range(240):
            shot.wait(1.0)
            if "again" in page.inner_text("#dataset-btn"):
                break
        shot.wait(2.0)
    shoot(wide, "h1_data", f"/?run={run_id}", 0, during=data)


def rest(wide, narrow):
    shoot(narrow, "d1_infeasible", f"/?run={APPLE_NO}", 7, before=to_step("Grounder", 6))
    shoot(narrow, "d2_lime", f"/?run={FRUIT}", 7, before=to_step("Grounder", 6))
    shoot(wide, "e1_compare", f"/?run={TENNIS}&view=compare{TH}", 10.5)
    shoot(narrow, "e2_critic", f"/?run={TENNIS}", 6.5, before=to_step("Critic", 6))
    shoot(wide, "e3_compare_end", f"/?run={TENNIS}&view=compare&t=99{TH}&orbit=1", 4.5)

    def drag(page, shot):
        shot.wait(0.6)
        page.click("text=Move objects")
        shot.wait(0.7)
        x, y = page.evaluate("""() => { const g = viewer.bodies.find(b => b.userData.body === 'obj/rubiks_cube');
          const v = g.position.clone().project(viewer.camera); const r = viewer.renderer.domElement.getBoundingClientRect();
          return [r.left + (v.x+1)/2*r.width, r.top + (1-v.y)/2*r.height]; }""")
        page.mouse.move(x, y)
        shot.wait(0.3)
        page.mouse.down()
        for k in range(1, 25):
            page.mouse.move(x - 9 * k, y + 4 * k)
            page.wait_for_timeout(35)
        page.mouse.up()
        shot.wait(0.8)
        page.click(".v-rerun")
        for _ in range(40):
            shot.wait(0.25)
            if page.evaluate("state.selected > 1000"):
                break
        shot.wait(8.5)
    shoot(wide, "f2_try", f"/?run={TENNIS}&attempt=2{TH}", 0, during=drag)
    shoot(wide, "g1_drawer", f"/?run={DRAWER1}{TH}", 16)
    shoot(wide, "g2_knock", f"/?run={KNOCK}{TH}", 4.2)


CARD_CSS = """<style>
.rise { opacity: 0; transform: translateY(18px); animation: rise .7s ease-out forwards; }
@keyframes rise { to { opacity: 1; transform: none; } }
.fit { display: grid; grid-template-columns: 1fr 1fr 1.15fr; gap: 34px; align-items: stretch; }
.fit .col h3 { font: 700 40px/1.1 "Archivo"; font-stretch: 85%; color: #fff; margin: 14px 0 18px; }
.fit .col p { font-size: 29px; }
.fit .arrow { font: 500 26px/1.4 "IBM Plex Mono"; color: #7fb0e0; margin-top: 18px; }
.fit .us { border-color: #6fc38f; background: rgba(111,195,143,0.10); }
.tree { font: 400 27px/1.55 "IBM Plex Mono"; color: #d8e3ea; }
</style>"""


def card_pages():
    d = lambda s: f'style="animation-delay:{s}s"'
    return {
        "b_problem": (14.8, f"""<div class="pad"><span class="label rise" {d(0.1)}>The problem</span>
<h2 class="rise" {d(0.3)} style="font-size:72px">Robot test scenes are built by hand</h2>
<div class="row" style="margin-top:14px"><div class="col rise" {d(2.0)}><span class="label" style="color:#ec8a72">Today, by hand</span>
<p style="margin-top:18px">Measure objects · write scene files · place everything · write success checks · debug. <b>Days per scene.</b></p></div>
<div class="col rise" {d(6.5)}><span class="label" style="color:#6fc38f">Spec-to-Sim</span>
<p style="margin-top:18px">One sentence → real sizes from the web → scene, motion plan, checked result. <b>About a minute, typically under a cent.</b></p></div></div>
<div class="rise" {d(9.0)}>{BADGES}</div></div>"""),
        "h2_lerobot": (6.8, f"""<div class="pad"><span class="label rise" {d(0.1)}>Training data · one click</span>
<h2 class="rise" {d(0.3)}>20 successful demonstrations, LeRobot format</h2>
<div class="row"><pre class="tree rise" {d(0.7)} style="flex:0 0 760px">dataset/
├── meta/info.json        v2.1, 10 fps
├── meta/episodes.jsonl   task text
├── data/chunk-000/       parquet per episode
│   state (9) · action (8) · object poses
└── videos/chunk-000/     camera mp4 per episode</pre>
<pre class="rise" {d(1.6)} style="flex:1;font-size:26px">from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("spec2sim/bins", root="dataset")
# loads: 20 episodes, 3,440 frames
# observation.images.front (3, 240, 320)
# observation.state (9,)   action (8,)</pre></div>
<p class="lede rise" {d(2.6)} style="font-size:30px">Varied starts, masses and friction; failures dropped. A format NVIDIA GR00T post-training can read.
Scripted demos, not yet used to train a policy. Packaged for a Nebius Serverless Job: <span class="mono">scripts/generate_data.py</span></p></div>"""),
        "i_fit": (18.7, f"""<div class="pad" style="gap:30px"><span class="label rise" {d(0.1)}>Where it fits in NVIDIA's robotics stack</span>
<div class="fit"><div class="col rise" {d(1.0)}><span class="label">Isaac Lab-Arena</span><h3>Language → environment spec</h3>
<p>An LLM writes a spec that is checked for validity.</p><div class="arrow">valid ≠ task done</div></div>
<div class="col rise" {d(5.0)}><span class="label">Isaac Lab Mimic · GR00T blueprint</span><h3>Human demos → many demos</h3>
<p>A few teleoperated demonstrations become thousands of synthetic ones.</p><div class="arrow">needs human demos first</div></div>
<div class="col us rise" {d(9.0)}><span class="label" style="color:#6fc38f">Spec-to-Sim</span><h3>One sentence → checked scene + demos</h3>
<p>Running MuJoCo scene, goals checked in code, VLM-guided repair, robustness score, LeRobot dataset.</p>
<div class="arrow" style="color:#6fc38f">no human demo · scripted, GR00T-readable</div></div></div>
<p class="mono rise" {d(13.5)} style="font-size:26px;color:#9aabb8">Success predicates use NVIDIA RoboLab's vocabulary · scenes are plain MJCF (MuJoCo, MuJoCo Warp)</p></div>"""),
    }


def cards(card_ctx):
    for name, (secs, body) in card_pages().items():
        page = card_ctx.new_page()
        page.set_content(HEAD + CARD_CSS + body.replace('class="rise"', 'class="rise"'))
        page.wait_for_timeout(50)
        shot = Shot(page, FRAMES / name, every_nth=1)
        shot.start()
        page.evaluate("""() => document.getAnimations().forEach(a => { a.currentTime = 0; a.play(); })""")
        shot.wait(secs)
        shot.stop()
        dur = to_mp4(FRAMES / name, CLIPS / f"{name}.mp4")
        shutil.rmtree(FRAMES / name)
        page.close()
        print(f"{name}: {dur:.1f} s", flush=True)


if __name__ == "__main__":
    pw = sync_playwright().start()
    b = pw.chromium.launch(channel="chrome", headless=True,
                           args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist", "--hide-scrollbars"])
    wide, narrow, card_ctx = contexts(b)
    try:
        what = sys.argv[1] if len(sys.argv) > 1 else "rest"
        if what == "live":
            live(wide, narrow)
        elif what == "after":
            after_live(wide, narrow, json.loads((HERE / "live_run2.json").read_text())["run_id"])
        elif what == "cards":
            cards(card_ctx)
        else:
            rest(wide, narrow)
    finally:
        b.close()
        pw.stop()
