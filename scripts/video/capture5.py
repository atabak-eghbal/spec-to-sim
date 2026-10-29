"""v4 shots: everything in filming mode (?clean=1: no playback bar, replays hold their final frame
instead of looping) and a dedicated, centred training-data shot.

    .venv/bin/python capture4.py live | replays | data | tavily
"""
import json
import shutil
import sys
import time
from pathlib import Path

import capture2
import capture3
from capture2 import DRAWER1, KNOCK, TENNIS, contexts
from playwright.sync_api import sync_playwright
from rec import BASE, Shot, to_mp4

HERE = Path(__file__).parent
CLIPS4 = HERE / "clips5"
CLIPS4.mkdir(exist_ok=True)
FRAMES = HERE / "shots"
TH = "&theater=1&clean=1"
DRAWER2 = "20261028-142918-490a"


def shoot(ctx, name, url, seconds, **kw):
    capture2.CLIPS = CLIPS4
    return capture2.shoot(ctx, name, url, seconds, **kw)


def live(wide, prompt=None):
    capture3.CLIPS3 = CLIPS4
    page = wide.new_page()
    page.goto(BASE + "/?hero=0&clean=1")
    page.wait_for_timeout(2500)
    shot = Shot(page, FRAMES / "c_live", every_nth=2)
    shot.start()
    shot.wait(0.8)
    page.click("#prompt")
    page.keyboard.type(prompt or capture3.LIVE_PROMPT, delay=26)
    shot.wait(0.5)
    page.click("#runbtn")
    marks, seen, last_h, t0 = [("submit", time.time())], set(), 0, time.time()
    while time.time() - t0 < 400:
        shot.wait(0.5)
        for k in page.evaluate("state.steps.map(s => s.step + ':' + (s.attempt || 0) + ':' + (s.doneTs ? 'done' : 'run'))"):
            if k not in seen:
                seen.add(k)
                marks.append((k, time.time()))
        h = page.evaluate("document.body.scrollHeight")
        if h != last_h:
            page.evaluate("window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})")
            last_h = h
        if page.evaluate("state.done !== null || state.error !== null"):
            marks.append(("done", time.time()))
            break
    shot.wait(2.5)
    run_id = page.evaluate("state.runId")
    dur, first = capture3._finish(shot, "c_live")
    page.close()
    (HERE / "live_run5.json").write_text(json.dumps({"run_id": run_id, "seconds": dur,
                                                      "marks": [(k, round(t - first, 2)) for k, t in marks]}, indent=1))
    print(f"c_live: {dur:.1f} s run {run_id}", flush=True)
    shoot(wide, "c3_result", f"/?run={run_id}&t=99{TH}&orbit=1", 4.6)
    shoot(wide, "j1_cascade", f"/?run={run_id}&clean=1", 5.5)


def from_start(seconds):
    """Restart the replay (and anything following it) at t=0 when filming starts."""
    def during(page, shot):
        page.evaluate("(cmp ? cmp.right : viewer).t = 0; (cmp ? cmp.right : viewer).setPlaying(true)")
        shot.wait(seconds)
    return during


def replays(wide):
    shoot(wide, "a_hook", f"/?run={DRAWER2}{TH}&orbit=1", 0, during=from_start(20.5))
    shoot(wide, "e1_compare", f"/?run={TENNIS}&view=compare{TH}", 0, during=from_start(10.5))
    shoot(wide, "e3_compare_end", f"/?run={TENNIS}&view=compare&t=99{TH}&orbit=1", 4.5)
    shoot(wide, "g1_drawer", f"/?run={DRAWER1}{TH}", 0, during=from_start(16))
    shoot(wide, "g2_knock", f"/?run={KNOCK}{TH}", 0, during=from_start(4.5))

    page = wide.new_page()  # drag test: edit mode opened directly (its button is hidden when filming)
    page.goto(BASE + f"/?run={TENNIS}&attempt=2{TH}")
    page.wait_for_timeout(4000)
    shot = Shot(page, FRAMES / "f2_try", every_nth=2)
    shot.start()
    shot.wait(0.6)
    page.evaluate("viewer.startEdit()")
    shot.wait(0.7)
    x, y = page.evaluate("""() => { const g = viewer.bodies.find(b => b.userData.body === 'obj/rubiks_cube');
      const v = g.position.clone().project(viewer.camera); const r = viewer.renderer.domElement.getBoundingClientRect();
      return [r.left + (v.x+1)/2*r.width, r.top + (1-v.y)/2*r.height]; }""")
    page.mouse.move(x, y)
    shot.wait(0.3)
    page.mouse.down()
    for k in range(1, 31):
        page.mouse.move(x - 7.5 * k, y + 3.4 * k)
        page.wait_for_timeout(40)
    page.mouse.up()
    shot.wait(0.9)
    page.click(".v-rerun")
    t_click = time.time()
    while not page.evaluate("state.selected > 1000") and time.time() - t_click < 20:
        shot.wait(0.1)
    t_result = time.time()
    shot.wait(8.5)
    shot.stop()
    dur = to_mp4(FRAMES / "f2_try", CLIPS4 / "f2_try.mp4")
    first = shot.frames[0][1]
    shutil.rmtree(FRAMES / "f2_try")
    page.close()
    (HERE / "drag5.json").write_text(json.dumps({"click": round(t_click - first, 2), "result": round(t_result - first, 2)}))
    print(f"f2_try: {dur:.1f} s", flush=True)


def data(narrow):
    run = json.loads((HERE / "live_run5.json").read_text())["run_id"]
    page = narrow.new_page()
    page.goto(BASE + f"/?run={run}&clean=1")
    page.wait_for_timeout(4500)
    page.evaluate("document.getElementById('dataset').scrollIntoView({block: 'center'})")
    page.wait_for_timeout(600)
    shot = Shot(page, FRAMES / "h1_data", every_nth=2)
    shot.start()
    shot.wait(1.4)
    page.click("#dataset-btn")
    t_click = time.time()
    while "again" not in page.inner_text("#dataset-btn") and time.time() - t_click < 300:
        shot.wait(0.5)
    t_done = time.time()
    shot.wait(4.0)
    shot.stop()
    dur = to_mp4(FRAMES / "h1_data", CLIPS4 / "h1_data.mp4")
    first = shot.frames[0][1]
    shutil.rmtree(FRAMES / "h1_data")
    page.close()
    (HERE / "data5.json").write_text(json.dumps({"click": round(t_click - first, 2), "done": round(t_done - first, 2)}))
    print(f"h1_data: {dur:.1f} s", flush=True)


def after(wide):
    run_id = json.loads((HERE / "live_run5.json").read_text())["run_id"]
    shoot(wide, "c3_result", f"/?run={run_id}&t=99{TH}&orbit=1", 4.6)
    shoot(wide, "j1_cascade", f"/?run={run_id}&clean=1", 5.5)


def tavily(narrow):
    run = json.loads((HERE / "live_run5.json").read_text())["run_id"]
    capture3.CLIPS3 = CLIPS4
    capture3.slow_scroll(narrow, "d1_ground", f"/?run={run}&clean=1", "Grounder", 13.8, 240)
    capture3.slow_scroll(narrow, "d2_grapefruit", "/?run=20261028-180549-7019&clean=1", "Grounder", 13.8, 360)


if __name__ == "__main__":
    pw = sync_playwright().start()
    b = pw.chromium.launch(channel="chrome", headless=True,
                           args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist", "--hide-scrollbars"])
    wide, narrow, _ = contexts(b)
    try:
        {"live": lambda: live(wide, sys.argv[2] if len(sys.argv) > 2 else None), "replays": lambda: replays(wide), "data": lambda: data(narrow),
         "after": lambda: after(wide),
         "tavily": lambda: tavily(narrow)}[sys.argv[1]]()
    finally:
        b.close()
        pw.stop()
