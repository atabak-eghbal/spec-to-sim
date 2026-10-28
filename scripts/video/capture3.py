"""v3 shots: event-timed live run (for jump cuts at 1x), drag test with result time, slow-scroll
close-ups of the new Grounder card and the grapefruit stop. Reuses capture2 helpers.

    .venv/bin/python capture3.py live | drag | tavily | after
"""
import json
import shutil
import sys
import time
from pathlib import Path

from capture2 import FRAMES, TENNIS, TH, contexts, shoot, to_step
from playwright.sync_api import sync_playwright
from rec import BASE, Shot, to_mp4

HERE = Path(__file__).parent
CLIPS3 = HERE / "clips3"
CLIPS3.mkdir(exist_ok=True)
LIVE_PROMPT = "Put the lime and the tennis ball into the fruit basket, and the soda can into the recycling bin"
GRAPEFRUIT = (HERE.parent / "gf.id").read_text().strip() if (HERE.parent / "gf.id").exists() else None


def _finish(shot, name):
    shot.stop()
    dur = to_mp4(FRAMES / name, CLIPS3 / f"{name}.mp4")
    first = shot.frames[0][1]
    shutil.rmtree(FRAMES / name)
    return dur, first


def live(wide):
    page = wide.new_page()
    page.goto(BASE + "/?hero=0")
    page.wait_for_timeout(2500)
    shot = Shot(page, FRAMES / "c_live", every_nth=2)
    shot.start()
    shot.wait(0.8)
    page.click("#prompt")
    page.keyboard.type(LIVE_PROMPT, delay=26)
    shot.wait(0.5)
    page.click("#runbtn")
    marks, seen, last_h, t0 = [("submit", time.time())], set(), 0, time.time()
    while time.time() - t0 < 400:
        shot.wait(0.5)
        steps = page.evaluate("state.steps.map(s => s.step + ':' + (s.attempt || 0) + ':' + (s.doneTs ? 'done' : 'run'))")
        for k in steps:
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
    dur, first = _finish(shot, "c_live")
    page.close()
    (HERE / "live_run3.json").write_text(json.dumps({"run_id": run_id, "seconds": dur,
                                                      "marks": [(k, round(t - first, 2)) for k, t in marks]}, indent=1))
    print(f"c_live: {dur:.1f} s run {run_id}", flush=True)


def drag(wide):
    page = wide.new_page()
    page.goto(BASE + f"/?run={TENNIS}&attempt=2{TH}")
    page.wait_for_timeout(4000)
    shot = Shot(page, FRAMES / "f2_try", every_nth=2)
    shot.start()
    shot.wait(0.6)
    page.click("text=Move objects")
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
    dur, first = _finish(shot, "f2_try")
    page.close()
    (HERE / "drag3.json").write_text(json.dumps({"click": round(t_click - first, 2), "result": round(t_result - first, 2)}))
    print(f"f2_try: {dur:.1f} s, click {t_click - first:.1f}, result {t_result - first:.1f}", flush=True)


def slow_scroll(narrow, name, url, step_name, seconds, pixels):
    page = narrow.new_page()
    page.goto(BASE + url)
    page.wait_for_timeout(4000)
    to_step(step_name, 6)(page)
    shot = Shot(page, FRAMES / name, every_nth=2)
    shot.start()
    shot.wait(0.8)
    n = int((seconds - 1.6) / 0.05)
    for _ in range(n):
        page.evaluate(f"window.scrollBy(0, {pixels / n:.3f})")
        page.wait_for_timeout(50)
    shot.wait(0.8)
    dur, _ = _finish(shot, name)
    page.close()
    print(f"{name}: {dur:.1f} s", flush=True)


def tavily(narrow):
    run = json.loads((HERE / "live_run3.json").read_text())["run_id"]
    slow_scroll(narrow, "d1_ground", f"/?run={run}", "Grounder", 13.8, 240)
    if GRAPEFRUIT:
        slow_scroll(narrow, "d2_grapefruit", f"/?run={GRAPEFRUIT}", "Grounder", 13.8, 360)


def after(wide, narrow):
    run = json.loads((HERE / "live_run3.json").read_text())["run_id"]
    import capture2
    capture2.CLIPS = CLIPS3
    shoot(wide, "c3_result", f"/?run={run}&t=99{TH}&orbit=1", 4.6)
    shoot(wide, "j1_cascade", f"/?run={run}", 5.5)


if __name__ == "__main__":
    pw = sync_playwright().start()
    b = pw.chromium.launch(channel="chrome", headless=True,
                           args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist", "--hide-scrollbars"])
    wide, narrow, _ = contexts(b)
    try:
        {"live": lambda: live(wide), "drag": lambda: drag(wide), "tavily": lambda: tavily(narrow),
         "after": lambda: after(wide, narrow)}[sys.argv[1]]()
    finally:
        b.close()
        pw.stop()
