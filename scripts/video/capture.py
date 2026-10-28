"""Capture every shot of the demo video from the running app (localhost:8000).

    .venv/bin/python capture.py            # all shots except the live run
    .venv/bin/python capture.py live       # the live run (calls the models, costs a few cents)
"""
import json
import shutil
import sys
import time
from pathlib import Path

from rec import BASE, Shot, browser, to_mp4

HERE = Path(__file__).parent
CLIPS = HERE / "clips"
FRAMES = HERE / "shots"
CLIPS.mkdir(exist_ok=True)

TENNIS, DRAWER2, DRAWER1, KNOCK = ("20261028-151115-e705", "20261028-142918-490a",
                                   "20261028-142206-5e4c", "20261028-141428-6c26")
LIVE_PROMPT = "Put the lime and the tennis ball into the fruit basket, and the soda can into the recycling bin"


def record(ctx, name, url, seconds, before=None, during=None, settle=3.5):
    """Open url, optionally act, then capture `seconds` of screen into clips/<name>.mp4."""
    page = ctx.new_page()
    page.goto(BASE + url)
    page.wait_for_timeout(int(settle * 1000))
    if before:
        before(page)
    shot = Shot(page, FRAMES / name)
    shot.start()
    if during:
        during(page, shot)
    else:
        shot.wait(seconds)
    shot.stop()
    dur = to_mp4(FRAMES / name, CLIPS / f"{name}.mp4")
    shutil.rmtree(FRAMES / name)  # disk is tight: keep only the encoded clip
    page.close()
    print(f"{name}: {dur:.1f} s", flush=True)
    return dur


def scroll_to_text(page, text, offset=90):
    page.evaluate("""([t, off]) => {
      const el = [...document.querySelectorAll('.step-name')].find(e => e.textContent.trim() === t);
      if (el) window.scrollTo({top: el.getBoundingClientRect().top + window.scrollY - off, behavior: 'smooth'});
    }""", [text, offset])


def recorded_shots(ctx):
    th = "&theater=1&clean=1"
    # A: hook - the two-object drawer task, slow orbit
    record(ctx, "a_hook", f"/?run={DRAWER2}{th}&orbit=1", 20.5)
    # E1: tennis ball attempt 1 from the start: released too high, rolls off, red label
    record(ctx, "e1_fail", f"/?run={TENNIS}&attempt=1{th}", 8.5)
    # E2: the Critic card in the timeline
    record(ctx, "e2_critic", f"/?run={TENNIS}", 8, before=lambda p: (scroll_to_text(p, "Critic"), p.wait_for_timeout(900)))
    # E3: attempt 2 final state, green label, orbit
    record(ctx, "e3_fixed", f"/?run={TENNIS}&attempt=2&t=99{th}&orbit=1", 5)
    # F1: the "Beyond pick & place" tab
    def tab(page, shot):
        shot.wait(0.8)
        page.click("text=Beyond pick & place")
        shot.wait(2.2)
    record(ctx, "f1_tab", "/", 3, during=tab)
    # F2: drawer task, F3: knock over
    record(ctx, "f2_drawer", f"/?run={DRAWER1}{th}", 16)
    record(ctx, "f3_knock", f"/?run={KNOCK}{th}", 4.2)


def live_shots(ctx):
    page = ctx.new_page()
    page.goto(BASE + "/")
    page.wait_for_timeout(3000)
    shot = Shot(page, FRAMES / "c_live", every_nth=2)
    shot.start()
    shot.wait(1.0)
    page.click("#prompt")
    page.keyboard.type(LIVE_PROMPT, delay=28)
    shot.wait(0.6)
    page.click("#runbtn")
    t0, last_h = time.time(), 0
    while time.time() - t0 < 400:
        shot.wait(1.0)
        done = page.evaluate("state.done !== null || state.error !== null")
        h = page.evaluate("document.body.scrollHeight")
        if h != last_h:  # follow the newest step
            page.evaluate("window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'})")
            last_h = h
        if done:
            break
    shot.wait(2.5)
    run_id = page.evaluate("state.runId")
    shot.stop()
    dur = to_mp4(FRAMES / "c_live", CLIPS / "c_live.mp4")
    shutil.rmtree(FRAMES / "c_live")
    page.close()
    print(f"c_live: {dur:.1f} s, run {run_id}", flush=True)
    (HERE / "live_run.json").write_text(json.dumps({"run_id": run_id, "seconds": dur}))
    result_shots(ctx, run_id)


def result_shots(ctx, run_id):
    # D1: the live run's result, theater + orbit, ending on the goal labels
    record(ctx, "d1_result", f"/?run={run_id}&theater=1&clean=1&orbit=1", 19)
    # D2: the stress-test card and robustness line, then G1: costs and model time
    record(ctx, "d2_stress", f"/?run={run_id}", 6, before=lambda p: (scroll_to_text(p, "Stress test", 200), p.wait_for_timeout(900)))
    record(ctx, "g1_costs", f"/?run={run_id}", 6,
           before=lambda p: (p.evaluate("window.scrollTo({top: document.body.scrollHeight})"), p.wait_for_timeout(900)))


if __name__ == "__main__":
    pw, b, ctx = browser()
    try:
        if sys.argv[1:] == ["live"]:
            live_shots(ctx)
        elif sys.argv[1:2] == ["result"]:
            result_shots(ctx, json.loads((HERE / "live_run.json").read_text())["run_id"])
        else:
            recorded_shots(ctx)
    finally:
        b.close()
        pw.stop()
