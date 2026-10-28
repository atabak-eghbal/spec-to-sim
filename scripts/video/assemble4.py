"""Assemble demo video v4 (filming mode: no playback bars, replays hold their last frame): no frozen or rushed footage.

Rules enforced in code:
- every part plays at 1.0x except 3D replays capped at 1.3x; waiting is removed with jump cuts at
  1x, each labelled with the skipped time;
- a part may never be longer than its footage (no frozen frames); static shots (cards, UI close-ups)
  get a slow push-in instead.
Output: out/spec2sim_demo_v3.mp4 and .srt
"""
import json
import subprocess
from pathlib import Path

import cards
import narrate
import numpy as np
import soundfile as sf
from assemble2 import caption_chunks, srt_time
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
C1, C2, C3, C4, CARDS, AUDIO, PARTS, OUT, CAPS = (HERE / d for d in ("clips", "clips2", "clips3", "clips4", "cards",
                                                                     "audio", "parts4", "out", "caps4"))
for d in (PARTS, OUT, CAPS):
    d.mkdir(exist_ok=True)
FPS = 30
MAX_SPEED = 1.3
DUR = json.loads((AUDIO / "durations.json").read_text())
ORDER = ["a_hook", "b_problem", "c_live", "d_tavily", "e_repair", "f_robust", "g_beyond", "h_data", "i_fit",
         "j_rigor", "k_close"]


def probe(p: Path) -> float:
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", str(p)]).decode())


def clip(path, start, end, out=None, tag=None, band=False, push=False):
    foot = probe(path)
    end = min(end, foot)
    out = out if out is not None else end - start
    speed = (end - start) / out
    if speed > MAX_SPEED + 1e-6:
        raise ValueError(f"{path.name}: {speed:.2f}x is too fast (max {MAX_SPEED}x)")
    if speed < 0.995 and not push:
        raise ValueError(f"{path.name}: would freeze ({end - start:.2f} s of footage for {out:.2f} s)")
    return {"kind": "clip", "src": path, "start": start, "end": end, "out": out, "tag": tag, "band": band, "push": push}


def card(name, out):
    return {"kind": "card", "src": CARDS / f"{name}.png", "out": out, "tag": None, "band": False, "push": True}


def segments():
    T = {k: DUR[k] + 0.5 for k in ORDER}
    T["k_close"] = DUR["k_close"] + 1.6
    live = C4 / "c_live.mp4"
    m = dict(json.loads((HERE / "live_run4.json").read_text())["marks"])
    drag = json.loads((HERE / "drag4.json").read_text())
    dat = json.loads((HERE / "data4.json").read_text())
    W = [(0.6, m["submit"] + 1.0, "● LIVE")]
    for k in ("planner:0:done", "grounder:0:done", "builder:1:done", "runner:1:done", "critic:1:done",
              "repair:2:done", "runner:2:done", "stress:0:done"):
        if k in m:
            W.append((m[k] - 0.3, m[k] + (3.2 if k.startswith("grounder") else 2.5), None))
    parts, prev_end = [], 0.0
    for a, b, tag in W:
        parts.append(clip(live, a, b, tag=tag or f"● LIVE · ⏭ {a - prev_end:.0f} s skipped"))
        prev_end = b
    used = sum(p["out"] for p in parts)
    if T["c_live"] - used < 2.0:
        raise ValueError(f"live windows {used:.1f}s leave no room for the result shot")
    parts.append(clip(C4 / "c3_result.mp4", 0, T["c_live"] - used))
    d_half = T["d_tavily"] / 2
    data = C4 / "h1_data.mp4"
    data_len = probe(data)
    a_end = dat["click"] + 2.2
    mid = (dat["click"] + dat["done"]) / 2
    tail = T["h_data"] - 6.85 - a_end - 1.6
    data_parts = [clip(data, 0, a_end), clip(data, mid, mid + 1.6, tag=f"⏭ {mid - a_end:.0f} s skipped"),
                  clip(data, data_len - tail, data_len, tag=f"⏭ {data_len - tail - mid - 1.6:.0f} s skipped")]
    try_a = drag["click"] + 0.4
    try_b = T["f_robust"] - 5.0 - try_a
    return [
        ("a_hook", [clip(C4 / "a_hook.mp4", 1.5, 1.5 + T["a_hook"])]),
        ("b_problem", [clip(C2 / "b_problem.mp4", 0, T["b_problem"])]),
        ("c_live", parts),
        ("d_tavily", [clip(C4 / "d1_ground.mp4", 0, d_half, band=True),
                      clip(C3 / "d2_grapefruit.mp4", 0, d_half, band=True)]),
        ("e_repair", [clip(C4 / "e1_compare.mp4", 0, 10.5), clip(C2 / "e2_critic.mp4", 0, 6.5, out=5.5, band=True),
                      clip(C4 / "e3_compare_end.mp4", 0, T["e_repair"] - 16.0)]),
        ("f_robust", [card("e4_stress", 5.0), clip(C4 / "f2_try.mp4", 0, try_a),
                      clip(C4 / "f2_try.mp4", drag["result"], drag["result"] + try_b,
                           tag=f"⏭ {drag['result'] - try_a:.0f} s skipped")]),
        ("g_beyond", [clip(C4 / "g1_drawer.mp4", 2.5, 7.5, out=4.0, tag="1.3× speed"),
                      clip(C4 / "g2_knock.mp4", 0.3, 0.3 + T["g_beyond"] - 4.0)]),
        ("h_data", [*data_parts, clip(C2 / "h2_lerobot.mp4", 0, 6.8, out=6.85, push=True)]),
        ("i_fit", [clip(C2 / "i_fit.mp4", 0, 18.7, out=T["i_fit"], push=True)]),
        ("j_rigor", [clip(C4 / "j1_cascade.mp4", 0, 5.5, push=True), card("g2_eval", T["j_rigor"] - 5.5)]),
        ("k_close", [card("h_close", T["k_close"])]),
    ]


def render(p, i, tag_png):
    out = PARTS / f"{i:02d}.mp4"
    common = ["-r", str(FPS), "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-an"]
    push = (f"scale=w='2*trunc(1920*(1+0.05*t/{p['out']:.3f})/2)':h=-2:eval=frame,"
            "crop=1920:1080:(in_w-1920)/2:(in_h-1080)/2") if p["push"] else None
    band = "crop=1920:944:0:0,pad=1920:1080:0:0:color=0x0b1116" if p["band"] else None
    if p["kind"] == "card":
        vf = ",".join(x for x in ["scale=1920:1080", f"fps={FPS}", push, "format=yuv420p"] if x)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-t", f"{p['out']:.3f}", "-i", str(p["src"]),
                        "-vf", vf, *common, str(out)], check=True)
        return out
    speed = (p["end"] - p["start"]) / p["out"]
    vf = ["scale=1920:1080:flags=lanczos", f"setpts=(PTS-STARTPTS)/{speed:.5f}", f"fps={FPS}",
          "tpad=stop_mode=clone:stop_duration=1", f"trim=duration={p['out']:.3f}", "setpts=PTS-STARTPTS"]
    vf += [x for x in (push, band) if x]
    inputs = ["-ss", f"{p['start']:.3f}", "-to", f"{p['end']:.3f}", "-i", str(p["src"])]
    if tag_png:
        fc = f"[0:v]{','.join(vf)}[v];[v][1:v]overlay=0:0,format=yuv420p[o]"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-i", str(tag_png), "-filter_complex", fc,
                        "-map", "[o]", *common, str(out)], check=True)
    else:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-vf", ",".join(vf + ["format=yuv420p"]),
                        *common, str(out)], check=True)
    return out


def main():
    segs = segments()
    starts, t = {}, 0.0
    for name, parts in segs:
        starts[name] = t
        t += sum(p["out"] for p in parts)
    total = t
    tags, cues = {}, []
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        page = b.new_page(viewport={"width": 1920, "height": 1080})
        for _, parts in segs:
            for p in parts:
                if p["tag"] and p["tag"] not in tags:
                    f = CARDS / f"tag3_{len(tags)}.png"
                    page.set_content(cards.HEAD + cards.tag(p["tag"])); page.wait_for_timeout(300)
                    page.screenshot(path=str(f), omit_background=True)
                    tags[p["tag"]] = f
        for name, _ in segs:
            chunks = caption_chunks(narrate.SEGMENTS[name])
            span, t0 = DUR[name] - 0.5, starts[name] + 0.25
            n = sum(len(c) for c in chunks)
            for c in chunks:
                d = span * len(c) / n
                cues.append((t0, t0 + d, c))
                t0 += d
        cap_files = []
        for k, (_, _, text) in enumerate(cues):
            f = CAPS / f"cap_{k:03d}.png"
            page.set_content(cards.HEAD + f"""<style>body{{background:transparent!important}}</style>
<div style="position:absolute;left:0;right:0;bottom:58px;text-align:center"><span style="display:inline-block;max-width:1600px;
font:500 38px/1.3 'IBM Plex Sans';color:#fff;background:rgba(8,12,16,0.78);padding:12px 26px;border-radius:10px">{text}</span></div>""")
            page.wait_for_timeout(200)
            page.screenshot(path=str(f), omit_background=True)
            cap_files.append(f)
        b.close()

    files, i = [], 0
    for _, parts in segs:
        for p in parts:
            files.append(render(p, i, tags.get(p["tag"])))
            i += 1
    (PARTS / "list.txt").write_text("".join(f"file '{f}'\n" for f in files))
    video = PARTS / "video.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(PARTS / "list.txt"),
                    "-c", "copy", str(video)], check=True)
    sr = 24000
    track = np.zeros(int((total + 1) * sr), dtype=np.float32)
    for name, _ in segs:
        a, _sr = sf.read(AUDIO / f"{name}.wav", dtype="float32")
        k = int(starts[name] * sr)
        track[k:k + len(a)] += a[: len(track) - k]
    sf.write(PARTS / "narration.wav", track[: int(total * sr)], sr)
    inputs = ["-i", str(video), "-i", str(PARTS / "narration.wav")]
    for f in cap_files:
        inputs += ["-i", str(f)]
    chain, last = [], "[0:v]"
    for k, (a, b2, _) in enumerate(cues):
        chain.append(f"{last}[{k + 2}:v]overlay=0:0:enable='gte(t,{a:.3f})*lt(t,{b2 - 0.04:.3f})'[c{k}]")
        last = f"[c{k}]"
    chain.append(f"{last}fade=t=in:st=0:d=0.5,fade=t=out:st={total - 1.2:.2f}:d=1.2,format=yuv420p[vout]")
    out = OUT / "spec2sim_demo_v4.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(chain),
                    "-map", "[vout]", "-map", "1:a", "-af", f"afade=t=out:st={total - 1.0:.2f}:d=1.0",
                    "-c:v", "libx264", "-crf", "19", "-preset", "slow", "-c:a", "aac", "-b:a", "160k",
                    "-movflags", "+faststart", "-shortest", str(out)], check=True)
    (OUT / "spec2sim_demo_v4.srt").write_text("\n".join(
        f"{k + 1}\n{srt_time(a)} --> {srt_time(b2)}\n{text}\n" for k, (a, b2, text) in enumerate(cues)))
    print(f"total {total:.1f} s -> {out}")
    for name, parts in segs:
        print(f"  {starts[name]:6.1f}s  {name}: " + ", ".join(
            f"{p['src'].stem}[{p.get('start', 0):.1f}-{p.get('end', 0):.1f}]->{p['out']:.1f}s" for p in parts))


if __name__ == "__main__":
    main()
