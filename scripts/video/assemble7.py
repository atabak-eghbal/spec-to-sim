"""Assemble demo video v7 (the redesigned app and the current pipeline).

v6 over v5: a title card opens the video; every cut is a short crossfade (no hard jump cuts); a
chapter label appears as each section's narration starts; and a section may never be shorter than its
narration (in v5 the stress-test section ran out of footage and its voice overlapped the next one).
Rules kept: 3D replays play at most 1.3x; waiting is removed with labelled cuts at 1x; no frozen footage
except push-in shots. Output: out/spec2sim_demo_v7.mp4 (no subtitles), a copy with a subtitle track, and .srt.
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
C1, C2, C3, C4, CARDS, AUDIO, PARTS, OUT, CAPS = (HERE / d for d in ("clips", "clips2", "clips3", "clips7", "cards",
                                                                     "audio", "parts7", "out", "caps7"))
for d in (PARTS, OUT, CAPS):
    d.mkdir(exist_ok=True)
FPS = 30
MAX_SPEED = 1.3
XF = 0.4  # crossfade between consecutive parts, s
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
    m = dict(json.loads((HERE / "live_run7.json").read_text())["marks"])
    drag = json.loads((HERE / "drag7.json").read_text())
    dat = json.loads((HERE / "data7.json").read_text())
    W = [(0.6, m["submit"] + 1.0, "● LIVE")]
    for k in ("grounder:0:done", "builder:1:done", "runner:1:done", "critic:1:done", "repair:2:done",
              "runner:2:done", "stress:0:done"):
        if k in m:
            W.append((m[k] - 0.3, m[k] + (3.0 if k.startswith("grounder") else 2.3), None))
    parts, prev_end = [], 0.0
    for a, b, tag in W:
        if a < prev_end:  # steps that finish close together: extend the previous window instead
            if b - prev_end < 0.8:
                continue
            p = parts[-1]
            parts[-1] = clip(live, p["start"], b, tag=p["tag"])
            prev_end = b
            continue
        skipped = a - prev_end
        parts.append(clip(live, a, b, tag=tag or (f"● LIVE · ⏭ {skipped:.0f} s skipped" if skipped >= 1 else "● LIVE")))
        prev_end = b
    used = sum(p["out"] for p in parts)
    if T["c_live"] - used < 2.0:
        raise ValueError(f"live windows {used:.1f}s leave no room for the result shot")
    rest = T["c_live"] - used
    parts.append(clip(C4 / "c3_result.mp4", 0, min(rest, probe(C4 / "c3_result.mp4")), out=rest, push=True))
    d3 = 6.0
    d1 = 11.0
    data = C4 / "h1_data.mp4"
    data_len = probe(data)
    a_end = dat["click"] + 2.2
    mid = (dat["click"] + dat["done"]) / 2
    tail = T["h_data"] - 7.0 - a_end - 1.6
    data_parts = [clip(data, 0, a_end), clip(data, mid, mid + 1.6, tag=f"⏭ {mid - a_end:.0f} s skipped"),
                  clip(data, data_len - tail, data_len, tag=f"⏭ {data_len - tail - mid - 1.6:.0f} s skipped")]
    try_a = drag["click"] + 0.4
    try_b = min(T["f_robust"] - 5.0 - try_a, probe(C4 / "f2_try.mp4") - drag["result"])
    e3 = probe(C4 / "e3_compare_end.mp4")
    return [
        ("a_hook", [clip(C4 / "a_hook.mp4", 0, 9.0, out=T["a_hook"], push=True)]),  # cold open on fail -> fix; title as a lower third
        ("b_problem", [clip(C2 / "b_problem.mp4", 0, T["b_problem"])]),
        ("c_live", parts),
        ("d_tavily", [clip(C4 / "d1_ground.mp4", 0, d1, band=True),
                      clip(C4 / "d2_grapefruit.mp4", 0, T["d_tavily"] - d1 - d3, band=True),
                      clip(C4 / "d3_refused.mp4", 0, d3)]),
        ("e_repair", [clip(C4 / "e1_compare.mp4", 0, 6.5),
                      clip(C4 / "e2_critic.mp4", 0, 3.0, band=True),
                      clip(C4 / "e2b_repair.mp4", 0, T["e_repair"] - 6.5 - 3.0 - e3, band=True, push=True),
                      clip(C4 / "e3_compare_end.mp4", 0, e3)]),
        ("f_robust", [card("e4_stress", T["f_robust"] - try_a - try_b), clip(C4 / "f2_try.mp4", 0, try_a),
                      clip(C4 / "f2_try.mp4", drag["result"], drag["result"] + try_b,
                           tag=f"⏭ {drag['result'] - try_a:.0f} s skipped")]),
        ("g_beyond", [clip(C4 / "g1_drawer.mp4", 2.5, 7.5, out=4.0, tag="1.3× speed"),
                      clip(C4 / "g2_knock.mp4", 0.3, 0.3 + T["g_beyond"] - 4.0)]),
        ("h_data", [*data_parts, clip(C4 / "h3_policy.mp4", 0, 7.0, band=True)]),
        ("i_fit", [card("i_harness", T["i_fit"])]),
        ("j_rigor", [clip(C4 / "j1_cascade.mp4", 0, 5.5, push=True), card("g2_eval", T["j_rigor"] - 5.5)]),
        ("k_close", [card("h_close", T["k_close"])]),
    ]


def check_lengths(segs, T):
    """Each section's footage must cover its narration, or the voice runs into the next section."""
    for name, parts in segs:
        got = sum(p["out"] for p in parts)
        if got < T[name] - 0.05:
            raise ValueError(f"{name}: {got:.2f} s of footage for {T[name]:.2f} s of narration")


def render(p, i, tag_png, extra=0.0):
    out = PARTS / f"{i:02d}.mp4"
    p = dict(p, out=p["out"] + extra)  # extra = overlap consumed by the crossfade into the next part
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
        b.close()

    check_lengths(segs, {k: DUR[k] + 0.5 for k in ORDER} | {"k_close": DUR["k_close"] + 1.6})
    flat = [p for _, parts in segs for p in parts]
    # Crossfade only where a shot jumps forward within the same footage (a cut-out wait): between
    # different screens a crossfade leaves two panels on top of each other, so those are hard cuts
    # (a one-frame "fade").
    CUT = 1 / FPS
    fade = [XF if i < len(flat) - 1 and flat[i]["kind"] == "clip" and flat[i + 1]["kind"] == "clip"
            and flat[i]["src"] == flat[i + 1]["src"] else CUT for i in range(len(flat))]
    fade[-1] = 0.0
    # Render each part with a little spare footage beyond its fade, so frame rounding never leaves a
    # part shorter than the next crossfade's offset (which would end the whole video there).
    files = [render(p, i, tags.get(p["tag"]), fade[i] + (0.15 if i < len(flat) - 1 else 0.0)) for i, p in enumerate(flat)]
    sr = 24000
    track = np.zeros(int((total + 1) * sr), dtype=np.float32)
    for name, _ in segs:
        a, _sr = sf.read(AUDIO / f"{name}.wav", dtype="float32")
        k = int(starts[name] * sr)
        track[k:k + len(a)] += a[: len(track) - k]
    sf.write(PARTS / "narration.wav", track[: int(total * sr)], sr)
    srt = OUT / "spec2sim_demo_v7.srt"
    srt.write_text("\n".join(f"{k + 1}\n{srt_time(a)} --> {srt_time(b2)}\n{text}\n" for k, (a, b2, text) in enumerate(cues)))

    # One pass: crossfade every part into the next (part k+1 starts exactly where its planned slot
    # starts, because part k was rendered XF longer), then chapter labels, then the opening/closing fade.
    inputs, chain = [], []
    for f in files:
        inputs += ["-i", str(f)]
    last, t_end = "[0:v]", flat[0]["out"]
    for k in range(1, len(files)):
        chain.append(f"{last}[{k}:v]xfade=transition=fade:duration={fade[k - 1]:.4f}:offset={t_end:.3f}[x{k}]")
        last, t_end = f"[x{k}]", t_end + flat[k]["out"]
    n = len(files)
    for name, _ in segs:
        chip = CARDS / f"chapter_{name}.png"
        if not chip.exists():
            continue
        t0 = starts[name] + 0.15
        dur = 6.0 if name == "a_hook" else 3.2  # the opening title stays up longer
        inputs += ["-loop", "1", "-t", f"{dur}", "-i", str(chip)]
        chain.append(f"[{n}:v]format=rgba,fade=t=in:st=0:d=0.35:alpha=1,fade=t=out:st={dur - 0.4}:d=0.4:alpha=1,"
                     f"setpts=PTS-STARTPTS+{t0:.3f}/TB[c{n}]")
        chain.append(f"{last}[c{n}]overlay=0:0:eof_action=pass[o{n}]")
        last, n = f"[o{n}]", n + 1
    chain.append(f"{last}fade=t=in:st=0:d=0.5,fade=t=out:st={total - 1.2:.2f}:d=1.2,format=yuv420p[vout]")
    inputs += ["-i", str(PARTS / "narration.wav")]
    out = OUT / "spec2sim_demo_v7.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(chain),
                    "-map", "[vout]", "-map", f"{n}:a", "-af", f"afade=t=out:st={total - 1.0:.2f}:d=1.0",
                    "-t", f"{total:.3f}", "-r", str(FPS), "-c:v", "libx264", "-crf", "19", "-preset", "medium",
                    "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(out)], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(out), "-i", str(srt), "-map", "0", "-map", "1",
                    "-c", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng", "-movflags", "+faststart",
                    str(OUT / "spec2sim_demo_v7_subtitle_track.mp4")], check=True)
    print(f"total {total:.1f} s -> {out}")
    for name, parts in segs:
        print(f"  {starts[name]:6.1f}s  {name}: " + ", ".join(
            f"{p['src'].stem}[{p.get('start', 0):.1f}-{p.get('end', 0):.1f}]->{p['out']:.1f}s" for p in parts))


if __name__ == "__main__":
    main()
