"""Assemble demo video v2: shots fitted to narration, speed tags, burned-in captions, .srt.

Output: out/spec2sim_demo_v2.mp4 and out/spec2sim_demo_v2.srt
"""
import json
import re
import subprocess
from pathlib import Path

import cards
import narrate
import numpy as np
import soundfile as sf
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
C1, C2, CARDS, AUDIO, PARTS, OUT = (HERE / d for d in ("clips", "clips2", "cards", "audio", "parts2", "out"))
CAPS = HERE / "caps"
for d in (PARTS, OUT, CAPS):
    d.mkdir(exist_ok=True)
FPS = 30
DUR = json.loads((AUDIO / "durations.json").read_text())
ORDER = ["a_hook", "b_problem", "c_live", "d_tavily", "e_repair", "f_robust", "g_beyond", "h_data", "i_fit",
         "j_rigor", "k_close"]


def probe(p: Path) -> float:
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", str(p)]).decode())


def clip(path, out_s, start=0.0, end=None, tag=None, band=False):
    """band=True: UI close-up; keep the top of the frame and leave a dark caption band below it."""
    end = end if end is not None else probe(path)
    return {"kind": "clip", "src": path, "start": start, "end": min(end, probe(path)), "out": out_s, "tag": tag, "band": band}


def still(name, out_s):
    return {"kind": "card", "src": CARDS / f"{name}.png", "out": out_s}


def segments():
    T = {k: DUR[k] + 0.5 for k in ORDER}
    T["k_close"] = DUR["k_close"] + 1.6
    live = C2 / "c_live.mp4"
    c_rest = T["c_live"] - 5.5 - 3.4 - 4.0
    return [
        ("a_hook", [clip(C1 / "a_hook.mp4", T["a_hook"], 0, 14.2, tag="speed")]),
        ("b_problem", [clip(C2 / "b_problem.mp4", T["b_problem"])]),
        ("c_live", [clip(live, 5.5, 0, 5.5, tag="LIVE"), clip(live, c_rest, 5.5, None, tag="LIVE speed"),
                    clip(C2 / "c2_ground.mp4", band=True, out_s=3.4), clip(C2 / "c3_result.mp4", 4.0)]),
        ("d_tavily", [clip(C2 / "d1_infeasible.mp4", band=True, out_s=T["d_tavily"] / 2), clip(C2 / "d2_lime.mp4", band=True, out_s=T["d_tavily"] / 2)]),
        ("e_repair", [clip(C2 / "e1_compare.mp4", 10.5), clip(C2 / "e2_critic.mp4", band=True, out_s=5.5),
                      clip(C2 / "e3_compare_end.mp4", T["e_repair"] - 16.0)]),
        ("f_robust", [still("e4_stress", 5.5), clip(C2 / "f2_try.mp4", T["f_robust"] - 5.5, tag="speed")]),
        ("g_beyond", [clip(C2 / "g1_drawer.mp4", 4.4, tag="speed"), clip(C2 / "g2_knock.mp4", T["g_beyond"] - 4.4, 0, 4.2)]),
        ("h_data", [clip(C2 / "h1_data.mp4", T["h_data"] - 6.8, tag="speed"), clip(C2 / "h2_lerobot.mp4", 6.8)]),
        ("i_fit", [clip(C2 / "i_fit.mp4", T["i_fit"])]),
        ("j_rigor", [clip(C2 / "j1_cascade.mp4", 5.5), still("g2_eval", T["j_rigor"] - 5.5)]),
        ("k_close", [still("h_close", T["k_close"])]),
    ]


def tag_text(p):
    s = (p["end"] - p["start"]) / p["out"]
    sp = f"{s:.1f}× speed" if s > 1.15 else ""
    if p["tag"] and p["tag"].startswith("LIVE"):
        return "● LIVE" + (f" · {sp}" if sp else "")
    return sp or None


def render(p, i, tag_png):
    out = PARTS / f"{i:02d}.mp4"
    common = ["-r", str(FPS), "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-an"]
    if p["kind"] == "card":
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-t", f"{p['out']:.3f}", "-i", str(p["src"]),
                        "-vf", "scale=1920:1080,format=yuv420p", *common, str(out)], check=True)
        return out
    speed = (p["end"] - p["start"]) / p["out"]
    vf = ["scale=1920:1080:flags=lanczos"] + (["crop=1920:944:0:0", "pad=1920:1080:0:0:color=0x0b1116"] if p.get("band") else []) + [f"setpts=(PTS-STARTPTS)/{speed:.5f}", f"fps={FPS}",
          "tpad=stop_mode=clone:stop_duration=30", f"trim=duration={p['out']:.3f}"]
    inputs = ["-ss", f"{p['start']:.3f}", "-to", f"{p['end']:.3f}", "-i", str(p["src"])]
    if tag_png:
        fc = f"[0:v]{','.join(vf)}[v];[v][1:v]overlay=0:0,format=yuv420p[o]"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-i", str(tag_png), "-filter_complex", fc,
                        "-map", "[o]", *common, str(out)], check=True)
    else:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-vf", ",".join(vf + ["format=yuv420p"]),
                        *common, str(out)], check=True)
    return out


def caption_chunks(text):
    """Sentences, with long ones split at commas/colons into readable lines (<= ~78 chars)."""
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = s.strip()
        if not s:
            continue
        if len(s) <= 78:
            out.append(s)
            continue
        cur = ""
        for piece in re.split(r"(?<=[,:])\s+", s):
            if cur and len(cur) + len(piece) + 1 > 78:
                out.append(cur)
                cur = piece
            else:
                cur = f"{cur} {piece}".strip()
        if cur:
            out.append(cur)
    return out


def srt_time(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{min(999, round((s % 1) * 1000)):03d}"


def main():
    segs = segments()
    # speed/LIVE tags
    tags = {}
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True)
        page = b.new_page(viewport={"width": 1920, "height": 1080})
        for _, parts in segs:
            for p in parts:
                if p["kind"] == "clip" and p.get("tag"):
                    t = tag_text(p)
                    if t and t not in tags:
                        f = CARDS / f"tag2_{len(tags)}.png"
                        page.set_content(cards.HEAD + cards.tag(t)); page.wait_for_timeout(400)
                        page.screenshot(path=str(f), omit_background=True)
                        tags[t] = f
        # captions
        cues, t = [], 0.0
        starts = {}
        for name, parts in segs:
            starts[name] = t
            t += sum(p["out"] for p in parts)
        total = t
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
            page.wait_for_timeout(250)
            page.screenshot(path=str(f), omit_background=True)
            cap_files.append(f)
        b.close()

    files, i = [], 0
    for _, parts in segs:
        for p in parts:
            tp = tags.get(tag_text(p)) if p["kind"] == "clip" and p.get("tag") else None
            files.append(render(p, i, tp))
            i += 1
    lst = PARTS / "list.txt"
    lst.write_text("".join(f"file '{f}'\n" for f in files))
    video = PARTS / "video.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(video)], check=True)

    sr = 24000
    track = np.zeros(int((total + 1) * sr), dtype=np.float32)
    for name, _ in segs:
        a, _sr = sf.read(AUDIO / f"{name}.wav", dtype="float32")
        k = int(starts[name] * sr)
        track[k:k + len(a)] += a[: len(track) - k]
    sf.write(PARTS / "narration.wav", track[: int(total * sr)], sr)

    # Burn captions: one overlay per cue, enabled only during its time.
    inputs = ["-i", str(video), "-i", str(PARTS / "narration.wav")]
    for f in cap_files:
        inputs += ["-i", str(f)]
    chain, last = [], "[0:v]"
    for k, (a, b2, _) in enumerate(cues):
        nxt = f"[c{k}]"
        chain.append(f"{last}[{k + 2}:v]overlay=0:0:enable='between(t,{a:.2f},{b2:.2f})'{nxt}")
        last = nxt
    chain.append(f"{last}fade=t=in:st=0:d=0.5,fade=t=out:st={total - 1.2:.2f}:d=1.2,format=yuv420p[vout]")
    out = OUT / "spec2sim_demo_v2.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", ";".join(chain),
                    "-map", "[vout]", "-map", "1:a", "-af", f"afade=t=out:st={total - 1.0:.2f}:d=1.0",
                    "-c:v", "libx264", "-crf", "19", "-preset", "slow", "-c:a", "aac", "-b:a", "160k",
                    "-movflags", "+faststart", "-shortest", str(out)], check=True)
    (OUT / "spec2sim_demo_v2.srt").write_text("\n".join(
        f"{k + 1}\n{srt_time(a)} --> {srt_time(b2)}\n{text}\n" for k, (a, b2, text) in enumerate(cues)))
    print(f"total {total:.1f} s -> {out}")
    for name, _ in segs:
        print(f"  {starts[name]:6.1f}s  {name}")


if __name__ == "__main__":
    main()
