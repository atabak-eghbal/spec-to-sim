"""Assemble the demo video: clips + cards fitted to the narration, speed tags, audio, captions.

Output: out/spec2sim_demo.mp4 (1080p30, AAC) and out/spec2sim_demo.srt
"""
import json
import re
import subprocess
from pathlib import Path

import cards
import narrate
import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
CLIPS, CARDS, AUDIO, PARTS, OUT = (HERE / d for d in ("clips", "cards", "audio", "parts", "out"))
PARTS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
FPS = 30
DUR = json.loads((AUDIO / "durations.json").read_text())
LIVE = json.loads((HERE / "live_run.json").read_text())
NARR = dict(narrate.SEGMENTS, d_result=(
    "The first scene put the two bins on top of each other, so code rejected it, and the plan was repaired. "
    "Attempt two works. Then the plan is re-run sixteen times, with the objects nudged, made heavier, or more slippery. "
    "Sixteen out of sixteen. It's robust, not lucky."))


def probe(p: Path) -> float:
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", str(p)]).decode())


def clip(name, out_s, start=0.0, end=None, crop=None, tag=None):
    """A slice of clips/<name>.mp4 retimed to last out_s seconds."""
    src = CLIPS / f"{name}.mp4"
    end = end if end is not None else probe(src)
    return {"kind": "clip", "src": src, "start": start, "end": end, "out": out_s, "crop": crop, "tag": tag}


def card(name, out_s):
    return {"kind": "card", "src": CARDS / f"{name}.png", "out": out_s}


def segments():
    T = {k: v + 0.5 for k, v in DUR.items()}
    live_len = probe(CLIPS / "c_live.mp4")
    c2 = T["c_live"] + 0.3 - 5.5
    d2 = 5.8
    return [
        ("a_hook", [clip("a_hook", T["a_hook"], tag="auto")]),
        ("b_problem", [card("b_problem", T["b_problem"])]),
        ("c_live", [clip("c_live", 5.5, 0, 5.5), clip("c_live", c2, 5.5, live_len, tag="auto")]),
        ("d_result", [clip("d1_result", T["d_result"] - d2, tag="auto"), clip("d2_stress", d2, 0, d2)]),
        ("e_repair", [clip("e1_fail", 8.5), clip("e2_critic", T["e_repair"] - 8.5 - 5.0 - 4.0, 0, 7.9, crop=(1280, 720, 20, 70)),
                      clip("e3_fixed", 5.0), card("e4_stress", 4.0)]),
        ("f_beyond", [clip("f1_tab", 2.2), clip("f2_drawer", T["f_beyond"] - 2.2 - 3.6 - 3.3, tag="auto"),
                      clip("f3_knock", 3.6, 0, 3.6), card("f4_export", 3.3)]),
        ("g_rigor", [clip("g1_costs", 5.6, 0, 5.6), card("g2_eval", T["g_rigor"] - 5.6)]),
        ("h_close", [card("h_close", DUR["h_close"] + 1.6)]),
    ]


def speed_label(p):
    s = (p["end"] - p["start"]) / p["out"]
    return f"{s:.1f}× speed" if s > 1.15 else None


def render(p, i):
    out = PARTS / f"{i:02d}.mp4"
    common = ["-r", str(FPS), "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-an"]
    if p["kind"] == "card":
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-t", f"{p['out']:.3f}", "-i", str(p["src"]),
                        "-vf", "scale=1920:1080,format=yuv420p", *common, str(out)], check=True)
        return out
    speed = (p["end"] - p["start"]) / p["out"]
    vf = []
    if p["crop"]:
        w, h, x, y = p["crop"]
        vf.append(f"crop={w}:{h}:{x}:{y}")
    vf += ["scale=1920:1080:flags=lanczos", f"setpts=(PTS-STARTPTS)/{speed:.5f}", f"fps={FPS}",
           "tpad=stop_mode=clone:stop_duration=2", f"trim=duration={p['out']:.3f}"]
    inputs = ["-ss", f"{p['start']:.3f}", "-to", f"{p['end']:.3f}", "-i", str(p["src"])]
    label = speed_label(p) if p.get("tag") == "auto" else None
    if label:
        tag_png = CARDS / f"tag_{label.split()[0].replace('×', 'x')}.png"
        if not tag_png.exists():
            cards.main(tags=[label])
        fc = f"[0:v]{','.join(vf)}[v];[v][1:v]overlay=0:0,format=yuv420p[o]"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-i", str(tag_png), "-filter_complex", fc,
                        "-map", "[o]", *common, str(out)], check=True)
    else:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-vf", ",".join(vf + ["format=yuv420p"]),
                        *common, str(out)], check=True)
    return out


def srt_time(t):
    h, m = int(t // 3600), int(t % 3600 // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int(round((s % 1) * 1000)):03d}"


def main():
    segs = segments()
    files, starts, t = [], {}, 0.0
    i = 0
    for name, parts in segs:
        starts[name] = t
        for p in parts:
            files.append(render(p, i))
            t += p["out"]
            i += 1
    total = t
    lst = PARTS / "list.txt"
    lst.write_text("".join(f"file '{f}'\n" for f in files))
    video = PARTS / "video.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(video)], check=True)

    # Audio: each narration starts with its segment.
    sr = 24000
    track = np.zeros(int((total + 1) * sr), dtype=np.float32)
    for name, _ in segs:
        a, r = sf.read(AUDIO / f"{name}.wav", dtype="float32")
        assert r == sr
        k = int(starts[name] * sr)
        track[k:k + len(a)] += a[: len(track) - k]
    sf.write(PARTS / "narration.wav", track[: int(total * sr)], sr)

    out = OUT / "spec2sim_demo.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-i", str(PARTS / "narration.wav"),
                    "-vf", f"fade=t=in:st=0:d=0.6,fade=t=out:st={total - 1.2:.2f}:d=1.2",
                    "-af", f"afade=t=out:st={total - 1.0:.2f}:d=1.0", "-c:v", "libx264", "-crf", "19", "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-shortest", str(out)],
                   check=True)

    # Captions: sentences spread over each narration by length.
    cues, n = [], 1
    for name, _ in segs:
        text = NARR[name]
        sents = [s.strip() for s in re.split(r"(?<=[.:!?])\s+", text) if s.strip()]
        span, t0 = DUR[name] - 0.5, starts[name] + 0.25
        total_chars = sum(len(s) for s in sents)
        for s in sents:
            d = span * len(s) / total_chars
            cues.append(f"{n}\n{srt_time(t0)} --> {srt_time(t0 + d)}\n{s}\n")
            n, t0 = n + 1, t0 + d
    (OUT / "spec2sim_demo.srt").write_text("\n".join(cues))
    print(f"total {total:.1f} s -> {out}")
    for name, _ in segs:
        print(f"  {starts[name]:6.1f}s  {name}")


if __name__ == "__main__":
    main()
