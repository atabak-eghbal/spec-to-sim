"""Screen capture helpers: Chrome DevTools screencast -> timestamped JPEG frames -> MP4."""
import base64
import json
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://localhost:8000"
W, H, DSF = 1280, 720, 1.5  # CSS viewport; frames come out at 1920x1080


class Shot:
    def __init__(self, page, out: Path, every_nth: int = 2):
        self.page, self.out, self.every_nth = page, out, every_nth
        out.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.jpg"):
            f.unlink()
        self.frames = []
        self.cdp = page.context.new_cdp_session(page)
        self.cdp.on("Page.screencastFrame", self._frame)

    def _frame(self, ev):
        if getattr(self, "stopped", False):  # a frame can arrive after stopScreencast
            return
        i = len(self.frames)
        p = self.out / f"{i:06d}.jpg"
        p.write_bytes(base64.b64decode(ev["data"]))
        self.frames.append((p.name, time.time()))
        self.cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]})

    def start(self):
        self.t0 = time.time()
        self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 82, "everyNthFrame": self.every_nth,
                                               "maxWidth": int(W * DSF), "maxHeight": int(H * DSF)})

    def wait(self, s):
        end = time.time() + s
        while time.time() < end:
            self.page.wait_for_timeout(50)

    def stop(self) -> Path:
        self.cdp.send("Page.stopScreencast")
        self.stopped = True
        self.t1 = time.time()
        if not self.frames:  # a page that never changes sends no screencast frames: use a still
            p = self.out / "000000.jpg"
            self.page.screenshot(path=str(p), type="jpeg", quality=90)
            self.frames.append((p.name, self.t0))
        (self.out / "frames.json").write_text(json.dumps({"frames": self.frames, "t0": self.t0, "t1": self.t1}))
        return self.out


def to_mp4(shot_dir: Path, out: Path, speed: float = 1.0, fps: int = 30) -> float:
    """Frames (variable timing) -> constant-fps MP4. Returns duration in seconds."""
    shot_dir = shot_dir.resolve()
    meta = json.loads((shot_dir / "frames.json").read_text())
    frames, t1 = meta["frames"], meta["t1"]
    lines = []
    for k, (name, t) in enumerate(frames):
        nxt = frames[k + 1][1] if k + 1 < len(frames) else t1
        lines.append(f"file '{shot_dir / name}'\nduration {max(0.001, (nxt - t) / speed):.4f}")
    lines.append(f"file '{shot_dir / frames[-1][0]}'")
    lst = shot_dir / "list.txt"
    lst.write_text("\n".join(lines))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-vf", f"fps={fps},scale=1920:1080:flags=lanczos,format=yuv420p", "-c:v", "libx264",
                    "-crf", "18", "-preset", "medium", str(out)], check=True)
    return (t1 - frames[0][1]) / speed


def browser():
    pw = sync_playwright().start()
    b = pw.chromium.launch(channel="chrome", headless=True,
                           args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist", "--hide-scrollbars"])
    ctx = b.new_context(viewport={"width": W, "height": H}, device_scale_factor=DSF)
    return pw, b, ctx
