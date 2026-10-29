"""Hugging Face Space entrypoint (Gradio SDK; free tier = ZeroGPU hardware, PRO = CPU Basic).

Docker Spaces need a paid plan, so the demo runs as a Gradio-SDK Space: Hugging Face installs
requirements.txt and packages.txt (OSMesa for headless MuJoCo) and runs this file, which serves the
same FastAPI app as `uvicorn app.main:app` on the port Spaces expect (7860). Everything runs on CPU.
ZeroGPU Spaces refuse to start without at least one @spaces.GPU function, so a no-op one is declared;
nothing in Spec-to-Sim ever asks for the GPU.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for k, v in {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "SPEC2SIM_SHADOWS": "0", "SPEC2SIM_VIDEO": "0"}.items():
    os.environ.setdefault(k, v)
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

# Pre-built web facts (material friction/bounciness, Franka limits) are uploaded as kb_physics/
# because Hugging Face rejects '.cache/' paths; the grounder reads them from .cache/physics/.
_kb, _cache = ROOT / "kb_physics", ROOT / ".cache" / "physics"
if _kb.is_dir():
    import shutil

    _cache.mkdir(parents=True, exist_ok=True)
    for f in _kb.glob("*.json"):
        if not (_cache / f.name).exists():
            shutil.copy2(f, _cache / f.name)

import uvicorn

try:  # only present on ZeroGPU hardware
    import spaces

    @spaces.GPU(duration=1)
    def _gpu_placeholder() -> int:
        return 0

    # `spaces` reports the GPU functions to ZeroGPU when a *Gradio* app launches; this app is plain
    # FastAPI, so run that start-up report directly (else the Space is stopped: "No @spaces.GPU
    # function detected during startup").
    from spaces import zero as _zero

    if hasattr(_zero, "startup"):
        _zero.startup()
except ImportError:
    pass

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")), proxy_headers=True,
                forwarded_allow_ips="*")
