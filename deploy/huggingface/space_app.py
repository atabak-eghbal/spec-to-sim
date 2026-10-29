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

import uvicorn  # noqa: E402

try:  # only present on ZeroGPU hardware
    import spaces

    @spaces.GPU(duration=1)
    def _gpu_placeholder() -> int:
        return 0
except ImportError:
    pass

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")), proxy_headers=True,
                forwarded_allow_ips="*")
