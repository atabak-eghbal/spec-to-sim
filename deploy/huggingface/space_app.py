"""Hugging Face Space entrypoint (Gradio SDK, free CPU Basic hardware).

Docker Spaces need a paid plan, so the demo runs as a Gradio-SDK Space: Hugging Face installs
requirements.txt and packages.txt (OSMesa for headless MuJoCo) and runs this file, which serves the
same FastAPI app as `uvicorn app.main:app` on the port Spaces expect (7860).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for k, v in {"MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "SPEC2SIM_SHADOWS": "0", "SPEC2SIM_VIDEO": "0"}.items():
    os.environ.setdefault(k, v)
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")), proxy_headers=True,
                forwarded_allow_ips="*")
