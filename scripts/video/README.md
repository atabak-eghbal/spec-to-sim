# Demo video pipeline

Builds the demo video from the running app, with no manual editing:

1. `capture.py`: Playwright drives Chrome; the DevTools screencast records shots of recorded
   runs (`?run=...&theater=1&orbit=1&t=...`) plus one live run typed into the app.
2. `narrate.py`: narration with Kokoro TTS (Apache-2.0, runs locally, voice `af_heart`), using
   misaki phonemes with explicit pronunciations for Nemotron, Nebius, MuJoCo, Tavily.
3. `cards.py`: title/info cards and "N× speed" tags rendered as 1080p PNGs.
4. `assemble.py`: fits each shot to its narration, overlays speed tags, mixes the audio,
   writes `out/spec2sim_demo.mp4` and captions `out/spec2sim_demo.srt`.

Setup (kept out of the project's dependencies):
```bash
uv venv .venv-video -p 3.11 && uv pip install -p .venv-video/bin/python kokoro-onnx soundfile playwright "misaki[en]"
# plus the spaCy en_core_web_sm wheel, and kokoro-v1.0.onnx + voices-v1.0.bin in models/
uv run uvicorn app.main:app &      # the app must be running on :8000
python capture.py && python capture.py live && python narrate.py && python cards.py && python assemble.py
```
