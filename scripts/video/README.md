# Demo video pipeline

Builds the demo video from the running app, with no manual editing. Current: **v7** (2:47): title card, chapter labels, crossfades instead of hard cuts, the redesigned app.

1. `capture7.py live "<prompt>" | replays | data | tavily`: Playwright drives Chrome; the DevTools
   screencast records a live run typed into the app and replays of recorded runs, in filming mode
   (`?clean=1`: no playback bar, replays hold their last frame). `capture2.py cards` renders the
   animated slides; `capture2.py` and `capture3.py` hold shared helpers.
2. `narrate.py`: narration with Kokoro TTS (Apache-2.0, runs locally, voice `af_heart`), using
   misaki phonemes with explicit pronunciations for Nemotron, Nebius, MuJoCo, Tavily.
3. `cards.py`: static slides and "N s skipped" tags as 1080p PNGs. Every number on them comes from
   the eval (`docs/eval.md`) or from a recorded run's stress test.
4. `assemble7.py`: fits each shot to its narration. 3D replays play at most 1.3x; waiting is cut at 1x
   with "N s skipped" tags; no shot is longer than its footage. Writes `out/spec2sim_demo_v7.mp4`
   with English subtitles as a separate track (off by default), plus `out/spec2sim_demo_v7.srt` to
   upload to YouTube as captions.

Setup (kept out of the project's dependencies):
```bash
uv venv .venv-video -p 3.11 && uv pip install -p .venv-video/bin/python kokoro-onnx soundfile playwright "misaki[en]"
# plus the spaCy en_core_web_sm wheel, and kokoro-v1.0.onnx + voices-v1.0.bin in models/
uv run uvicorn app.main:app &      # the app must be running on :8000
python capture7.py live "Put the tennis ball and the Rubik's cube into the storage bin, then stack the red block on the blue block"
python capture7.py replays && python capture7.py story && python capture7.py data && python capture7.py tavily && python capture7.py refused
python capture2.py cards && python narrate.py && python cards.py && python assemble7.py
```
