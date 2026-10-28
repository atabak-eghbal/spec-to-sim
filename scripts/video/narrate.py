"""Narration with Kokoro (Apache-2.0, runs locally). One WAV per segment."""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from kokoro_onnx import Kokoro
from misaki import en

# Pronunciations for names the dictionary doesn't know (misaki inline phoneme syntax).
SAY = {
    "Nemotron": "[Nemotron](/nˈɛmətɹɑn/)", "Nebius": "[Nebius](/nˈɛbiəs/)",
    "MuJoCo": "[MuJoCo](/mjˈuʤˌOkO/)", "Tavily": "[Tavily](/tˈævᵻli/)", "Rubik's": "[Rubik's](/ɹˈubɪks/)",
}


VOICE, SPEED = "af_heart", 1.0
HERE = Path(__file__).parent
OUT = HERE / "audio"

SEGMENTS = {
    "a_hook": "Put the tennis ball and the Rubik's cube in the drawer, and close it. One sentence. "
              "About a minute later, a robot does it in a physics simulation, and every result is checked.",
    "b_problem": "Robotics teams spend days building simulation scenes by hand: measuring objects, "
                 "writing scene files, writing success checks. Spec-to-Sim does it from plain English, "
                 "with NVIDIA Nemotron models on Nebius Token Factory.",
    "c_live": "Here's a live run. Nemotron Super turns the request into goals that code can check. "
              "Tavily searches the web for each object's real size and weight, and Nemotron Nano reads off "
              "the numbers, with sources. Super then places the objects and plans the motion, as data, never code. "
              "MuJoCo runs the simulation, and code, not a model, checks every goal.",
    "e_repair": "When it fails, it fixes itself. Here the ball was released a centimetre and a half too high, "
                "and rolled off. Code measures the miss. Nemotron Super V L looks at the frames and explains why. "
                "Super repairs the plan, and attempt two succeeds. If a repair fails again, it escalates to "
                "Nemotron Ultra. The first plan passed only twelve of twenty stress tests. The repaired one: all twenty.",
    "f_beyond": "It goes beyond pick and place: opening drawers, like NVIDIA's Isaac Lab tasks, and knocking "
                "things over. And every scene exports as a MuJoCo file with its success checks, ready to test "
                "your own robot policy.",
    "g_rigor": "Each step shows which model ran it, how long it took, and what it cost. A typical task costs "
               "about a cent. On our evaluation suite of eleven tasks, every one passes.",
    "h_close": "Spec-to-Sim. Describe it. Simulate it. Trust the result.",
}


def main(only=None, extra=None):
    OUT.mkdir(exist_ok=True)
    k = Kokoro(str(HERE / "models/kokoro-v1.0.onnx"), str(HERE / "models/voices-v1.0.bin"))
    g2p = en.G2P(trf=False, british=False, fallback=None)
    segs = dict(SEGMENTS)
    if extra:
        segs.update(extra)
    durations = {}
    for name, text in segs.items():
        if only and name not in only:
            continue
        for word, rep in SAY.items():
            text = text.replace(word, rep)
        phonemes, _ = g2p(text)
        if "❓" in phonemes:
            print(f"  WARNING unknown word in {name}: {phonemes}")
        audio, sr = k.create(phonemes, voice=VOICE, speed=SPEED, is_phonemes=True)
        pad = np.zeros(int(sr * 0.25), dtype=audio.dtype)
        audio = np.concatenate([pad, audio, pad])
        sf.write(OUT / f"{name}.wav", audio, sr)
        durations[name] = round(len(audio) / sr, 2)
        print(f"{name}: {durations[name]} s")
    old = json.loads((OUT / "durations.json").read_text()) if (OUT / "durations.json").exists() else {}
    old.update(durations)
    (OUT / "durations.json").write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    extra = json.loads(sys.argv[1]) if len(sys.argv) > 1 else None
    main(only=list(extra) if extra else None, extra=extra)
