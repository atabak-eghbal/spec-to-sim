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
    "GR00T": "[GR00T](/ɡɹˈut/)", "Franka": "[Franka](/fɹˈæŋkə/)", "Serverless": "[Serverless](/sˈɜɹvəɹləs/)",
    "verified": "[verified](/vˈɛɹəfˌId/)", "LeRobot": "[LeRobot](/lə ɹˈObˌɑt/)", "LangSmith": "[LangSmith](/lˈæŋ smˈɪθ/)",
}


VOICE, SPEED = "af_heart", 1.06
HERE = Path(__file__).parent
OUT = HERE / "audio"

SEGMENTS = {
    "a_hook": "One sentence in. A robot simulation out: checked by code, repaired by AI when it fails, "
              "and turned into training data.",
    "b_problem": "Building one robot test scene by hand takes days: measuring objects, writing scene files, "
                 "writing success checks. Spec-to-Sim does it from plain English, with NVIDIA Nemotron models "
                 "on Nebius Token Factory.",
    "c_live": "Here's a live run. Nemotron Super turns the request into goals that code can check. Tavily searches "
              "the web for each object's real size and weight, and Nemotron Nano reads off the numbers, with their "
              "sources. Super places the objects and plans the motion, as data, never code. MuJoCo runs it, and code, "
              "not a model, checks every goal.",
    "d_tavily": "Those real facts change outcomes. Here, Tavily found that a large apple is eight point three "
                "centimetres wide, more than the Panda's gripper can open. So Spec-to-Sim stops in seconds and says "
                "why, instead of paying for repairs that can't work. And web numbers that make no sense, like a "
                "two-millimetre lime, are caught and replaced.",
    "e_repair": "When a plan fails, it fixes itself. On the left, the ball was released a centimetre and a half too "
                "high, and rolled off. Code measures the miss. Nemotron Super V L looks at the frames and explains "
                "why, and Super rewrites the plan. On the right, attempt two. If a repair fails again, it escalates "
                "to Nemotron Ultra.",
    "f_robust": "But is it robust, or lucky? Every success is re-run sixteen times, with objects nudged, heavier, or "
                "more slippery. The first plan survived twelve of twenty. The repaired one, all twenty. And you can "
                "try to break it yourself: drag an object, re-run the same plan, no AI cost.",
    "g_beyond": "It goes beyond pick and place: opening drawers, like Isaac Lab's Franka drawer task, and knocking "
                "things over.",
    "h_data": "Then one click turns a solved task into training data: twenty successful demonstrations with varied "
              "starts, failures dropped, saved in the LeRobot format that NVIDIA GR00T trains on. The same script is "
              "packaged to run as a Nebius Serverless Job.",
    "i_fit": "Where it fits: Isaac Lab Arena turns language into a valid environment spec. Mimic and the GR00T "
             "blueprint multiply human demonstrations. Spec-to-Sim goes from one sentence to a verified, running "
             "scene and its demonstrations, no human demo needed, ready to feed those pipelines.",
    "j_rigor": "Four Nemotron models, each sized to its job. About a cent per task. Eleven of eleven evaluation tasks "
               "pass, and every call is traced in LangSmith.",
    "k_close": "Spec-to-Sim. Describe it. Simulate it. Trust it.",
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
