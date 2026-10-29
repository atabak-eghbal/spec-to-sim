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
    "MuJoCo": "[MuJoCo](/mjˈuʤˌOkO/)", "datasheet": "[datasheet](/dˈATəʃˌit/)", "Tavily": "[Tavily](/tˈævᵻli/)", "Rubik's": "[Rubik's](/ɹˈubɪks/)",
    "GR00T": "[GR00T](/ɡɹˈut/)", "Franka": "[Franka](/fɹˈæŋkə/)", "Serverless": "[Serverless](/sˈɜɹvəɹləs/)",
    "verified": "[verified](/vˈɛɹəfˌId/)", "NumPy": "[NumPy](/nˈʌmpˌI/)", "LeRobot": "[LeRobot](/lə ɹˈObˌɑt/)", "LangSmith": "[LangSmith](/lˈæŋ smˈɪθ/)",
}


VOICE, SPEED = "af_heart", 1.06
HERE = Path(__file__).parent
OUT = HERE / "audio"

SEGMENTS = {
    "a_hook": "One sentence in. A robot simulation out: checked by code, repaired when it fails, and turned into a "
              "test that any robot policy can take.",
    "b_problem": "Building one robot test scene by hand takes days: measuring objects, writing scene files, "
                 "writing success checks. Spec-to-Sim does it from plain English, with NVIDIA Nemotron models "
                 "on Nebius Token Factory.",
    "c_live": "Here's a live run. Super turns the request into goals code can check, Tavily grounds the real sizes, "
              "and Super writes the plan as skills. Attempt one misses: code measures the red block level with the blue "
              "one, not in front. Super V L explains why. Super writes three repairs, and the simulator is the judge: it "
              "tests each one and runs the sturdiest. Attempt two passes, and fifteen of sixteen perturbed re-runs agree.",
    "d_tavily": "Tavily is the reality check. Every size is cross-checked across sources, with photos and links, and "
                "friction and bounciness come from the web too. Even the robot's limits come from Franka's datasheet. "
                "When the web says a grapefruit is eighteen centimetres but weighs two hundred and forty-six grams, the "
                "weight wins: it's really eight. Still too wide for the gripper, so Spec-to-Sim stops in seconds and says why.",
    "e_repair": "Another task, side by side. On the left, the first plan let the soda can go too high, and it fell "
                "over. On the right, the repaired plan sets it down upright. The vision model's diagnosis is there for "
                "you to read, and the three candidates it led to, checked by physics.",
    "f_robust": "Robust, or lucky? Every success is re-run sixteen times: objects moved, heavier, more slippery, and the "
                "robot's view half a centimetre off. Good plans pass almost every time; a fragile one passes once. "
                "And you can try to break it yourself: drag an object, or change its size or weight, and re-run.",
    "g_beyond": "It goes beyond pick and place: gripping a drawer handle to open and close it, and knocking "
                "things over.",
    "h_data": "One click turns a solved task into training data: twenty demonstrations in the LeRobot format. We trained "
              "a small policy on them and tested it on the same sixteen variations. It matches the plan on a simple "
              "stack, but drops to four of sixteen on the tennis ball. That's what a test is for.",
    "i_fit": "And every export is that test: the scene, the same random variations, and the same goal checks, in a "
             "small package that needs only MuJoCo and NumPy. Bring your own policy, and get its score.",
    "j_rigor": "On fifteen held-out prompts, forty-two of forty-five runs succeed, typically for under a cent. When the "
               "data showed the biggest model wasn't helping, we cut it: three cheap repairs, checked by the simulator, "
               "do as well. And reasoning pays for itself: switched off, success fell from eleven of eleven to six. Every "
               "call is traced in LangSmith.",
    "k_close": "Spec-to-Sim. Describe it. Simulate it. Test it.",
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
