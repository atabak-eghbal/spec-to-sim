# Models — Nemotron family on Token Factory

Prices are $ per 1M tokens (input / output), from third-party listings around Oct 20, 2026.
🔎 Confirm IDs and prices with `GET /v1/models` and the Token Factory catalog before relying on them.

| Model (expected ID) | In / Out | Context | Our use |
|---|---|---|---|
| Nemotron 3 Nano 30B-A3B (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`) | $0.06 / $0.24 | 32K–262K | Extract numbers from Tavily results |
| Nemotron 3.5 Lightning (`nvidia/Nemotron-3_5-Lightning`) | $0.06 / $0.24 | 1M | Alternative fast text model |
| Nemotron 3 Super 120B-A12B (`nvidia/nemotron-3-super-120b-a12b`) | $0.30 / $0.90 | 262K | Planner, Builder, early repairs |
| Nemotron 3.5 Super VL (`nvidia/Nemotron-3.5-Super-VL`) | $0.30 / $0.90 | 1M | Critic: watches frames |
| Nemotron 3 Ultra 550B-A55B (`nvidia/Nemotron-3-Ultra-550b-a55b`) | $1.00 / $3.00 | 1M | Fixer, only on escalation |

Sources:
- https://benchlm.ai/hosts/nebius (Token Factory price table)
- https://modelcompare.dev/models/nvidia/nemotron-3-nano-omni
- https://openrouter.ai/nvidia/nemotron-3.5-super-vl/providers (Super VL description)

## Notes on Nemotron 3.5 Super VL
- Released Oct 15, 2026. 120B total / 12B active MoE; text + image + document + video input; 1M context.
- Reasoning can be switched off/on — 🔎 check the Token Factory parameter for this; turn it off for fast critiques.
- 🔎 Does Token Factory accept video input? Default plan: send 6–8 sampled frames as base64 images + metrics JSON.

## Cost estimate per run
~4 calls × ~4K in / 1K out on Super/VL ≈ $0.01 per iteration; Ultra escalation ≈ $0.02.
Full dev + demo + 20-prompt eval should stay well under $15.

## Verified 2026-10-28 (scripts/list_models.py, scripts/smoke_apis.py)
- ✅ All 4 IDs above exist exactly as written (+ `nvidia/Nemotron-3_5-Lightning`). 30 models total on Token Factory.
- ✅ Super VL accepts base64 PNG via `image_url` data URI.
- Super returns `reasoning_content` alongside `content`; Nano returned none on a trivial prompt.
- ⚠️ Super VL latency on one 848x480 frame: 35 s, 34 s, 10 s. `chat_template_kwargs={"enable_thinking": False}` is accepted
  and removes reasoning (25 output tokens) but did NOT cut latency → server-side image/queue time. Measure with
  6–8 frames; consider downscaling frames or a contact-sheet grid image to send 1 image instead of 8.

📌 Escalation policy: Super handles attempts 1–2, Ultra only from attempt 3. Show this in the UI.
