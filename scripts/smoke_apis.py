"""Smoke test: one tiny live call per external service, all traced in LangSmith.

Checks: Nano + Super text, Super VL on a rendered frame (image input), one Tavily search.
Ultra is skipped on purpose (cost); its ID is confirmed by scripts/list_models.py.

Run: uv run python scripts/smoke_apis.py   (needs runs/smoke/panda_smoke.png from smoke_render.py)
"""
import base64
import time
from pathlib import Path

from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from tavily import TavilyClient

from spec2sim import config

ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT / "runs/smoke/panda_smoke.png"
client = wrap_openai(OpenAI(base_url=config.NEBIUS_BASE_URL, api_key=config.NEBIUS_API_KEY))


def call(model: config.ModelInfo, content, max_tokens: int = 200) -> None:
    t0 = time.time()
    r = client.chat.completions.create(
        model=model.id, messages=[{"role": "user", "content": content}], max_tokens=max_tokens
    )
    u, msg = r.usage, r.choices[0].message
    text = (msg.content or "").strip().replace("\n", " ")
    reasoning = getattr(msg, "reasoning_content", None) or ""
    print(f"[{model.label}] {time.time() - t0:.1f}s  in={u.prompt_tokens} out={u.completion_tokens} "
          f"${model.cost(u.prompt_tokens, u.completion_tokens):.5f}  finish={r.choices[0].finish_reason}"
          f"  reasoning_chars={len(reasoning)}")
    print(f"   -> {text[:300]!r}")


@traceable(name="smoke_apis")
def main() -> None:
    call(config.NANO, "Reply with only the JSON {\"ok\": true}.")
    call(config.SUPER, "Reply with only the JSON {\"ok\": true}.")

    img = base64.b64encode(FRAME.read_bytes()).decode()
    call(config.VL, [
        {"type": "text", "text": "In one sentence: what robot is shown, and what colored objects "
                                 "are on the floor?"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
    ], max_tokens=400)

    t0 = time.time()
    res = TavilyClient(config.TAVILY_API_KEY).search("Franka Emika Panda payload and reach",
                                                     max_results=3)
    print(f"[Tavily] {time.time() - t0:.1f}s  {len(res['results'])} results")
    for x in res["results"]:
        print("   -", x["url"])


if __name__ == "__main__":
    main()
