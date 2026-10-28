"""One Token Factory client for every pipeline step: structured output, tracing, cost, budget.

- OpenAI-compatible client pointed at Nebius Token Factory, wrapped by LangSmith (`wrap_openai`),
  so every call lands in the trace tree.
- `chat_json` asks for JSON constrained to a Pydantic schema (`response_format=json_schema`),
  validates it, and retries once with the validation error if needed.
- Every call returns a `CallStat` (model, latency, tokens, $) that the UI shows per step, and is
  appended to `.cache/spend.jsonl`; calls stop once dev spend passes DEV_SPEND_LIMIT_USD.
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import TypeVar

from langsmith.wrappers import wrap_openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from spec2sim import config

T = TypeVar("T", bound=BaseModel)
ROOT = Path(__file__).resolve().parents[2]
SPEND_LOG = ROOT / ".cache/spend.jsonl"

_client = wrap_openai(
    OpenAI(base_url=config.NEBIUS_BASE_URL, api_key=config.NEBIUS_API_KEY, timeout=180, max_retries=2)
)


class BudgetError(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


class CallStat(BaseModel):
    step: str
    model: str
    label: str
    latency_s: float
    tokens_in: int
    tokens_out: int
    cost_usd: float


def total_spend() -> float:
    if not SPEND_LOG.exists():
        return 0.0
    return sum(json.loads(line)["cost_usd"] for line in SPEND_LOG.read_text().splitlines() if line)


def _image_part(path: str) -> dict:
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def chat_json(
    step: str,
    model: config.ModelInfo,
    system: str,
    user: str,
    out: type[T],
    images: list[str] | None = None,
    max_tokens: int = 8000,
    temperature: float = 0.3,
    thinking: bool = True,
) -> tuple[T, list[CallStat]]:
    """Call `model` and parse its reply as `out`. Returns the object and one CallStat per attempt.

    thinking=False turns off Nemotron's reasoning trace (chat_template_kwargs.enable_thinking):
    on Super VL this cuts a critique from ~25 s (and sometimes an exhausted token budget) to ~2 s.
    """
    spent = total_spend()
    if spent > config.DEV_SPEND_LIMIT_USD:
        raise BudgetError(f"spend ${spent:.2f} exceeds limit ${config.DEV_SPEND_LIMIT_USD:.2f}")

    content: str | list = user
    if images:
        content = [{"type": "text", "text": user}, *(_image_part(p) for p in images)]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
    fmt = {"type": "json_schema",
           "json_schema": {"name": out.__name__, "schema": out.model_json_schema(), "strict": True}}

    stats: list[CallStat] = []
    for attempt in range(2):
        t0 = time.time()
        r = _client.chat.completions.create(
            model=model.id, messages=messages, response_format=fmt,
            max_tokens=max_tokens, temperature=temperature,
            extra_body=None if thinking else {"chat_template_kwargs": {"enable_thinking": False}},
            langsmith_extra={"name": f"{step} ({model.label})", "metadata": {"step": step}},
        )
        u = r.usage
        stat = CallStat(step=step, model=model.id, label=model.label,
                        latency_s=round(time.time() - t0, 2), tokens_in=u.prompt_tokens,
                        tokens_out=u.completion_tokens,
                        cost_usd=round(model.cost(u.prompt_tokens, u.completion_tokens), 6))
        stats.append(stat)
        SPEND_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SPEND_LOG.open("a") as f:
            f.write(stat.model_dump_json() + "\n")

        text = r.choices[0].message.content or ""
        if not text.strip() and r.choices[0].finish_reason == "length":
            # Reasoning ate the whole budget. Retry once with reasoning off rather than fail the run.
            if attempt == 1 or not thinking:
                raise LLMError(f"{step}: {model.label} used all {max_tokens} tokens reasoning, no answer")
            thinking = False
            continue
        try:
            return out.model_validate_json(text), stats
        except ValidationError as e:
            if attempt == 1:
                raise LLMError(f"{step}: invalid {out.__name__} after retry: {e}") from e
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"That JSON failed validation:\n{e}\nReturn corrected JSON only."}]
    raise AssertionError("unreachable")
