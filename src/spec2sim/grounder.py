"""Step 2 - Grounder (Tavily + Nemotron Nano): real-world sizes and masses, with sources.

Runs the Planner's search queries on Tavily (cached on disk by query hash to protect credits,
max 3 per run), then Nemotron Nano extracts one size + mass per object, citing the source URL.
If the web gives nothing usable, Nano falls back to a stated estimate (source_url = null).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from langsmith import traceable
from tavily import TavilyClient

from spec2sim import config
from spec2sim.llm import CallStat, chat_json
from spec2sim.schemas import GroundedFacts, Source, TaskSpec

CACHE = Path(__file__).resolve().parents[2] / ".cache/tavily"
MAX_SEARCHES = 3

SYSTEM = """You extract physical facts for simulation objects from web search results.
For EACH object listed, return its real full size in METERS as (x, y, z) - for a cylinder or
sphere use (diameter, diameter, height) - and its mass in KILOGRAMS.
Prefer numbers stated in the search results and set source_url to the page they came from.
Convert units carefully (mm -> m divide by 1000, g -> kg divide by 1000).
Assume each object is in its normal everyday state (a soda can is full, a mug is empty).
If the results do not cover an object, give a typical real-world value, set source_url to null,
and say "estimate" in the note. Keep notes under 15 words. Return JSON only."""


@traceable(name="tavily_search", run_type="retriever")
def search(query: str) -> tuple[dict, bool]:
    """Tavily search with disk cache. Returns (response, was_cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16] + ".json")
    if f.exists():
        return json.loads(f.read_text()), True
    res = TavilyClient(config.TAVILY_API_KEY).search(query, max_results=4, include_answer="basic")
    f.write_text(json.dumps(res))
    return res, False


@traceable(name="grounder")
def ground(task: TaskSpec) -> tuple[GroundedFacts, list[Source], dict, list[CallStat]]:
    """Returns facts, the source list for the UI, Tavily stats, and LLM call stats."""
    blocks, sources, cached, t0 = [], [], 0, time.time()
    for q in task.search_queries[:MAX_SEARCHES]:
        try:
            res, hit = search(q)
        except Exception as e:  # noqa: BLE001 - network/quota problems must not kill the run
            blocks.append(f"## Search: {q}\n(search failed: {e})")
            continue
        cached += hit
        lines = [f"## Search: {q}"]
        if res.get("answer"):
            lines.append(f"Summary: {res['answer']}")
        for r in res.get("results", []):
            sources.append(Source(title=r.get("title", r["url"])[:90], url=r["url"]))
            lines.append(f"- [{r.get('title', '')}]({r['url']}): {r.get('content', '')[:600]}")
        blocks.append("\n".join(lines))
    tavily_stats = {"queries": task.search_queries[:MAX_SEARCHES], "cached": cached,
                    "latency_s": round(time.time() - t0, 2)}

    objects = "\n".join(f"- {o.name}: {o.description} ({o.shape}, {o.color})" for o in task.objects)
    user = f"Objects:\n{objects}\n\nSearch results:\n" + "\n\n".join(blocks)
    facts, stats = chat_json("grounder", config.NANO, SYSTEM, user, GroundedFacts, max_tokens=3000,
                             temperature=0.1)
    # Keep the sources list short and unique for the UI.
    seen, uniq = set(), []
    for s in sources:
        if s.url not in seen:
            seen.add(s.url)
            uniq.append(s)
    return facts, uniq[:8], tavily_stats, stats
