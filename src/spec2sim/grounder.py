"""Step 2 - Grounder (Tavily + Nemotron Nano): real-world sizes and masses, with sources.

Runs the Planner's search queries on Tavily (cached on disk by query hash to protect credits,
max 3 per run), then Nemotron Nano extracts one size + mass per object, citing the source URL.
Nano copies numbers in the units the web uses (mm, g) plus the quote they came from; our code
converts to SI and sanity-checks mass against volume. If the web gives nothing usable, Nano falls
back to a stated estimate (source_url = null).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

from langsmith import traceable
from pydantic import BaseModel
from tavily import TavilyClient

from spec2sim import config
from spec2sim.llm import CallStat, chat_json
from spec2sim.schemas import GroundedFacts, ObjectFact, Source, TaskSpec

CACHE = Path(__file__).resolve().parents[2] / ".cache/tavily"
MAX_SEARCHES = 3

SYSTEM = """You extract physical facts for simulation objects from web search results.
For EACH object listed, fill in, in this order:
- quote: the exact phrase from the search results that states its size and/or weight (copy it),
  or "" if the results do not mention this object's numbers
- source_url: the URL of the result that quote came from, or null
- size_mm: full size in MILLIMETERS as [x, y, z]; for a cylinder or sphere [diameter, diameter, height]
- mass_g: mass in GRAMS
- note: under 12 words, e.g. "from quote" or "estimate: typical wooden block"
Copy numbers from the quote when it has them; convert inches/ounces to mm/g if needed.
Assume each object is in its normal everyday state (a drink can is full, a mug is empty).
If there is no quote, give a typical real-world value and say "estimate" in the note.
Return JSON only."""


class _Extracted(BaseModel):
    name: str
    quote: str
    source_url: str | None
    size_mm: tuple[float, float, float]
    mass_g: float
    note: str


class _Extraction(BaseModel):
    facts: list[_Extracted]


# Density bounds (kg/m^3) for the sanity check; outside these the mass is replaced by an estimate.
_MIN_DENSITY, _MAX_DENSITY, _DEFAULT_DENSITY = 150, 9000, 900


def _to_fact(e: _Extracted, shape: str) -> ObjectFact:
    """Convert mm/g to SI and sanity-check the mass against the object's volume."""
    size = tuple(v / 1000 for v in e.size_mm)
    mass = e.mass_g / 1000
    vol = size[0] * size[1] * size[2]
    if shape == "cylinder":
        vol *= math.pi / 4
    elif shape == "sphere":
        vol *= math.pi / 6
    note = e.note
    density = mass / vol if vol > 0 else 0
    if vol > 0 and not _MIN_DENSITY <= density <= _MAX_DENSITY:
        mass = round(vol * _DEFAULT_DENSITY, 4)
        note = f"mass corrected: {e.mass_g:g} g implies {density:.0f} kg/m3"
    return ObjectFact(name=e.name, size=size, mass=mass,
                      source_url=e.source_url if e.quote.strip() else None, note=note)


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
    ex, stats = chat_json("grounder", config.NANO, SYSTEM, user, _Extraction, max_tokens=3000,
                          temperature=0.1, thinking=False)
    shapes = {o.name: o.shape for o in task.objects}
    facts = GroundedFacts(facts=[_to_fact(e, shapes.get(e.name, "box")) for e in ex.facts])
    # Keep the sources list short and unique for the UI.
    seen, uniq = set(), []
    for s in sources:
        if s.url not in seen:
            seen.add(s.url)
            uniq.append(s)
    return facts, uniq[:8], tavily_stats, stats
