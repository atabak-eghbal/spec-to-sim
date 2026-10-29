"""Step 2 - Grounder (Tavily + Nemotron Nano): real-world facts that change the simulation.

1. Object facts: the Planner's searches (max 3, cached on disk) return snippets, a Tavily answer
   and product images. Nano extracts each object's size and mass with the quote it came from,
   plus up to 3 independent pieces of evidence, so code can count how many sources agree.
   Code converts units, rejects sizes far from the Planner's own estimate, checks density.
2. Material physics: friction and bounciness (coefficient of restitution) per material, searched
   once and cached for every later run, then applied to MuJoCo's contact settings.
3. Robot limits: the Franka's gripper opening, payload and reach from its spec sheet (cached),
   so "physically infeasible" cites a source on both sides.
Every number keeps its source URL; values outside physical bounds fall back to labelled estimates.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path

from langsmith import traceable
from pydantic import BaseModel
from tavily import TavilyClient

from spec2sim import config
from spec2sim.llm import CallStat, chat_json
from spec2sim.schemas import GroundedFacts, MaterialFact, ObjectFact, RobotFact, Source, TaskSpec
from spec2sim.sim.scene import BIN_MAX_SIZE

BIN_TARGET_HEIGHT = 0.12  # walls low enough for the arm to reach over comfortably
BIN_MIN_WIDTH = 0.15  # room for a few objects plus the fingers

CACHE = Path(__file__).resolve().parents[2] / ".cache/tavily"
MAX_SEARCHES = 3

SYSTEM = """You extract physical facts for simulation objects from web search results.
For EACH object listed, fill in, in this order:
- quote: the exact phrase from the search results that states its size and/or weight (copy it),
  or "" if the results do not mention this object's numbers
- source_url: the URL of the result that quote came from, or null
- size_mm: full size in MILLIMETERS as [x, y, z]; for a cylinder or sphere [diameter, diameter, height];
  for a bin (open container) or a drawer unit its outer [depth, width, height]
- mass_g: mass in GRAMS
- note: under 12 words, e.g. "from quote" or "estimate: typical wooden block"
- evidence: up to 3 INDEPENDENT sizes for this object from DIFFERENT results (or the search
  Summary), each {source: URL or "summary", size_mm: [x,y,z] or null, mass_g: number or null};
  [] if only one source mentions it
- image: index of the IMAGES entry that shows this exact kind of object, or -1
Only use a quote about the SAME kind of thing: a result about an Apple iPhone or AirPods is not
about an apple fruit; a full-size curbside cart is not a desktop bin. If no result matches, estimate.
Copy numbers from the quote when it has them; convert inches/ounces to mm/g if needed.
Assume each object is in its normal everyday state (a drink can is full, a mug is empty).
If a source lists several grades or sizes, use the typical/median one, not the largest.
If there is no quote, give a typical real-world value and say "estimate" in the note.
Return JSON only."""


class _Evidence(BaseModel):
    source: str
    size_mm: tuple[float, float, float] | None
    mass_g: float | None


class _Extracted(BaseModel):
    name: str
    quote: str
    source_url: str | None
    size_mm: tuple[float, float, float]
    mass_g: float
    note: str
    evidence: list[_Evidence] = []
    image: int = -1


class _Extraction(BaseModel):
    facts: list[_Extracted]


# ---------- material physics + robot limits (searched once, cached for every later run) ----------
PHYS_CACHE = Path(__file__).resolve().parents[2] / ".cache/physics"
FRICTION_QUERY = {"rubber": "rubber on wood coefficient of friction table", "fruit": "apple fruit friction coefficient on wood measured",
                  "fabric": "felt fabric coefficient of friction on wood"}
RESTITUTION_QUERY = {"rubber": "tennis ball rubber ball coefficient of restitution value",
                     "fruit": "apple fruit coefficient of restitution drop impact measured"}
ROBOT_QUERIES = {"gripper": "Franka Emika Panda Hand gripper technical data stroke maximum opening",
                 "arm": "Franka Emika Panda robot arm technical data payload maximum reach"}
# Typical values (engineering tables), used only when the web gives nothing usable or verifiable.
TYPICAL = {"rubber": (0.9, 0.75), "wood": (0.4, 0.5), "plastic": (0.35, 0.6), "metal": (0.3, 0.55),
           "glass": (0.4, 0.6), "ceramic": (0.45, 0.5), "cardboard": (0.5, 0.2), "paper": (0.45, 0.15),
           "fabric": (0.6, 0.2), "foam": (0.8, 0.1), "fruit": (0.5, 0.35)}
# Plausible (friction, restitution) ranges per material; web values outside are rejected.
PLAUSIBLE = {"rubber": ((0.5, 1.2), (0.5, 0.9)), "wood": ((0.2, 0.7), (0.25, 0.7)), "plastic": ((0.15, 0.6), (0.3, 0.8)),
             "metal": ((0.15, 0.8), (0.2, 0.8)), "glass": ((0.2, 0.9), (0.4, 0.9)), "ceramic": ((0.3, 0.8), (0.3, 0.7)),
             "cardboard": ((0.3, 0.8), (0.05, 0.4)), "paper": ((0.2, 0.8), (0.05, 0.4)), "fabric": ((0.3, 1.0), (0.05, 0.5)),
             "foam": ((0.4, 1.2), (0.02, 0.3)), "fruit": ((0.2, 0.9), (0.1, 0.6))}
ROBOT_TYPICAL = (80.0, 3.0, 855.0)  # Panda Hand stroke 80 mm, payload 3 kg, reach 855 mm
MAX_EXTRACTS = 2  # Tavily /extract calls per run, only when search snippets carry no size
MAX_UNCACHED = 5  # Tavily searches per run that are not served from the cache (CLAUDE.md budget)

PHYS_SYSTEM = """You extract physical constants from web search results, for a robot simulation.
Blocks are labelled MATERIAL <m> FRICTION, MATERIAL <m> RESTITUTION, ROBOT GRIPPER, ROBOT ARM.
- friction: sliding (kinetic) friction coefficient of that material on a wooden or similar table.
- restitution: coefficient of restitution of an object of that material dropped on a hard surface
  (0 = no bounce, 1 = perfect bounce). It is NOT a friction coefficient.
- robot: the Franka Emika Panda Hand's maximum gripper opening (stroke) in mm; the arm's payload in kg
  and maximum reach in mm.
Prefer a value measured for this material sliding on DRY wood (or a wooden table); a value for wet,
lubricated or other surfaces, or from a worked example, is not acceptable: give 0 instead.
FULL PAGE lines are passages read from the page itself; quote from them as well as from the snippets.
For every number, copy the EXACT text it appears in, character for character, as its quote (at least
6 words, including the material and the surface it slides on), and give that result's URL. If a block does not state the number, set the value to 0 and
the quote to "". Never write a quote that is not in the results. Return JSON only."""


class _MaterialEx(BaseModel):
    material: str
    friction: float
    friction_quote: str
    friction_source: str | None
    restitution: float
    restitution_quote: str
    restitution_source: str | None


class _RobotEx(BaseModel):
    gripper_opening_mm: float
    gripper_quote: str
    gripper_source: str | None
    payload_kg: float
    reach_mm: float
    arm_quote: str
    arm_source: str | None


class _PhysicsEx(BaseModel):
    materials: list[_MaterialEx]
    robot: _RobotEx | None


def _norm(t: str) -> str:
    t = t.lower().replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    t = re.sub(r"[*_`#>|\[\]()]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def verified(quote: str, corpus: str) -> bool:
    """True if the quote (or a substantial piece of it) appears in the fetched text and its numbers
    do too. Guards against a model inventing a citation."""
    q = _norm(quote)
    if len(q) < 12:
        return False
    nums = re.findall(r"\d+(?:\.\d+)?", q)
    if nums and not all(n in corpus for n in nums):
        return False
    pieces = [p.strip() for p in re.split(r"\.\.\.|…|;| - ", q) if len(p.strip()) >= 12]
    return any(p in corpus for p in pieces) or q in corpus


def _volume(size: tuple[float, float, float], shape: str) -> float:
    v = size[0] * size[1] * size[2]
    return v * (math.pi / 4 if shape == "cylinder" else math.pi / 6 if shape == "sphere" else 1)


# Density bounds (kg/m^3) for the sanity check; outside these the mass is replaced by an estimate.
_MIN_DENSITY, _MAX_DENSITY, _DEFAULT_DENSITY = 150, 9000, 900


PRIOR_TOLERANCE = 2.0  # web size may differ from the Planner's own estimate by at most this factor


# Typical bulk density per material (kg/m^3) for the size-vs-mass consistency check; hollow things
# (tennis balls, cans of air, cardboard boxes) are low.
DENSITY = {"fruit": 850, "wood": 600, "plastic": 700, "metal": 1000, "rubber": 400, "glass": 2500,
           "ceramic": 2000, "cardboard": 200, "paper": 700, "fabric": 300, "foam": 50}


def _to_fact(e: _Extracted, shape: str, prior_cm: tuple[float, float, float] | None = None,
             material: str | None = None) -> ObjectFact:
    """Convert mm/g to SI and sanity-check size against the Planner's prior and mass against volume."""
    size = tuple(v / 1000 for v in e.size_mm)
    mass = e.mass_g / 1000
    if prior_cm and all(v > 0 for v in prior_cm) and min(size) > 0:
        # Web pages list dimensions in any order ("12.3 x 6.6 x 6.6 cm"). Orient them like the Planner's
        # estimate: the web's smallest value goes where the estimate is smallest, and so on. A soda
        # can once came back as 12.3 x 12.3 x 6.6 (lying down) and was refused as too wide to grip.
        order = sorted(range(3), key=lambda i: prior_cm[i])
        web = sorted(size)
        oriented = [0.0, 0.0, 0.0]
        for rank, i in enumerate(order):
            oriented[i] = web[rank]
        size = tuple(oriented)
    if shape in ("cylinder", "sphere"):
        # Nano sometimes scrambles the order, e.g. a can as [122, 66, 122]. A round object's
        # diameter is its smaller horizontal value (spheres: all three equal the diameter).
        d = min(size[0], size[1])
        size = (d, d, d) if shape == "sphere" else (d, d, size[2])
    if min(size) <= 0:  # the web gave an incomplete size (a 0 or missing dimension)
        fallback = tuple(v / 100 for v in prior_cm) if prior_cm and all(v > 0 for v in prior_cm) else (0.05, 0.05, 0.05)
        e = e.model_copy(update={"quote": "", "note": f"web size {list(e.size_mm)} mm incomplete; used the estimate"})
        size = fallback
    if prior_cm and all(v > 0 for v in prior_cm):
        prior = tuple(v / 100 for v in prior_cm)
        ratio = max(max(a / b, b / a) for a, b in zip(sorted(size), sorted(prior), strict=True))
        if ratio > PRIOR_TOLERANCE:
            web_cm = [round(v * 100, 1) for v in size]
            size = prior
            mass = mass if _MIN_DENSITY <= mass / max(_volume(size, shape), 1e-9) <= _MAX_DENSITY else 0
            e = e.model_copy(update={"quote": "", "note": f"web size {web_cm} cm rejected (expected ~{list(prior_cm)} cm)"})
    vol = _volume(size, shape)
    note = e.note
    if shape in ("bin", "drawer"):  # hollow and fixed in place: mass does not matter, volume is mostly air
        # Real storage bins are floor-sized. Shrink the footprint to fit the workspace (keeping its
        # proportions) and cap the walls at a height the arm can reach over.
        k = min(1.0, BIN_MAX_SIZE / max(size[:2]))
        scaled = (round(size[0] * k, 4), round(size[1] * k, 4), round(min(size[2] * k, BIN_TARGET_HEIGHT), 4))
        if min(scaled[:2]) < BIN_MIN_WIDTH:  # a pencil-cup "bin" can't take several objects
            up = BIN_MIN_WIDTH / min(scaled[:2])
            scaled = (round(min(scaled[0] * up, BIN_MAX_SIZE), 4), round(min(scaled[1] * up, BIN_MAX_SIZE), 4), scaled[2])
        if scaled != tuple(round(v, 4) for v in size):
            note = f"resized to tabletop size from real {[round(v * 100) for v in size]} cm"
            size = scaled
        return ObjectFact(name=e.name, size=size, mass=min(max(mass, 0.05), 2.0),
                          source_url=e.source_url if e.quote.strip() else None, note=note)
    density = mass / vol if vol > 0 else 0
    if vol > 0 and not _MIN_DENSITY <= density <= _MAX_DENSITY:
        # Size and mass disagree. Two out of three: whichever of them agrees with the Planner's own
        # size estimate wins (e.g. a "19 cm" grapefruit weighing 246 g is really ~8 cm).
        k = (mass / (_DEFAULT_DENSITY if material is None else DENSITY.get(material, _DEFAULT_DENSITY)) / vol) ** (1 / 3)
        from_mass = tuple(round(v * k, 4) for v in size)
        if prior_cm and mass > 0 and not note.startswith("web size"):
            prior = sorted(v / 100 for v in prior_cm)
            off = lambda sz: max(max(a / b, b / a) for a, b in zip(sorted(sz), prior, strict=True))
            if off(from_mass) < off(size) and off(from_mass) <= 1.6:
                web_cm = [round(v * 100, 1) for v in size]
                return ObjectFact(name=e.name, size=from_mass, mass=mass, source_url=e.source_url if e.quote.strip() else None,
                                  note=f"size corrected: web {web_cm} cm doesn't fit its {e.mass_g:g} g weight; "
                                       f"used the size the weight implies")
        mass = round(vol * _DEFAULT_DENSITY, 4)
        fix = (f"web mass {e.mass_g:g} g is impossible for this size ({density:.0f} kg/m3); "
               f"mass estimated from the size instead")
        note = f"{note}; {fix}" if note.startswith("web size") else fix
    return ObjectFact(name=e.name, size=size, mass=mass,
                      source_url=e.source_url if e.quote.strip() else None, note=note)


def _fit_drawers(task: TaskSpec, facts: GroundedFacts) -> None:
    """A drawer must be tall enough for what goes in it (with room to slide shut) and big enough
    to reach into; real drawer units vary from flat organizers to furniture."""
    by = {f.name: f for f in facts.facts}
    for o in task.objects:
        f = by.get(o.name)
        if o.shape != "drawer" or f is None:
            continue
        contents = [by[g.obj].size[2] for g in task.goals if g.type == "inside" and g.target == o.name and g.obj in by]
        need_h = max([0.08] + [h + 0.045 for h in contents])
        sx, sy, sz = f.size
        new = (min(max(sx, 0.18), 0.24), min(max(sy, 0.18), BIN_MAX_SIZE), min(max(sz, need_h), 0.15))
        if new != tuple(f.size):
            f.size = new
            f.note = (f.note + "; " if f.note else "") + "sized to fit its contents (tabletop drawer unit)"


@traceable(name="tavily_search", run_type="retriever")
def search(query: str, images: bool = True, advanced: bool = False) -> tuple[dict, bool]:
    """Tavily search with disk cache. Returns (response, was_cached). advanced=True uses Tavily's
    advanced depth with 3 relevant chunks per page (for the shared physics facts: worth 2 credits)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = ("v2:" if images else "") + ("adv:" if advanced else "") + query.strip().lower()  # v2: with product images
    f = CACHE / (hashlib.sha256(key.encode()).hexdigest()[:16] + ".json")
    if f.exists():
        return json.loads(f.read_text()), True
    kw = {"search_depth": "advanced", "chunks_per_source": 3} if advanced else {}
    res = TavilyClient(config.TAVILY_API_KEY).search(query, max_results=4, include_answer="basic",
                                                     include_images=images, include_image_descriptions=images, **kw)
    f.write_text(json.dumps(res))
    return res, False


def extract(urls: list[str], query: str) -> tuple[dict[str, str], bool]:
    """Tavily /extract: read the pages themselves (not just search snippets) and keep the passages
    relevant to `query`. Disk-cached. Returns ({url: text}, was_cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = "extract:" + query.strip().lower() + "|" + "|".join(sorted(urls))
    f = CACHE / (hashlib.sha256(key.encode()).hexdigest()[:16] + ".json")
    if f.exists():
        return json.loads(f.read_text()), True
    res = TavilyClient(config.TAVILY_API_KEY).extract(urls, query=query, chunks_per_source=3, timeout=30)
    out = {r["url"]: (r.get("raw_content") or "")[:2500] for r in res.get("results", [])}
    f.write_text(json.dumps(out))
    return out, False


SIZE_PATTERN = re.compile(r"\d+(?:[.,]\d+)?\s*(?:mm|cm|in\b|inch|inches|\"|millimet|centimet)", re.IGNORECASE)
MATERIAL_WORDS = {"metal": ("metal", "steel", "alumin", "iron", "brass", "copper", "tin"),
                  "plastic": ("plastic", "polyethylene", "polypropylene", "abs", "pvc", "acrylic", "nylon", "ptfe", "polystyrene"),
                  "rubber": ("rubber",), "glass": ("glass",), "ceramic": ("ceramic", "porcelain"),
                  "cardboard": ("cardboard", "paperboard", "corrugated"), "paper": ("paper",),
                  "fabric": ("fabric", "felt", "cloth", "cotton", "textile", "wool"), "foam": ("foam", "polystyrene", "polyurethane"),
                  "fruit": ("apple", "fruit", "orange", "banana", "lime", "lemon"), "wood": ("wood", "oak", "pine", "timber")}
TABLE_WORDS = ("wood", "oak", "pine", "timber", "table", "plywood")
BAD_CONTEXT = ("wet", "lubricat", "greas", "oiled", "example", "hypothetical", "assume", "suppose", "calculated as")


def context_ok(quote: str, material: str, kind: str) -> tuple[bool, str]:
    """Does a quote measure the right thing? The text must exist (checked separately); this checks it
    is about this material, on a table-like surface for friction, and not a wet surface or a worked
    example ("wood on wet metal 0.2" is not metal on a dry table)."""
    q = quote.lower()
    if any(b in q for b in BAD_CONTEXT):
        return False, "quote is about a wet surface or a worked example"
    if not any(w in q for w in MATERIAL_WORDS.get(material, (material,))):
        return False, f"quote doesn't name {material}"
    if kind == "friction":
        others = q
        for w in MATERIAL_WORDS.get(material, (material,)):
            others = others.replace(w, " ", 1)  # a "wood on wood" quote must name wood twice
        if not any(w in others for w in TABLE_WORDS):
            return False, "quote isn't against wood or a table"
    return True, ""


AGREE_BAND = 0.08  # two sizes "agree" if their largest dimensions differ by at most 8 %


def _agreement(e: _Extracted, fact: ObjectFact) -> tuple[int, int]:
    """(independent evidence sources that gave a size, how many agree with the chosen size within
    AGREE_BAND). The chosen quote itself is not counted: a source can't vote for itself."""
    sizes = [ev.size_mm for ev in e.evidence if ev.size_mm and max(ev.size_mm) > 0]
    chosen = max(fact.size) * 1000
    agree = sum(1 for sz in sizes if abs(max(sz) - chosen) <= AGREE_BAND * chosen)
    return len(sizes), agree


def _verified_evidence(e: _Extracted, page_text: dict[str, str]) -> list[_Evidence]:
    """Keep only evidence the code can check: from a page Tavily actually returned (not the search
    summary, which is built from those same pages), one vote per page, and with the size's largest
    value written in that page's text (as mm, cm or inches). Unchecked votes would let the extraction
    model invent "4/4 agree"."""
    out, seen = [], set()
    for ev in e.evidence:
        text = page_text.get(ev.source)
        if not text or ev.source in seen or ev.source == e.source_url or not ev.size_mm:
            continue
        mx = max(ev.size_mm)
        if mx <= 0:
            continue
        nums = set(re.findall(r"\d+(?:[.,]\d+)?", text))
        forms = {f"{mx:g}", f"{mx / 10:g}", f"{mx / 10:.1f}", f"{mx / 25.4:.1f}", f"{mx / 25.4:.2f}", f"{round(mx / 25.4 * 4) / 4:g}"}
        if not any(f.replace(".0", "") in nums or f in nums for f in forms):
            continue
        seen.add(ev.source)
        out.append(ev)
    return out


def _consensus(e: _Extracted) -> _Extracted:
    """With three or more sizes from the web, use the median one: a single outlier page (a farm
    guide's 8.9 cm 'apple') then can't decide whether a task is refused."""
    sizes = [tuple(e.size_mm)] if e.quote.strip() and max(e.size_mm) > 0 else []
    sizes += [tuple(ev.size_mm) for ev in e.evidence if ev.size_mm and max(ev.size_mm) > 0]
    if len(sizes) < 3:
        return e
    med = sorted(sizes, key=max)[len(sizes) // 2]
    if abs(max(med) - max(e.size_mm)) <= AGREE_BAND * max(med):
        return e
    return e.model_copy(update={"size_mm": med, "note": f"median of {len(sizes)} web sizes "
                                f"({sorted(round(max(x) / 10, 1) for x in sizes)} cm)"})


def _clamp(v: float, lo_hi: tuple[float, float]) -> tuple[float, bool]:
    lo, hi = lo_hi
    return min(max(v, lo), hi), not lo <= v <= hi


def _block(label: str, q: str, res: dict) -> str:
    lines = [f"## {label}  (search: {q})"]
    if res.get("answer"):
        lines.append(f"Summary: {res['answer']}")
    lines += [f"- [{r.get('title', '')}]({r['url']}): {r.get('content', '')[:700]}" for r in res.get("results", [])]
    return "\n".join(lines)


@traceable(name="physics_facts")
def physics_facts(materials: list[str], budget: int) -> tuple[list[MaterialFact], RobotFact, list[CallStat], dict]:
    """Material friction/bounciness and robot limits, cached forever; uncached ones are searched
    (within `budget` searches), extracted by Nano in one call, and kept only if their quotes are
    found in the fetched text and the values are physically plausible."""
    PHYS_CACHE.mkdir(parents=True, exist_ok=True)
    mats, todo = {}, []
    for m in materials:
        f = PHYS_CACHE / f"material_{m}.json"
        if f.exists():
            mats[m] = MaterialFact(**json.loads(f.read_text()))
        else:
            todo.append(m)
    rf = PHYS_CACHE / "robot_franka_panda.json"
    robot = RobotFact(**json.loads(rf.read_text())) if rf.exists() else None
    stats, info = [], {"searched": [], "cached": sorted(mats) + (["robot"] if robot else [])}
    wanted = [("ROBOT GRIPPER", ROBOT_QUERIES["gripper"]), ("ROBOT ARM", ROBOT_QUERIES["arm"])] if robot is None else []
    for m in todo:
        wanted += [(f"MATERIAL {m} FRICTION", FRICTION_QUERY.get(m, f"{m} on wood coefficient of friction table")),
                   (f"MATERIAL {m} RESTITUTION", RESTITUTION_QUERY.get(m, f"{m} coefficient of restitution value"))]
    blocks = []
    for label, q in wanted[:budget]:
        try:
            res, _ = search(q, images=False, advanced=True)
        except Exception as e:  # noqa: BLE001
            blocks.append(f"## {label}\n(search failed: {e})")
            continue
        info["searched"].append(q)
        block = _block(label, q, res)
        # Read the top two pages themselves (Tavily /extract, only the passages about this query):
        # engineering tables are often cut off in search snippets.
        top = [r["url"] for r in res.get("results", [])[:2]]
        if top:
            try:
                pages, _ = extract(top, q)
                info.setdefault("extracted", []).extend(pages)
                block += "".join(f"\n- FULL PAGE [{u}]: {t}" for u, t in pages.items() if t)
            except Exception as e:  # noqa: BLE001 - extraction is a bonus; snippets still work
                info.setdefault("extract_errors", []).append(str(e)[:120])
        blocks.append(block)
    corpus = _norm("\n".join(blocks))
    searched_mats = {b.split("\n")[0].split()[2] for b in blocks if b.startswith("## MATERIAL")}
    if blocks:
        ex, stats = chat_json("grounder_physics", config.NANO, PHYS_SYSTEM, "\n\n".join(blocks), _PhysicsEx,
                              max_tokens=2500, temperature=0.1, thinking=False)
        for me in ex.materials:
            m = me.material.strip().lower()
            if m not in todo or m not in searched_mats:
                continue
            (flo, fhi), (rlo, rhi) = PLAUSIBLE.get(m, ((0.1, 1.2), (0.05, 0.9)))
            typ_f, typ_r = TYPICAL.get(m, (0.5, 0.4))
            notes = []
            f_ctx, f_why = context_ok(me.friction_quote, m, "friction")
            r_ctx, r_why = context_ok(me.restitution_quote, m, "restitution")
            f_ok = flo <= me.friction <= fhi and verified(me.friction_quote, corpus) and f_ctx
            r_ok = rlo <= me.restitution <= rhi and verified(me.restitution_quote, corpus) and r_ctx
            if not f_ok:
                notes.append("friction: typical value" + (f" (web {me.friction:g} rejected"
                                                          + (f": {f_why})" if me.friction and not f_ctx else ")") if me.friction else ""))
            if not r_ok:
                notes.append("bounciness: typical value" + (f" (web {me.restitution:g} rejected"
                                                            + (f": {r_why})" if me.restitution and not r_ctx else ")") if me.restitution else ""))
            mats[m] = MaterialFact(
                material=m, friction=round(me.friction if f_ok else typ_f, 3), restitution=round(me.restitution if r_ok else typ_r, 3),
                friction_source=me.friction_source if f_ok else None, friction_quote=me.friction_quote[:240] if f_ok else "",
                restitution_source=me.restitution_source if r_ok else None,
                restitution_quote=me.restitution_quote[:240] if r_ok else "", note="; ".join(notes))
            (PHYS_CACHE / f"material_{m}.json").write_text(mats[m].model_dump_json())
        if ex.robot and robot is None and any(b.startswith("## ROBOT") for b in blocks):
            r = ex.robot
            g_ok = 60 <= r.gripper_opening_mm <= 120 and verified(r.gripper_quote, corpus)
            a_ok = 1 <= r.payload_kg <= 10 and 600 <= r.reach_mm <= 1200 and verified(r.arm_quote, corpus)
            robot = RobotFact(
                gripper_opening_mm=r.gripper_opening_mm if g_ok else ROBOT_TYPICAL[0],
                payload_kg=r.payload_kg if a_ok else ROBOT_TYPICAL[1], reach_mm=r.reach_mm if a_ok else ROBOT_TYPICAL[2],
                gripper_source=r.gripper_source if g_ok else None, gripper_quote=r.gripper_quote[:240] if g_ok else "",
                arm_source=r.arm_source if a_ok else None, arm_quote=r.arm_quote[:240] if a_ok else "",
                note="; ".join(x for x in ("" if g_ok else "gripper: spec-sheet value (web quote not verified)",
                                           "" if a_ok else "arm: spec-sheet value (web quote not verified)") if x))
            rf.write_text(robot.model_dump_json())
    for m in materials:  # anything still missing (search budget): labelled typical value, not cached
        if m not in mats:
            fr, re_ = TYPICAL.get(m, (0.5, 0.4))
            mats[m] = MaterialFact(material=m, friction=fr, restitution=re_, note="typical value (not searched yet)")
    if robot is None:
        robot = RobotFact(gripper_opening_mm=ROBOT_TYPICAL[0], payload_kg=ROBOT_TYPICAL[1], reach_mm=ROBOT_TYPICAL[2],
                          note="spec-sheet values (not searched yet)")
    return [mats[m] for m in materials], robot, stats, info


@traceable(name="grounder")
def ground(task: TaskSpec) -> tuple[GroundedFacts, list[Source], dict, list[CallStat]]:
    """Returns facts, the source list for the UI, Tavily stats, and LLM call stats."""
    blocks, sources, images, cached, t0 = [], [], [], 0, time.time()
    page_text: dict[str, str] = {}  # url -> the text Tavily returned for it (to check evidence)
    extracted: list[dict] = []
    search_s = 0.0
    queries = task.search_queries[:MAX_SEARCHES]
    for q in queries:
        ts = time.time()
        try:
            res, hit = search(q)
        except Exception as e:  # noqa: BLE001 - network/quota problems must not kill the run
            blocks.append(f"## Search: {q}\n(search failed: {e})")
            continue
        search_s += time.time() - ts
        cached += hit
        lines = [f"## Search: {q}"]
        if res.get("answer"):
            lines.append(f"Summary: {res['answer']}")
        for r in res.get("results", []):
            sources.append(Source(title=r.get("title", r["url"])[:90], url=r["url"]))
            page_text[r["url"]] = page_text.get(r["url"], "") + " " + r.get("content", "")[:600]
            lines.append(f"- [{r.get('title', '')}]({r['url']}): {r.get('content', '')[:600]}")
        # No size in any snippet: read the top two pages themselves with Tavily /extract (only the
        # passages about the query), at most MAX_EXTRACTS times per run.
        snippets = " ".join(r.get("content", "") for r in res.get("results", []))
        top = [r["url"] for r in res.get("results", [])[:2]]
        if top and not SIZE_PATTERN.search(snippets) and len(extracted) < MAX_EXTRACTS:
            try:
                pages, hit_x = extract(top, q + " dimensions size weight")
                extracted.append({"query": q, "urls": list(pages), "cached": hit_x})
                for u, t in pages.items():
                    if t:
                        page_text[u] = page_text.get(u, "") + " " + t
                        lines.append(f"- FULL PAGE [{u}]: {t[:1500]}")
            except Exception as e:  # noqa: BLE001 - the snippets still work without it
                extracted.append({"query": q, "error": str(e)[:120]})
        blocks.append("\n".join(lines))
        for im in res.get("images", []) or []:
            url, desc = (im.get("url"), im.get("description", "")) if isinstance(im, dict) else (im, "")
            if url:
                images.append((url, (desc or "")[:140]))
    img_block = "\n".join(f"[{i}] {d or '(no description)'}" for i, (_, d) in enumerate(images[:12]))

    objects = "\n".join(f"- {o.name}: {o.description} ({o.shape}, {o.color})" for o in task.objects)
    user = (f"Objects:\n{objects}\n\nSearch results:\n" + "\n\n".join(blocks)
            + f"\n\nIMAGES:\n{img_block or '(none)'}")
    ex, stats = chat_json("grounder", config.NANO, SYSTEM, user, _Extraction, max_tokens=4000,
                          temperature=0.1, thinking=False)
    shapes = {o.name: o.shape for o in task.objects}
    priors = {o.name: o.approx_size_cm for o in task.objects}
    mats = {o.name: o.material for o in task.objects}
    out = []
    corpus = _norm("\n".join(blocks))
    returned = {src.url for src in sources}
    for e in ex.facts:
        e = e.model_copy(update={"evidence": _verified_evidence(e, page_text)})
        if e.quote.strip() and not verified(e.quote, corpus):  # invented citation: keep numbers, drop the source
            e = e.model_copy(update={"quote": "", "evidence": [], "note": "estimate (quote not found in sources)"})
        if e.source_url and e.source_url not in returned:  # a URL Tavily never returned: drop it
            e = e.model_copy(update={"source_url": None})
        e = _consensus(e)
        f = _to_fact(e, shapes.get(e.name, "box"), priors.get(e.name), mats.get(e.name))
        checked, agree = _agreement(e, f) if not f.note.startswith("web size") else (0, 0)
        f.sources_checked, f.sources_agree = checked, agree
        f.material = mats.get(e.name)
        if 0 <= e.image < min(len(images), 12) and not f.note.startswith("web size"):
            f.image_url = images[e.image][0]
        out.append(f)
    facts = GroundedFacts(facts=out)
    _fit_drawers(task, facts)

    uncached_used = len(queries) - cached
    materials = sorted({o.material for o in task.objects})
    facts.materials, facts.robot, pstats, pinfo = physics_facts(materials, max(0, MAX_UNCACHED - uncached_used))
    stats += pstats
    tavily_stats = {"queries": queries, "cached": cached, "latency_s": round(search_s, 2), "extracted": extracted,
                    "total_s": round(time.time() - t0, 2),
                    "physics_searched": pinfo["searched"], "physics_cached": pinfo["cached"]}
    # Keep the sources list short and unique for the UI.
    seen, uniq = set(), []
    for s in sources:
        if s.url not in seen:
            seen.add(s.url)
            uniq.append(s)
    return facts, uniq[:8], tavily_stats, stats
