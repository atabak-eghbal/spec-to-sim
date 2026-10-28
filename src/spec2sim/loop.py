"""Orchestration: plan -> ground -> build -> (run -> critique -> repair)* with model escalation.

`run_task` is a generator of plain-dict events (JSON-serializable) so the CLI can print them and
the FastAPI app can stream them over SSE unchanged.

Escalation policy (docs/architecture.md): attempt 1 is built by Super. After a failure the VL
Critic diagnoses it; the first repair uses Super, later repairs escalate to Ultra. Max
MAX_ITERATIONS attempts in total.
"""
from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from pathlib import Path

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from spec2sim import builder, config, critic, grounder, planner
from spec2sim.llm import CallStat, LLMError
from spec2sim.schemas import GroundedFacts, TaskSpec
from spec2sim.sim import runner
from spec2sim.sim.controller import MAX_GRASP_WIDTH

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"


def repair_model(attempt_failed: int) -> config.ModelInfo:
    """Model that repairs after attempt N failed: Super for the first repair, Ultra after."""
    return config.SUPER if attempt_failed == 1 else config.ULTRA


def infeasible(task: TaskSpec, facts: GroundedFacts) -> str | None:
    """Check in code, before spending on a build, whether the grounded objects can be grasped at all.

    The Panda's parallel gripper opens 8 cm; an object whose narrowest horizontal side is wider than
    that cannot be picked up, whatever the plan. Better to say so than to burn repair attempts.
    """
    shapes = {o.name: o.shape for o in task.objects}
    sizes = {f.name: f.size for f in facts.facts}
    movers = {g.obj for g in task.goals if g.type != "upright"}
    for name in sorted(movers):
        if name not in sizes or shapes.get(name) == "bin":
            continue
        sx, sy, _ = sizes[name]
        width = min(sx, sy) if shapes.get(name) == "box" else max(sx, sy)
        if width > MAX_GRASP_WIDTH:
            return (f"{name} is {width * 100:.1f} cm wide at its narrowest (real-world size from the "
                    f"Grounder), but the Franka gripper opens only {MAX_GRASP_WIDTH * 100:.1f} cm, "
                    "so no motion plan can pick it up.")
    return None


def _ev(type_: str, **kw) -> dict:
    return {"type": type_, "ts": round(time.time(), 3), **kw}


def _stats(stats: list[CallStat]) -> dict:
    return {"model": stats[-1].model, "label": stats[-1].label,
            "latency_s": round(sum(s.latency_s for s in stats), 2),
            "tokens_in": sum(s.tokens_in for s in stats), "tokens_out": sum(s.tokens_out for s in stats),
            "cost_usd": round(sum(s.cost_usd for s in stats), 6), "calls": len(stats)}


@traceable(name="spec2sim_run")
def run_task(prompt: str, run_id: str | None = None, max_iterations: int | None = None) -> Iterator[dict]:
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    out = RUNS / run_id
    max_it = max_iterations or config.MAX_ITERATIONS
    total_cost = 0.0
    rt = get_current_run_tree()
    yield _ev("start", run_id=run_id, prompt=prompt, trace_url=_trace_url(rt))

    yield _ev("step", step="planner", status="running", model=config.SUPER.label)
    task, st = planner.plan(prompt)
    total_cost += sum(s.cost_usd for s in st)
    yield _ev("step", step="planner", status="done", stats=_stats(st), data=task.model_dump())

    yield _ev("step", step="grounder", status="running", model=config.NANO.label)
    facts, sources, tav, st = grounder.ground(task)
    total_cost += sum(s.cost_usd for s in st)
    yield _ev("step", step="grounder", status="done", stats=_stats(st), tavily=tav,
              data=facts.model_dump(), sources=[s.model_dump() for s in sources])

    reason = infeasible(task, facts)
    if reason:
        yield _ev("done", success=False, attempts=0, infeasible=reason, total_cost_usd=round(total_cost, 6))
        return

    yield _ev("step", step="builder", status="running", model=config.SUPER.label, attempt=1)
    b, st = builder.build(task, facts)
    total_cost += sum(s.cost_usd for s in st)
    yield _ev("step", step="builder", status="done", attempt=1, stats=_stats(st), data=b.model_dump())

    result = None
    for attempt in range(1, max_it + 1):
        yield _ev("step", step="runner", status="running", attempt=attempt)
        result = _traced_run(b, task.goals, out / f"attempt{attempt}", {o.name for o in task.objects})
        yield _ev("step", step="runner", status="done", attempt=attempt, data=_public(result, run_id))
        if result.success:
            break
        if attempt == max_it:
            break

        yield _ev("step", step="critic", status="running", model=config.VL.label, attempt=attempt)
        crit, st = critic.critique(task, b, result)
        total_cost += sum(s.cost_usd for s in st)
        yield _ev("step", step="critic", status="done", attempt=attempt, stats=_stats(st),
                  data=crit.model_dump())

        m = repair_model(attempt)
        yield _ev("step", step="repair", status="running", model=m.label, attempt=attempt + 1,
                  escalated=m is config.ULTRA)
        try:
            b, st = builder.repair(task, facts, b, result, crit, m, attempt)
        except LLMError as e:
            if m is config.SUPER:
                raise
            # Ultra returned unusable output; one Super repair keeps the run alive.
            yield _ev("note", message=f"{m.label} reply unusable ({str(e)[:120]}); retrying the repair with {config.SUPER.label}")
            m = config.SUPER
            b, st = builder.repair(task, facts, b, result, crit, m, attempt)
        total_cost += sum(s.cost_usd for s in st)
        yield _ev("step", step="repair", status="done", attempt=attempt + 1, stats=_stats(st),
                  escalated=m is config.ULTRA, data=b.model_dump())

    yield _ev("done", success=bool(result and result.success), attempts=attempt,
              total_cost_usd=round(total_cost, 6))


@traceable(name="runner", run_type="tool")
def _traced_run(b, goals, out_dir: Path, allowed: set[str]):
    return runner.run(b, goals, out_dir, allowed)


def _public(r, run_id: str) -> dict:
    """RunResult with paths made relative to runs/ so the web app can serve them."""
    d = r.model_dump()
    rel = lambda p: str(Path(p).relative_to(RUNS)) if p else None
    d["video_path"] = rel(r.video_path)
    d["contact_sheet_path"] = rel(r.contact_sheet_path)
    d["viz_path"] = rel(r.viz_path)
    d["frame_paths"] = [rel(p) for p in r.frame_paths]
    return d


def _trace_url(rt) -> str | None:
    try:
        return rt.get_url() if rt else None
    except Exception:  # noqa: BLE001 - a missing trace link must not fail the run
        return None
