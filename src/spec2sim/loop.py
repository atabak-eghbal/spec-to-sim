"""Orchestration: plan -> ground -> build -> scene check -> (run -> critique -> repair)*.

`run_task` is a generator of plain-dict events (JSON-serializable) so the CLI can print them and
the FastAPI app can stream them over SSE unchanged.

After a success, the plan is stress-tested: re-simulated under random perturbations (start pose,
mass, friction) with no model calls, like Isaac Lab-Arena's sensitivity analysis.

Policy: attempt 1 is built by Super; before it runs, Super VL checks a render of the scene
(advisory). After a failure the VL Critic diagnoses the video, then REPAIR_CANDIDATES Super repairs
are generated in parallel, each is simulated physics-only, and the best one runs next. (Escalating
to Ultra was dropped after the Oct 28 eval: it fixed 4 of 18 runs.) Max MAX_ITERATIONS attempts.
"""
from __future__ import annotations

import os
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
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


def infeasible(task: TaskSpec, facts: GroundedFacts) -> tuple[str, list[str]] | None:
    """Check in code, before spending on a build, whether the grounded objects can be held at all.

    Both sides of the comparison are sourced: the object's size and mass come from the web, and so
    do the Franka gripper's opening and payload (from its spec sheet, via the Grounder). The usable
    opening keeps a 2 mm margin. Better to say so than to burn repair attempts.
    """
    shapes = {o.name: o.shape for o in task.objects}
    by = {f.name: f for f in facts.facts}
    robot = facts.robot
    opening = (robot.gripper_opening_mm - 2) / 1000 if robot else MAX_GRASP_WIDTH
    payload = robot.payload_kg if robot else 3.0
    # Only objects that must be picked up need to fit the gripper (pushing or knocking over doesn't).
    movers = {g.obj for g in task.goals if g.type not in ("upright", "toppled", "opened", "closed")}
    for name in sorted(movers):
        f = by.get(name)
        if f is None or shapes.get(name) in ("bin", "drawer"):
            continue
        sx, sy, _ = f.size
        width = min(sx, sy) if shapes.get(name) == "box" else max(sx, sy)
        # Refuse only when the evidence agrees: the Planner's own typical size is too wide as well, or
        # two or more independent web sources agree (within 8 %) on the too-wide size. A single odd web number (an orange
        # listed at 8.9 cm) is not enough to stop a task; the simulator will test the grasp instead.
        est = next((o.approx_size_cm for o in task.objects if o.name == name), None)
        est_w = (min(est[:2]) if shapes.get(name) == "box" else max(est[:2])) / 100 if est else None
        # (A size with no web source is a model's guess: it can't refuse a task on its own.)
        confirmed = (est_w is not None and est_w > opening) or f.sources_agree >= 2
        if width > opening and confirmed:
            origin = ("size implied by its weight from the web" if f.note.startswith("size corrected")
                      else "size from the web" if f.source_url else "estimated size")
            full = robot.gripper_opening_mm / 10 if robot else 8.0
            msg = (f"{name} is {width * 100:.1f} cm wide at its narrowest ({origin}), but the Franka gripper opens "
                   f"{full:g} cm (Franka datasheet; {opening * 100:.1f} cm usable after 1 mm clearance per side), "
                   "so no motion plan can pick it up.")
            return msg, [u for u in (f.source_url, robot.gripper_source if robot else None) if u]
        if f.mass > payload:
            return (f"{name} weighs {f.mass:.1f} kg, more than the Franka's {payload:g} kg payload (Franka datasheet).",
                    [u for u in (f.source_url, robot.arm_source if robot else None) if u])
    return None


def _ev(type_: str, **kw) -> dict:
    return {"type": type_, "ts": round(time.time(), 3), **kw}


def _stats(stats: list[CallStat]) -> dict:
    return {"model": stats[-1].model, "label": stats[-1].label,
            "latency_s": round(sum(s.latency_s for s in stats), 2),
            "tokens_in": sum(s.tokens_in for s in stats), "tokens_out": sum(s.tokens_out for s in stats),
            "cost_usd": round(sum(s.cost_usd for s in stats), 6), "calls": len(stats)}


@traceable(name="spec2sim_run")
def run_task(prompt: str, run_id: str | None = None, max_iterations: int | None = None,
             use_critic: bool = True, use_scene_check: bool = True) -> Iterator[dict]:
    """use_critic=False: repairs get the measured failures only, no VLM review (eval comparison)."""
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

    stop = infeasible(task, facts)
    if stop:
        yield _ev("done", success=False, attempts=0, infeasible=stop[0], infeasible_sources=stop[1],
                  total_cost_usd=round(total_cost, 6))
        return

    yield _ev("step", step="builder", status="running", model=config.SUPER.label, attempt=1)
    b, st = builder.build(task, facts)
    total_cost += sum(s.cost_usd for s in st)
    yield _ev("step", step="builder", status="done", attempt=1, stats=_stats(st), data=b.model_dump())

    result, history, allowed, physics = None, [], {o.name for o in task.objects}, facts.physics()
    for attempt in range(1, max_it + 1):
        b, notes = fit_furniture(b)
        for n in notes:
            yield _ev("note", message=n)
        if attempt == 1 and use_scene_check:
            # The vision model looks at the scene as built, before the robot moves (advisory: it
            # never blocks a run; the eval measures how often its warnings predict a failure).
            img = runner.snapshot(b, out / "scene_check.jpg", allowed, physics)
            if img:
                yield _ev("step", step="scenecheck", status="running", model=config.VL.label)
                try:
                    rev, st = critic.scene_check(task, b, img)
                    total_cost += sum(s.cost_usd for s in st)
                    yield _ev("step", step="scenecheck", status="done", stats=_stats(st),
                              data=rev.model_dump() | {"image": str(Path(img).relative_to(RUNS))})
                except LLMError as e:
                    yield _ev("step", step="scenecheck", status="done", data={"error": str(e)[:200]})
        yield _ev("step", step="runner", status="running", attempt=attempt)
        result = _traced_run(b, task.goals, out / f"attempt{attempt}", allowed, physics)
        yield _ev("step", step="runner", status="done", attempt=attempt, data=_public(result, run_id))
        history.append((b, result))
        if result.success:
            break
        if attempt == max_it:
            break

        simulated = result.contact_sheet_path is not None
        if simulated and use_critic:
            yield _ev("step", step="critic", status="running", model=config.VL.label, attempt=attempt)
            crit, st = critic.critique(task, b, result)
            total_cost += sum(s.cost_usd for s in st)
            yield _ev("step", step="critic", status="done", attempt=attempt, stats=_stats(st),
                      data=crit.model_dump())
        else:  # nothing to watch (scene invalid), or the no-Critic comparison: no model call
            crit = critic.measured_only(result, simulated)
            yield _ev("step", step="critic", status="done", attempt=attempt, skipped=True,
                      data=crit.model_dump())

        # Repair: REPAIR_CANDIDATES Super repairs in parallel (different temperatures), each tested in
        # the simulator physics-only (seconds, no model cost): nominal run, then a few perturbed
        # re-runs, so a plan that only works in the exact scene loses to a sturdier one.
        # This replaced escalating to Ultra, which fixed 4 of 18 runs in the Oct 28 eval.
        yield _ev("step", step="repair", status="running", model=config.SUPER.label, attempt=attempt + 1,
                  candidates=config.REPAIR_CANDIDATES)
        temps = [0.3] + [0.8] * (config.REPAIR_CANDIDATES - 1)
        with ThreadPoolExecutor(len(temps)) as ex:
            futs = [ex.submit(builder.repair, task, facts, history, crit, config.SUPER, attempt, t) for t in temps]
            got, stats, errors = [], [], []
            for f in futs:
                try:
                    cand, st = f.result()
                    got.append(fit_furniture(cand)[0])
                    stats += st
                except LLMError as e:
                    errors.append(str(e)[:120])
        if not got:
            raise LLMError(f"all {len(temps)} repair candidates failed: {errors}")
        scores = [runner.quick_score(c, task.goals, allowed, physics) for c in got]
        best = max(range(len(got)), key=lambda i: (scores[i]["ok"], scores[i].get("robust", 0), scores[i]["passed"],
                                                   -scores[i]["miss_m"]))
        b = got[best]
        total_cost += sum(s.cost_usd for s in stats)
        yield _ev("step", step="repair", status="done", attempt=attempt + 1, stats=_stats(stats),
                  data=b.model_dump(), candidates=[s | {"chosen": i == best} for i, s in enumerate(scores)],
                  candidate_errors=errors)

    if result and result.success and config.STRESS_TRIALS > 0:
        yield _ev("step", step="stress", status="running", trials=config.STRESS_TRIALS)
        st = _traced_stress(b, task.goals, config.STRESS_TRIALS, facts.physics())
        yield _ev("step", step="stress", status="done", data=st)

    public = _share_trace(rt)
    if public:
        yield _ev("trace", trace_url=public)
    yield _ev("done", success=bool(result and result.success), attempts=attempt,
              total_cost_usd=round(total_cost, 6))


def _share_trace(rt) -> str | None:
    """With SPEC2SIM_SHARE_TRACES=1 (the public demo), publish this run's LangSmith trace so anyone
    can open the link; otherwise trace links stay inside the team's workspace."""
    if os.getenv("SPEC2SIM_SHARE_TRACES") != "1" or rt is None:
        return None
    try:
        from langsmith import Client
        return Client().share_run(rt.id)
    except Exception:  # noqa: BLE001 - a missing public link must not fail the run
        return None


def fit_furniture(b) -> tuple[object, list[str]]:
    """Move a drawer cabinet into the x range where its handle can be pulled open within reach.
    Deterministic, like rescaling bins: models often ignore this placement rule. Returns the build
    and a note per change (shown in the UI)."""
    from spec2sim.sim.scene import DRAWER_BACK_SLACK, DRAWER_PULL, HANDLE_STANDOFF, WORKSPACE
    notes, objs = [], []
    for o in b.objects:
        if o.shape == "drawer":
            hx = o.size[0] / 2
            lo = WORKSPACE["x"][0] + DRAWER_PULL + hx + HANDLE_STANDOFF
            hi = WORKSPACE["x"][1] + DRAWER_BACK_SLACK - hx
            x = min(max(o.pos[0], lo), hi) if lo <= hi else o.pos[0]
            if abs(x - o.pos[0]) > 1e-4 or o.yaw:
                notes.append(f"moved {o.name} to x={x:.2f} m (yaw 0) so its handle can be pulled open within reach")
                o = o.model_copy(update={"pos": (round(x + 0.002 * (x < hi), 4), o.pos[1], o.pos[2]), "yaw": 0.0})
        objs.append(o)
    return b.model_copy(update={"objects": objs}), notes


@traceable(name="stress_test", run_type="tool")
def _traced_stress(b, goals, trials: int, physics: dict) -> dict:
    """Stress test with the grasp assist (the main score), plus the same seeds without it: how often
    the fingers alone, by friction, would have held the objects (reported, not hidden)."""
    st = runner.stress_test(b, goals, trials=trials, physics=physics)
    raw = runner.stress_test(b, goals, trials=trials, physics=physics, assist=False)
    st["no_assist"] = {"passed": raw["passed"], "trials": raw["trials"]}
    return st


@traceable(name="runner", run_type="tool")
def _traced_run(b, goals, out_dir: Path, allowed: set[str], physics: dict):
    return runner.run(b, goals, out_dir, allowed, physics)


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
