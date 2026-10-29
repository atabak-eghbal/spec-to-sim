"""Step 5 - Critic (Nemotron 3.5 Super VL): watches the run and explains WHY it failed.

Input: one contact sheet of 8 timestamped frames + the measured goal results + sim events.
Pass/fail is already decided in code; the Critic's job is diagnosis and a concrete fix the
repair step can act on.
"""
from __future__ import annotations

from langsmith import traceable

from spec2sim import config
from spec2sim.llm import CallStat, chat_json
from spec2sim.schemas import Build, Critique, RunResult, TaskSpec

SYSTEM = """You are a robotics simulation reviewer. You see a contact sheet of 8 frames (left to
right, top to bottom, timestamped) from a MuJoCo simulation of a Franka Panda arm, plus the
measured results, which are ground truth. The run FAILED. Diagnose why.

First look at the frames and describe what they show (where the gripper went, whether the object
moved with it, where it ended up) before reading the numbers. Then quote the measurements, name the
root cause, and propose a concrete fix to the scene or motion plan with numbers
(e.g. "shift the place target 4 cm in -y", "move the mug 8 cm closer so it is reachable").
SIM EVENTS ARE GROUND TRUTH: if an event says "grasped X", the grasp worked - do not blame the
grasp. A release event states how high the object's bottom was above what was beneath it: more
than ~1 cm means it was dropped. Ground everything in the measurements and events; use the images
to explain them. Be concise. JSON only."""


def _plan_lines(b: Build) -> str:
    out = []
    for i, a in enumerate(b.plan.actions, 1):
        if a.op == "move_to":
            tgt = f"{a.object} + offset {list(a.offset)}" if a.object else f"xyz {list(a.xyz or ())}"
            out.append(f"{i}. move_to {tgt} yaw={a.yaw}")
        else:
            out.append(f"{i}. {a.op}")
    return "\n".join(out)


def critic_input(task: TaskSpec, b: Build, result: RunResult) -> str:
    failed = [g.detail for g in result.goals if not g.ok] or ["(no goal results)"]
    return (f"TASK: {task.summary}\n\n"
            "FAILED GOALS (measured):\n" + "\n".join(f"- {d}" for d in failed)
            + (f"\nRUN ERROR: {result.error}" if result.error else "")
            + "\n\nSIM EVENTS (ground truth, in order):\n" + "\n".join(f"- {e}" for e in result.events)
            + "\n\nPLAN THAT WAS EXECUTED:\n" + _plan_lines(b)
            + "\n\nOBJECTS (name, size m, start pos m): "
            + "; ".join(f"{o.name} {list(o.size)} @ {list(o.pos)}" for o in b.objects))


@traceable(name="critic")
def critique(task: TaskSpec, b: Build, result: RunResult) -> tuple[Critique, list[CallStat]]:
    images = [result.contact_sheet_path] if result.contact_sheet_path else None
    return chat_json("critic", config.VL, SYSTEM, critic_input(task, b, result), Critique,
                     images=images, max_tokens=2000, temperature=0.1, thinking=False)


def measured_only(result: RunResult, simulated: bool) -> Critique:
    """A diagnosis written by code from the measurements alone, with no model call. Used when nothing
    was simulated (the scene failed validation, so there are no frames to watch) and for the
    no-Critic comparison in the eval (does watching the video help repair?)."""
    failed = [g.detail for g in result.goals if not g.ok]
    evidence = "; ".join(([result.error] if result.error else []) + failed) or "no goal results"
    if not simulated:
        return Critique(evidence=evidence, category="scene_error",
                        what_happened="Nothing was simulated: the scene failed validation, so there are no frames to watch.",
                        root_cause=result.error or "invalid scene",
                        suggested_fix="Fix every scene problem listed in the evidence, keeping the rest of the build.")
    return Critique(evidence=evidence, category="other",
                    what_happened="(no video review: measured results only)",
                    root_cause="See the failed goals and sim events in the evidence.",
                    suggested_fix="Change the scene or plan so every failed goal passes, using the measured numbers.")
