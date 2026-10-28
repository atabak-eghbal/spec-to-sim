# Batch evaluation

`scripts/eval.py` runs a fixed suite of 11 prompts through the full pipeline and
reports success, first-try success, attempts, LLM cost and wall time. Success is decided only by
the code-checked goal predicates (`src/spec2sim/sim/metrics.py`), never by a model. The same
script is the entrypoint of the Nebius Serverless Job (`deploy/nebius/README.md`). The 11 prompts are
8 pick-and-place style tasks (easy to hard) and 3 "beyond pick-and-place" tasks (two drawer
open/insert/close tasks and a knock-over).

```bash
uv run python scripts/eval.py --reps 3 --workers 2
```

## How the numbers moved (Oct 28, 2026, local MacBook, 1 rep each)

Each eval round exposed failures that were then fixed in code, not by tuning prompts until they passed:

| Round | Success | First try | LLM cost | What the failures taught us |
|---|---|---|---|---|
| 1 (before harder-task guards) | 4/8 | 3/8 | $0.20 | Planner put a goal inside the robot base; IK failed inside the stated workspace; the new gripper-width check caught web sizes that were wrong (a 44 cm "toy block", a soda can with its dimensions scrambled) |
| 2 (hard prompts only) | 2/3 | 2/3 | $0.06 | One scene error found per attempt (bin over the edge, then bin too small, then an invented table) |
| 3 (all fixes) | **8/8** | 3/8 | **$0.17** | First-try misses: invented "table" object (2×), a 10 cm bin for three blocks, the tennis ball rolling off (real physics) |
| 4 (new: drawers, knock-over; 3 prompts) | 2/3 | 1/3 | $0.09 | Builder ignored the cabinet placement rule; a repair copied the prompt's placeholder name `D.handle`; a second object was dropped on top of the first in the drawer |
| 5 (same 3, after fixes) | **3/3** | **3/3** | **$0.02** | Drawer placement moved into code (like bin sizing); real names in the prompt; side-by-side slot offsets |

## Round 3 detail

| Difficulty | Prompt | Success | Attempts | Cost | Time |
|---|---|---|---|---|---|
| easy | A Franka arm stacks the red cube on the blue cube | ✅ | 2 | $0.006 | 46 s |
| easy | Put the Rubik's cube on top of the soda can | ✅ | 1 | $0.004 | 40 s |
| medium | Pick up the tennis ball and place it on the Rubik's cube | ✅ | 3 | $0.045 | 110 s |
| medium | Stack three wooden toy blocks into a tower: red at the bottom, then green, then blue on top | ✅ | 1 | $0.007 | 61 s |
| medium | Clear the table: put the red, green and blue toy blocks into the storage bin | ✅ | 3 | $0.067 | 164 s |
| hard | Put the apple and the lime into the fruit basket, and the soda can into the recycling bin | ✅ | 2 | $0.017 | 119 s |
| hard | Tidy up: stack the red block on the blue block, then put the green block and the tennis ball into the tray | ✅ | 2 | $0.016 | 101 s |
| hard | Put the soda can to the left of the Rubik's cube and keep it upright | ✅ | 1 | $0.006 | 38 s |

After round 3, the two most common first-try misses were fixed as well (the Builder may no longer add
objects such as a table; bins narrower than 15 cm are enlarged). The next run, on Nebius, uses 3
reps per prompt.

## Caveats
- One rep per prompt is a small sample. The same prompt can pass on the first try in one run and
  need 3 attempts in the next, because model output varies even at low temperature.
- Grasping uses the documented grasp assist (a weld when the closed gripper is within 2 cm of an
  object that fits between the fingers). Placement, stacking stability and rolling are physical.
