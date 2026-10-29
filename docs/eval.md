# Batch evaluation

`scripts/eval.py` runs prompt suites through the full pipeline and reports success, first-try
success, attempts, model cost, wall time and the stress-test score of each solved task. Success is
decided only by the code-checked goal predicates (`src/spec2sim/sim/metrics.py`), never by a model.

```bash
uv run python scripts/eval.py --reps 3 --no-critic-reps 1 --workers 3
```

## Round 2: the current pipeline (Oct 28, 2026, commit `5bb3636`)

After the second judge round the pipeline changed (commit `b367007`): the Builder writes skills
(`pick`, `place_on`, `drop_into`, `push`) and code computes the waypoints; repairs see every earlier
attempt; each repair is 3 Super candidates tested in the simulator, best one runs (Ultra escalation
removed); a vision scene check runs before the robot moves. Same 26 prompts, same method as round 1
below: 3 runs each plus 1 with the Critic off, 104 runs, `eval/20261028-223102/`.

| Suite | Success | First try | Model cost per task | Stress test (solved tasks) |
|---|---|---|---|---|
| Held-out, 15 prompts | **42/45 (93 %, 95 % CI 82–98 %)** | 30/45 (67 %) | median $0.005 | 554/624 (89 %) |
| Development, 11 prompts | 29/33 (88 %, CI 73–95 %) | 19/33 (58 %) | median $0.006 | 423/464 (91 %) |
| **All (Critic on)** | **71/78 (91 %, CI 83–96 %)** | 49/78 (63 %) | mean $0.009, median $0.005 | 977/1088 (90 %) |

**Caveat on "held out" in this round.** The 15 prompts were frozen before round 1 and never tuned,
but two of their round-1 failures (a planner "table" object; an orange refused as too wide) led to
general code fixes in `b367007`, so round 2 is not a fully blind test of those prompts. Round 1
(41/45) is the clean held-out number. Refusals are also counted as successes for the one
should-refuse prompt (basketball): without it, held-out success is 39/42.

Compared with round 1: success 70/78 → 71/78 (within noise), mean cost per task $0.013 → $0.010 (Critic-on runs; $0.009 over all 104)
(no Ultra), median time per task 47 s → 38 s. First try is lower (53 → 49 of 78), and repairs carry
more of the load: 26 of 46 repairs fixed the run (57 %), against 22 of 46 in round 1. All 104 runs
cost $0.94.

**What still fails (7 of 78 with the Critic on):**
- *Refused as too wide* (4): apples, and this was partly our bug. The size check counted the chosen
  source as agreeing with itself and used a loose ±20 % band, so one page's 8.9 cm apple could stop a
  task, while other runs grounded the same apple at 6.7–7.5 cm. Fixed after this round (commit
  `a0958fd`): sizes use the median of 3+ sources, only independent sources within ±8 % count as
  agreeing, and a size with no web source can't refuse a task. Re-grounding the 7 refused apple runs
  with the fix: 6 go ahead, 1 stays refused (three sources agree on 8.1 cm). Not yet re-run end to end.
- *Knock over* (2): pushing the soda can or the top block over still fails after 4 attempts.
- *Soda can left of the Rubik's cube* (1): 4 attempts without a pass.

**The vision model, measured twice:**
- *Critic (after a failure):* with it off, the same prompts succeeded 23/26 (88 %) against 71/78
  (91 %) with it on. Still no measurable difference.
- *Scene check (before the robot moves), new:* it flagged 16 of 77 scenes; only 3 of those then
  failed on attempt 1, and it called 10 scenes ok that then failed. As a predictor of failure it is
  close to noise. We keep it as a cheap ($0.0004) visible check in the UI, and we don't claim it helps.

**Verdict on the cascade:** the gains came from code (skills, cleaned planner output, oriented web
sizes) and from using the simulator as a verifier for several cheap Super repairs, not from bigger
models or from vision. The vision model's value here is explanation for humans.

### Round 2 per-prompt results

| Suite | Critic | Difficulty | Prompt | Runs | Success | First try | Avg attempts | Avg cost | Avg time | Stress (passed/trials) |
|---|---|---|---|---|---|---|---|---|---|---|
| dev | on | easy | A Franka arm stacks the red cube on the blue cube | 3 | 3/3 | 3/3 | 1.0 | $0.003 | 30 s | 16/16, 16/16, 16/16 |
| dev | on | easy | Put the Rubik's cube on top of the soda can | 3 | 3/3 | 3/3 | 1.0 | $0.004 | 31 s | 15/16, 16/16, 16/16 |
| dev | on | medium | Pick up the tennis ball and place it on the Rubik's cube | 3 | 3/3 | 2/3 | 1.3 | $0.005 | 30 s | 16/16, 16/16, 16/16 |
| dev | on | medium | Stack three wooden toy blocks into a tower: red at the bottom, then green, then blue on top | 3 | 3/3 | 2/3 | 1.3 | $0.008 | 50 s | 16/16, 0/16, 16/16 |
| dev | on | medium | Clear the table: put the red, green and blue toy blocks into the storage bin | 3 | 3/3 | 3/3 | 1.0 | $0.005 | 49 s | 16/16, 16/16, 16/16 |
| dev | on | hard | Put the apple and the lime into the fruit basket, and the soda can into the recycling bin | 3 | 1/3 | 0/3 | 2.0 | $0.014 | 71 s | 4/16 |
| dev | on | hard | Tidy up: stack the red block on the blue block, then put the green block and the tennis ball into the tray | 3 | 3/3 | 1/3 | 1.7 | $0.016 | 88 s | 15/16, 15/16, 14/16 |
| dev | on | hard | Put the soda can to the left of the Rubik's cube and keep it upright | 3 | 2/3 | 1/3 | 2.3 | $0.018 | 59 s | 12/16, 16/16 |
| dev | on | beyond | Open the drawer, put the Rubik's cube inside, and close it | 3 | 3/3 | 3/3 | 1.0 | $0.007 | 56 s | 16/16, 16/16, 16/16 |
| dev | on | beyond | Put the tennis ball and the Rubik's cube into the drawer and close it | 3 | 3/3 | 0/3 | 2.3 | $0.029 | 105 s | 15/16, 16/16, 14/16 |
| dev | on | beyond | Knock over the tower of three wooden toy blocks | 3 | 2/3 | 1/3 | 2.3 | $0.015 | 46 s | 16/16, 15/16 |
| heldout | on | easy | Put the lime on the red toy block | 3 | 3/3 | 2/3 | 2.0 | $0.014 | 149 s | 7/16, 11/16, 16/16 |
| heldout | on | easy | Stack the blue block on the green block | 3 | 3/3 | 2/3 | 1.3 | $0.004 | 30 s | 16/16, 16/16, 16/16 |
| heldout | on | easy | Put the Rubik's cube into the storage bin | 3 | 3/3 | 3/3 | 1.0 | $0.004 | 34 s | 16/16, 16/16, 16/16 |
| heldout | on | easy | Pick up the tennis ball and drop it into the fruit basket | 3 | 3/3 | 3/3 | 1.0 | $0.004 | 30 s | 16/16, 16/16, 16/16 |
| heldout | on | medium | Place the apple to the right of the soda can | 3 | 3/3 | 2/3 | 1.3 | $0.008 | 44 s | 16/16, 8/16, 16/16 |
| heldout | on | medium | Put the red block in front of the blue block | 3 | 3/3 | 0/3 | 2.3 | $0.015 | 150 s | 16/16, 6/16, 15/16 |
| heldout | on | medium | Put the orange and the apple into the fruit basket | 3 | 1/3 | 1/3 | 1.0 | $0.004 | 595 s | 16/16 |
| heldout | on | medium | Build a tower of two Rubik's cubes | 3 | 3/3 | 3/3 | 1.0 | $0.004 | 131 s | 16/16, 16/16, 16/16 |
| heldout | on | medium | Put the wooden block behind the soda can | 3 | 3/3 | 2/3 | 1.3 | $0.007 | 37 s | 12/16, 16/16, 16/16 |
| heldout | on | hard | Stack the green block on the Rubik's cube, then put the lime to the left of the Rubik's cube | 3 | 3/3 | 2/3 | 1.3 | $0.010 | 572 s | 16/16, 16/16, 5/16 |
| heldout | on | hard | Sort the toys: the red block goes into the red bin and the blue block into the blue bin | 3 | 3/3 | 3/3 | 1.0 | $0.010 | 601 s | 16/16, 16/16, 16/16 |
| heldout | on | beyond | Open the drawer | 3 | 3/3 | 1/3 | 1.7 | $0.008 | 514 s | 16/16, 16/16, 16/16 |
| heldout | on | beyond | Put the tennis ball into the drawer and close it | 3 | 3/3 | 3/3 | 1.0 | $0.007 | 51 s | 16/16, 16/16, 16/16 |
| heldout | on | beyond | Knock the soda can over | 3 | 2/3 | 0/3 | 3.7 | $0.023 | 57 s | 5/16, 5/16 |
| heldout | on | refuse | Put the basketball into the storage bin | 3 | 3/3 | 3/3 | 0.0 | $0.001 | 18 s | - |
| dev | off | easy | A Franka arm stacks the red cube on the blue cube | 1 | 1/1 | 1/1 | 1.0 | $0.003 | 30 s | 16/16 |
| dev | off | easy | Put the Rubik's cube on top of the soda can | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 26 s | 16/16 |
| dev | off | medium | Pick up the tennis ball and place it on the Rubik's cube | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 27 s | 16/16 |
| dev | off | medium | Stack three wooden toy blocks into a tower: red at the bottom, then green, then blue on top | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 42 s | 15/16 |
| dev | off | medium | Clear the table: put the red, green and blue toy blocks into the storage bin | 1 | 1/1 | 1/1 | 1.0 | $0.006 | 55 s | 16/16 |
| dev | off | hard | Put the apple and the lime into the fruit basket, and the soda can into the recycling bin | 1 | 0/1 | 0/1 | 0.0 | $0.002 | 17 s | - |
| dev | off | hard | Tidy up: stack the red block on the blue block, then put the green block and the tennis ball into the tray | 1 | 1/1 | 1/1 | 1.0 | $0.006 | 55 s | 12/16 |
| dev | off | hard | Put the soda can to the left of the Rubik's cube and keep it upright | 1 | 1/1 | 0/1 | 4.0 | $0.032 | 93 s | 15/16 |
| dev | off | beyond | Open the drawer, put the Rubik's cube inside, and close it | 1 | 1/1 | 1/1 | 1.0 | $0.008 | 69 s | 16/16 |
| dev | off | beyond | Put the tennis ball and the Rubik's cube into the drawer and close it | 1 | 1/1 | 0/1 | 4.0 | $0.057 | 171 s | 16/16 |
| dev | off | beyond | Knock over the tower of three wooden toy blocks | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 37 s | 10/16 |
| heldout | off | easy | Put the lime on the red toy block | 1 | 1/1 | 1/1 | 1.0 | $0.003 | 22 s | 7/16 |
| heldout | off | easy | Stack the blue block on the green block | 1 | 1/1 | 1/1 | 1.0 | $0.003 | 32 s | 16/16 |
| heldout | off | easy | Put the Rubik's cube into the storage bin | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 34 s | 16/16 |
| heldout | off | easy | Pick up the tennis ball and drop it into the fruit basket | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 36 s | 16/16 |
| heldout | off | medium | Place the apple to the right of the soda can | 1 | 0/1 | 0/1 | 0.0 | $0.001 | 12 s | - |
| heldout | off | medium | Put the red block in front of the blue block | 1 | 1/1 | 0/1 | 2.0 | $0.012 | 58 s | 14/16 |
| heldout | off | medium | Put the orange and the apple into the fruit basket | 1 | 0/1 | 0/1 | 0.0 | $0.002 | 20 s | - |
| heldout | off | medium | Build a tower of two Rubik's cubes | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 36 s | 16/16 |
| heldout | off | medium | Put the wooden block behind the soda can | 1 | 1/1 | 1/1 | 1.0 | $0.005 | 39 s | 16/16 |
| heldout | off | hard | Stack the green block on the Rubik's cube, then put the lime to the left of the Rubik's cube | 1 | 1/1 | 1/1 | 1.0 | $0.006 | 53 s | 16/16 |
| heldout | off | hard | Sort the toys: the red block goes into the red bin and the blue block into the blue bin | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 45 s | 16/16 |
| heldout | off | beyond | Open the drawer | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 36 s | 16/16 |
| heldout | off | beyond | Put the tennis ball into the drawer and close it | 1 | 1/1 | 1/1 | 1.0 | $0.006 | 46 s | 16/16 |
| heldout | off | beyond | Knock the soda can over | 1 | 1/1 | 0/1 | 2.0 | $0.008 | 33 s | 9/16 |
| heldout | off | refuse | Put the basketball into the storage bin | 1 | 1/1 | 1/1 | 0.0 | $0.001 | 17 s | - |

## Round 1 (Oct 28, 2026, commit `086388b`, previous pipeline with Ultra escalation)

**Setup.** Two suites, all 104 runs on one commit with no code changes during the run
(`eval/20261028-210032/`, which also stores the commit hash):
- **Development suite:** the 11 prompts we built the system against (see the history below). These are not a fair test on their own.
- **Held-out suite:** 15 prompts written and committed *before* any of them was run, and never used to change code. They mix known objects in new ways and include one task that should be refused (a basketball is wider than the gripper).
- Each prompt was run 3 times with the full pipeline, and once more with the vision Critic turned off (repairs then see only the measured failures).
- Tavily results are cached by query, as in production. Uncached searches: 101 over the 104 runs. Repeat runs therefore measure the models' variation more than the web's.

**Results (Critic on, 3 runs per prompt):**

| Suite | Success | First try | Model cost per task | Stress test (solved tasks) |
|---|---|---|---|---|
| Held-out, 15 prompts | **41/45 (91 %, 95 % CI 79–96 %)** | 30/45 (67 %, CI 52–79 %) | mean $0.010, median $0.005 | 555/608 re-runs pass (91 %) |
| Development, 11 prompts | 29/33 (88 %, CI 73–95 %) | 23/33 (70 %, CI 53–83 %) | mean $0.017, median $0.006 | 411/448 re-runs pass (92 %) |
| **All** | **70/78 (90 %, CI 81–95 %)** | 53/78 (68 %, CI 57–77 %) | mean $0.013, median $0.006 | 966/1056 (91 %) |

The held-out prompts did as well as the development prompts, so the success rate is not just from
tuning on the development suite. Median wall time per task: 47 s. Total model cost of the 104 runs: $1.22.

**What fails (all 8 failures with the Critic on):**
- *Knock the soda can over* (held-out): 1/3. Of the failed runs (two with the Critic, one without), two never tipped the can: the simulator reports the arm stopping short of its push target, and repairs didn't fix it. In the other, the Planner listed the table as an object to place, and no build could satisfy that (table objects are now dropped in code).
- *Put the orange and the apple into the fruit basket* (held-out): 1/3. Once the orange (8.4 cm) and once the apple (8.9 cm) came back wider than the gripper, so the run was refused before building. These were false refusals from single outlier web values (fixed later, see round 2).
- *Put the soda can to the left of the Rubik's cube* (dev): 1/3. Once the web size swapped the can's height and width (a false refusal), once 4 attempts didn't place it.
- *Fruit basket + recycling bin* (dev): 2/3. The hardest task. One run failed after 4 attempts ($0.23, the most expensive run), and the solved runs score only 5–6/16 on the stress test: the soda can barely stays upright.
- *Knock over the tower* (dev): 2/3. Once the plan swept the wrong block.

**Does the vision Critic help? Not measurably, at this sample size.** With the Critic off, the
same 26 prompts succeeded 23/26 times (88 %, CI 71–96 %), against 70/78 (90 %) with it on. Few runs
need a repair at all (22 with the Critic, 6 without), so this comparison cannot separate the two. The
Critic is skipped automatically when nothing was simulated (no frames to watch), which happened 23
times. Reporting this as it is: the Critic's value is its readable diagnosis in the UI, and we have
not shown it makes repairs succeed more often.

**Does escalating to Ultra pay off? Not in this eval.** Super's repairs fixed the run 18 of 28 times
(64 %). Ultra's escalated repairs fixed it 4 of 18 times (22 %; 4 of the 10 runs escalated to it), and Ultra only runs on the tasks
that already beat Super twice, so these are the hardest cases. The data argues for trying more Super
repairs before Ultra, or dropping Ultra.

**Stress test.** Each solved plan is re-run 16 times, physics only: every loose object starts up to
4 cm away and turned up to 30°, weighs ±20 %, shares one ±30 % friction scale with the other objects, and the robot sees it up to 0.5 cm
from where it really is. Most solved tasks score 14–16/16. Fragile plans show up clearly: fruit
sorting 5–6/16, *lime on the red block* 5/16 once, *knock the soda can over* 7/16. One solved run
scored 0/0: every perturbed variant overlapped (crowded scene), so it has no score.

**Limits of this eval.** 3 runs per prompt is small; the confidence intervals above say how small.
The simulator's grasp assist (see the README) makes grasping easier than on a real robot, so these
rates are for "the plan works in this simulator", not for a real Franka.

### Round 1 per-prompt results

| Suite | Critic | Difficulty | Prompt | Runs | Success | First try | Avg attempts | Avg cost | Avg time | Stress (passed/trials) |
|---|---|---|---|---|---|---|---|---|---|---|
| dev | on | easy | A Franka arm stacks the red cube on the blue cube | 3 | 3/3 | 3/3 | 1.0 | $0.004 | 40 s | 16/16, 14/16, 16/16 |
| dev | on | easy | Put the Rubik's cube on top of the soda can | 3 | 3/3 | 3/3 | 1.0 | $0.005 | 41 s | 16/16, 16/16, 16/16 |
| dev | on | medium | Pick up the tennis ball and place it on the Rubik's cube | 3 | 3/3 | 3/3 | 1.0 | $0.004 | 38 s | 15/16, 15/16, 15/16 |
| dev | on | medium | Stack three wooden toy blocks into a tower: red at the bottom, then green, then blue on top | 3 | 3/3 | 3/3 | 1.0 | $0.006 | 58 s | 16/16, 13/16, 16/16 |
| dev | on | medium | Clear the table: put the red, green and blue toy blocks into the storage bin | 3 | 3/3 | 2/3 | 1.3 | $0.008 | 68 s | 16/16, 16/16, 14/16 |
| dev | on | hard | Put the apple and the lime into the fruit basket, and the soda can into the recycling bin | 3 | 2/3 | 0/3 | 2.7 | $0.089 | 177 s | 6/16, 5/16 |
| dev | on | hard | Tidy up: stack the red block on the blue block, then put the green block and the tennis ball into the tray | 3 | 3/3 | 2/3 | 1.3 | $0.015 | 115 s | 14/16, 15/16, 0/0 |
| dev | on | hard | Put the soda can to the left of the Rubik's cube and keep it upright | 3 | 1/3 | 1/3 | 2.5 | $0.022 | 55 s | 16/16 |
| dev | on | beyond | Open the drawer, put the Rubik's cube inside, and close it | 3 | 3/3 | 3/3 | 1.0 | $0.008 | 63 s | 16/16, 16/16, 16/16 |
| dev | on | beyond | Put the tennis ball and the Rubik's cube into the drawer and close it | 3 | 3/3 | 1/3 | 1.7 | $0.014 | 103 s | 16/16, 15/16, 15/16 |
| dev | on | beyond | Knock over the tower of three wooden toy blocks | 3 | 2/3 | 2/3 | 2.0 | $0.013 | 54 s | 16/16, 15/16 |
| heldout | on | easy | Put the lime on the red toy block | 3 | 3/3 | 2/3 | 1.7 | $0.018 | 51 s | 10/16, 5/16, 16/16 |
| heldout | on | easy | Stack the blue block on the green block | 3 | 3/3 | 2/3 | 1.3 | $0.004 | 33 s | 16/16, 16/16, 16/16 |
| heldout | on | easy | Put the Rubik's cube into the storage bin | 3 | 3/3 | 1/3 | 1.7 | $0.008 | 50 s | 16/16, 16/16, 16/16 |
| heldout | on | easy | Pick up the tennis ball and drop it into the fruit basket | 3 | 3/3 | 3/3 | 1.0 | $0.004 | 31 s | 16/16, 16/16, 16/16 |
| heldout | on | medium | Place the apple to the right of the soda can | 3 | 3/3 | 3/3 | 1.0 | $0.005 | 39 s | 16/16, 16/16, 15/16 |
| heldout | on | medium | Put the red block in front of the blue block | 3 | 3/3 | 0/3 | 2.3 | $0.013 | 58 s | 12/16, 13/16, 14/16 |
| heldout | on | medium | Put the orange and the apple into the fruit basket | 3 | 1/3 | 1/3 | 1.0 | $0.003 | 34 s | 16/16 |
| heldout | on | medium | Build a tower of two Rubik's cubes | 3 | 3/3 | 3/3 | 1.0 | $0.005 | 36 s | 16/16, 15/16, 16/16 |
| heldout | on | medium | Put the wooden block behind the soda can | 3 | 3/3 | 2/3 | 2.0 | $0.023 | 57 s | 15/16, 16/16, 4/16 |
| heldout | on | hard | Stack the green block on the Rubik's cube, then put the lime to the left of the Rubik's cube | 3 | 3/3 | 3/3 | 1.0 | $0.006 | 44 s | 16/16, 16/16, 14/16 |
| heldout | on | hard | Sort the toys: the red block goes into the red bin and the blue block into the blue bin | 3 | 3/3 | 3/3 | 1.0 | $0.008 | 65 s | 16/16, 16/16, 16/16 |
| heldout | on | beyond | Open the drawer | 3 | 3/3 | 3/3 | 1.0 | $0.004 | 41 s | 16/16, 16/16, 16/16 |
| heldout | on | beyond | Put the tennis ball into the drawer and close it | 3 | 3/3 | 1/3 | 1.7 | $0.011 | 79 s | 15/16, 16/16, 16/16 |
| heldout | on | beyond | Knock the soda can over | 3 | 1/3 | 0/3 | 4.0 | $0.039 | 90 s | 7/16 |
| heldout | on | refuse | Put the basketball into the storage bin | 3 | 3/3 | 3/3 | 0.0 | $0.001 | 23 s | - |
| dev | off | easy | A Franka arm stacks the red cube on the blue cube | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 27 s | 16/16 |
| dev | off | easy | Put the Rubik's cube on top of the soda can | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 31 s | 16/16 |
| dev | off | medium | Pick up the tennis ball and place it on the Rubik's cube | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 27 s | 14/16 |
| dev | off | medium | Stack three wooden toy blocks into a tower: red at the bottom, then green, then blue on top | 1 | 1/1 | 1/1 | 1.0 | $0.005 | 44 s | 15/16 |
| dev | off | medium | Clear the table: put the red, green and blue toy blocks into the storage bin | 1 | 1/1 | 1/1 | 1.0 | $0.007 | 61 s | 16/16 |
| dev | off | hard | Put the apple and the lime into the fruit basket, and the soda can into the recycling bin | 1 | 0/1 | 0/1 | 0.0 | $0.003 | 17 s | - |
| dev | off | hard | Tidy up: stack the red block on the blue block, then put the green block and the tennis ball into the tray | 1 | 1/1 | 1/1 | 1.0 | $0.010 | 70 s | 14/16 |
| dev | off | hard | Put the soda can to the left of the Rubik's cube and keep it upright | 1 | 0/1 | 0/1 | 0.0 | $0.001 | 9 s | - |
| dev | off | beyond | Open the drawer, put the Rubik's cube inside, and close it | 1 | 1/1 | 1/1 | 1.0 | $0.007 | 51 s | 16/16 |
| dev | off | beyond | Put the tennis ball and the Rubik's cube into the drawer and close it | 1 | 1/1 | 1/1 | 1.0 | $0.010 | 87 s | 16/16 |
| dev | off | beyond | Knock over the tower of three wooden toy blocks | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 33 s | 15/16 |
| heldout | off | easy | Put the lime on the red toy block | 1 | 1/1 | 0/1 | 2.0 | $0.014 | 95 s | 15/16 |
| heldout | off | easy | Stack the blue block on the green block | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 34 s | 16/16 |
| heldout | off | easy | Put the Rubik's cube into the storage bin | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 31 s | 16/16 |
| heldout | off | easy | Pick up the tennis ball and drop it into the fruit basket | 1 | 1/1 | 1/1 | 1.0 | $0.005 | 38 s | 16/16 |
| heldout | off | medium | Place the apple to the right of the soda can | 1 | 1/1 | 0/1 | 2.0 | $0.008 | 51 s | 16/16 |
| heldout | off | medium | Put the red block in front of the blue block | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 47 s | 12/16 |
| heldout | off | medium | Put the orange and the apple into the fruit basket | 1 | 1/1 | 1/1 | 1.0 | $0.005 | 41 s | 16/16 |
| heldout | off | medium | Build a tower of two Rubik's cubes | 1 | 1/1 | 1/1 | 1.0 | $0.005 | 41 s | 16/16 |
| heldout | off | medium | Put the wooden block behind the soda can | 1 | 1/1 | 0/1 | 2.0 | $0.011 | 87 s | 13/16 |
| heldout | off | hard | Stack the green block on the Rubik's cube, then put the lime to the left of the Rubik's cube | 1 | 1/1 | 1/1 | 1.0 | $0.007 | 67 s | 16/16 |
| heldout | off | hard | Sort the toys: the red block goes into the red bin and the blue block into the blue bin | 1 | 1/1 | 0/1 | 2.0 | $0.011 | 84 s | 16/16 |
| heldout | off | beyond | Open the drawer | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 47 s | 16/16 |
| heldout | off | beyond | Put the tennis ball into the drawer and close it | 1 | 1/1 | 0/1 | 2.0 | $0.015 | 108 s | 11/16 |
| heldout | off | beyond | Knock the soda can over | 1 | 0/1 | 0/1 | 4.0 | $0.041 | 84 s | - |
| heldout | off | refuse | Put the basketball into the storage bin | 1 | 1/1 | 1/1 | 0.0 | $0.001 | 19 s | - |

## Development history (Oct 28, 2026, 1 run each, on the development suite)

Each round exposed failures that were then fixed in code. Because the code was changed after seeing these prompts fail, these rounds are not an unbiased measure: see the held-out suite above.

| Round | Success | First try | LLM cost | What the failures taught us |
|---|---|---|---|---|
| 1 (before harder-task guards) | 4/8 | 3/8 | $0.20 | Planner put a goal inside the robot base; IK failed inside the stated workspace; the new gripper-width check caught web sizes that were wrong (a 44 cm "toy block", a soda can with its dimensions scrambled) |
| 2 (hard prompts only) | 2/3 | 2/3 | $0.06 | One scene error found per attempt (bin over the edge, then bin too small, then an invented table) |
| 3 (all fixes) | **8/8** | 3/8 | **$0.17** | First-try misses: invented "table" object (2×), a 10 cm bin for three blocks, the tennis ball rolling off (real physics) |
| 4 (new: drawers, knock-over; 3 prompts) | 2/3 | 1/3 | $0.09 | Builder ignored the cabinet placement rule; a repair copied the prompt's placeholder name `D.handle`; a second object was dropped on top of the first in the drawer |
| 5 (same 3, after fixes) | **3/3** | **3/3** | **$0.02** | Drawer placement moved into code (like bin sizing); real names in the prompt; side-by-side slot offsets |
| 6 (all 11, web-sourced physics) | 9/11 (two crashes) | 7/11 | $0.11 | New: friction/bounciness per material from Tavily, verified quotes, size-vs-mass check. Two tasks crashed on a web size with a 0 dimension (fixed: treated as missing); after the fix only those two were re-run, and both passed first try. This round was not a single clean run, which is why the final evaluation above exists |

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
objects such as a table; bins narrower than 15 cm are enlarged). The next run (round 1 above) uses 3
reps per prompt.

## Caveats
- One rep per prompt is a small sample. The same prompt can pass on the first try in one run and
  need 3 attempts in the next, because model output varies even at low temperature.
- Grasping uses the documented grasp assist (a weld once both finger pads touch an object whose center
  lies between the fingers; see the README). Placement, stacking stability and rolling are physical.
