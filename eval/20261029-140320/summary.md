| Suite | Critic | Difficulty | Prompt | Runs | Success | First try | Avg attempts | Avg cost | Avg time | Stress (passed/trials) |
|---|---|---|---|---|---|---|---|---|---|---|
| dev | on | easy | A Franka arm stacks the red cube on the blue cube | 1 | 1/1 | 0/1 | 2.0 | $0.005 | 28 s | 16/16 |
| dev | on | easy | Put the Rubik's cube on top of the soda can | 1 | 1/1 | 0/1 | 2.0 | $0.005 | 29 s | 16/16 |
| dev | on | medium | Pick up the tennis ball and place it on the Rubik's cube | 1 | 1/1 | 1/1 | 1.0 | $0.002 | 23 s | 16/16 |
| dev | on | medium | Stack three wooden toy blocks into a tower: red at the bottom, then green, then blue on top | 1 | 1/1 | 0/1 | 2.0 | $0.006 | 44 s | 16/16 |
| dev | on | medium | Clear the table: put the red, green and blue toy blocks into the storage bin | 1 | 0/1 | 0/1 | 4.0 | $0.024 | 43 s | - |
| dev | on | hard | Put the apple and the lime into the fruit basket, and the soda can into the recycling bin | 1 | 0/1 | 0/1 | 0.0 | $0.001 | 6 s | - |
| dev | on | hard | Tidy up: stack the red block on the blue block, then put the green block and the tennis ball into the tray | 1 | 1/1 | 0/1 | 2.0 | $0.011 | 63 s | 13/16 |
| dev | on | hard | Put the soda can to the left of the Rubik's cube and keep it upright | 1 | 1/1 | 0/1 | 4.0 | $0.018 | 45 s | 9/16 |
| dev | on | beyond | Open the drawer, put the Rubik's cube inside, and close it | 1 | 0/1 | 0/1 | 4.0 | $0.027 | 51 s | - |
| dev | on | beyond | Put the tennis ball and the Rubik's cube into the drawer and close it | 1 | 0/1 | 0/1 | 4.0 | $0.031 | 49 s | - |
| dev | on | beyond | Knock over the tower of three wooden toy blocks | 1 | 0/1 | 0/1 | 4.0 | $0.018 | 40 s | - |

- **dev, Critic on:** success 6/11 (55 %, 95 % CI 28-79 %); first try 1/11 (9 %, 95 % CI 2-38 %); model cost mean $0.013, median $0.011, total $0.15; uncached Tavily searches 7; stress test 86/96 perturbed re-runs passed

- **Vision scene check vs attempt 1:** flagged and attempt 1 failed 2; flagged but it passed 0; said ok but it failed 4; said ok and it passed 1 (of 7)
