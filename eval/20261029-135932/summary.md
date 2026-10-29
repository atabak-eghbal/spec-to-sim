| Suite | Critic | Difficulty | Prompt | Runs | Success | First try | Avg attempts | Avg cost | Avg time | Stress (passed/trials) |
|---|---|---|---|---|---|---|---|---|---|---|
| dev | on | easy | A Franka arm stacks the red cube on the blue cube | 1 | 1/1 | 1/1 | 1.0 | $0.003 | 22 s | 16/16 |
| dev | on | easy | Put the Rubik's cube on top of the soda can | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 33 s | 15/16 |
| dev | on | medium | Pick up the tennis ball and place it on the Rubik's cube | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 30 s | 15/16 |
| dev | on | medium | Stack three wooden toy blocks into a tower: red at the bottom, then green, then blue on top | 1 | 1/1 | 1/1 | 1.0 | $0.004 | 42 s | 16/16 |
| dev | on | medium | Clear the table: put the red, green and blue toy blocks into the storage bin | 1 | 1/1 | 1/1 | 1.0 | $0.005 | 63 s | 16/16 |
| dev | on | hard | Put the apple and the lime into the fruit basket, and the soda can into the recycling bin | 1 | 1/1 | 1/1 | 1.0 | $0.019 | 168 s | 11/16 |
| dev | on | hard | Tidy up: stack the red block on the blue block, then put the green block and the tennis ball into the tray | 1 | 1/1 | 1/1 | 1.0 | $0.009 | 101 s | 13/16 |
| dev | on | hard | Put the soda can to the left of the Rubik's cube and keep it upright | 1 | 1/1 | 1/1 | 1.0 | $0.005 | 52 s | 15/16 |
| dev | on | beyond | Open the drawer, put the Rubik's cube inside, and close it | 1 | 1/1 | 1/1 | 1.0 | $0.008 | 59 s | 16/16 |
| dev | on | beyond | Put the tennis ball and the Rubik's cube into the drawer and close it | 1 | 1/1 | 0/1 | 2.0 | $0.024 | 130 s | 16/16 |
| dev | on | beyond | Knock over the tower of three wooden toy blocks | 1 | 1/1 | 0/1 | 2.0 | $0.011 | 85 s | 14/16 |

- **dev, Critic on:** success 11/11 (100 %, 95 % CI 74-100 %); first try 9/11 (82 %, 95 % CI 52-95 %); model cost mean $0.009, median $0.005, total $0.09; uncached Tavily searches 2; stress test 163/176 perturbed re-runs passed

- **Vision scene check vs attempt 1:** flagged and attempt 1 failed 0; flagged but it passed 0; said ok but it failed 1; said ok and it passed 9 (of 10)
