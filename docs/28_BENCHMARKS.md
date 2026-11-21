# Benchmarks (exploratory)

`swarm benchmark` runs each case through the normal runner and records its own environment.
Timings are wall clock on one machine. They are **not validated results**, and they are not
comparable across machines.

## Method
- **Seeds:** development seeds 1000–1004, with 3 repetitions per case.
- **Timing:** `time.perf_counter` around whole episodes. The report takes the median over
  repetitions of the total time for all seeds, then divides by the total steps.
- **Cases** vary one factor from the `basic` example (`auction_comm` unless stated):
  - extension overhead;
  - robots per type;
  - map side, with survivors scaled to density and time limit 10 × side;
  - time limit at recall target 1.0.
- **Reproduce:**
  `swarm benchmark --seeds 1000-1004 --repetitions 3 --out b.json --markdown b.md`

## Results for v0.1.0
Run at commit `ab99ba2` (clean tree): Python 3.13.18, NumPy 2.5.4, Linux x86_64, 2 CPUs (shared
cloud VM). Raw data: [`results/benchmarks/v0.1.0-benchmark.json`](results/benchmarks/v0.1.0-benchmark.json).

| Group | Case | Robots | Cells | Steps (5 episodes) | ms/step | Episodes/s |
|---|---|---|---|---|---|---|
| overhead | frontier, no channel | 4 | 400 | 470 | 2.58 | 4.1 |
| overhead | frontier_shared (belief sharing) | 4 | 400 | 235 | 4.08 | 5.2 |
| overhead | auction_comm | 4 | 400 | 180 | 6.75 | 4.1 |
| overhead | auction_comm + perception | 4 | 400 | 183 | 7.46 | 3.7 |
| overhead | auction_comm + failures | 4 | 400 | 360 | 3.68 | 3.8 |
| overhead | learned:mappo_s0 | 4 | 400 | 220 | 4.61 | 4.9 |
| robots | 1+1 robots | 2 | 400 | 387 | 2.81 | 4.6 |
| robots | 2+2 robots | 4 | 400 | 180 | 6.48 | 4.3 |
| robots | 4+4 robots | 8 | 400 | 106 | 15.39 | 3.1 |
| robots | 8+8 robots | 16 | 400 | 96 | 44.94 | 1.2 |
| map | 20x20 map | 4 | 400 | 180 | 6.33 | 4.4 |
| map | 30x30 map | 4 | 900 | 930 | 9.08 | 0.6 |
| map | 40x40 map | 4 | 1600 | 1779 | 12.70 | 0.2 |
| duration | time limit 50 | 4 | 400 | 250 | 6.21 | 3.2 |
| duration | time limit 100 | 4 | 400 | 396 | 7.25 | 1.7 |
| duration | time limit 200 | 4 | 400 | 402 | 7.39 | 1.7 |

## Reading it
- **Per-step cost is set by the policies, not the environment.**
  - Belief sharing costs about 1.6× independent frontier per step, and the auction about 2.6×.
  - Episodes are shorter with coordination, so episodes per second stay similar (4–5).
- **The perception overhead (about 10% per step) is within measurement noise.** On this shared VM,
  repetitions of the auction cases varied by up to 37% (`seconds_all` in the JSON).
- **Failures lower the cost per step** (fewer active robots) but double episode length.
- **Team size is the main scaling limit.** Per-step cost grows superlinearly: 2.8 ms (1+1),
  6.5 ms (2+2), 15.4 ms (4+4) and 44.9 ms (8+8). This matches docs/20.
- **Map size grows sublinearly per step** (6.3 → 12.7 ms from 400 to 1 600 cells). Episodes get
  much longer, though.
- **Bottleneck.** A `cProfile` run of `auction_comm` with 8+8 robots (3 episodes, 8.9 s
  profiled) puts the most self-time (1.9 s) in `BeliefMap.merge_records`. Including its callees,
  it accounts for 4.3 s, 48% of runtime. It runs once per delivered message, and deliveries grow
  with the square of team size.
  - I tried inlining its per-record type checks. The event streams stayed identical, but there was
    no measurable speedup (47.0 vs 51.5 ms/step, within noise), so it was **reverted**.
  - A real gain probably needs batching or vectorizing merges. That touches validated coordination
    code and is left for future work.

The 3D analysis (`analyze_scene`) costs about 0.01–0.02 ms per step, which is negligible
(docs/24, ADR-0024).
