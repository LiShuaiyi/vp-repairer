# VP repair vs. 2-D sampling replan

Strict success means the returned trajectory is collision/kinematics feasible and the independent full STL monitor reports `updated_tv = +inf`.

Planning time excludes scenario/monitor setup and the redundant final validation. Sampling time includes rule checks performed while searching candidates.

Candidate-check budget per case/planner invocation: 0. Runs used parallel rule cohorts; each planner invocation itself is single-process.

| Cohort | Rule | N | VP success | Sampling success | VP mean (ms) | Sampling mean (ms) | Sampling/VP |
|---|---:|---:|---:|---:|---:|---:|---:|
| rg1 | R_G1 | 100 | 100.0% | 100.0% | 11.79 | 272.12 | 23.0x |
| rg2 | R_G2 | 69 | 100.0% | 98.6% | 13.73 | 196.10 | 18.9x |
| rg3 | R_G3 | 100 | 100.0% | 100.0% | 7.88 | 241.21 | 30.8x |
| rg1_rg3 | R_G1_R_G3 | 87 | 100.0% | 86.2% | 12.45 | 2255.49 | 191.7x |
| in1 | R_IN1 | 91 | 100.0% | 44.0% | 13.47 | 801.48 | 62.7x |
| in3 | R_IN3_hand_draft | 68 | 98.5% | 52.9% | 20.66 | 1664.97 | 100.0x |
| in4 | R_IN4 | 49 | 98.0% | 38.8% | 16.34 | 1459.90 | 96.1x |
| in5 | R_IN5 | 50 | 98.0% | 48.0% | 18.35 | 635.76 | 44.0x |

Overall matched cases: 614; VP success 99.5%; sampling success 75.2%.
Overall core-time mean (available timings): VP 13.59 ms (614 cases), sampling 888.84 ms (599 cases), ratio of means 65.4x.
Sampling failures that exhausted the explicit candidate budget: 0. Error categories (counted as method failures): {'none': 599, 'Exception': 15}.
