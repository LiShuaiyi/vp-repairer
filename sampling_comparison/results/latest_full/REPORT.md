# VP repair vs. 2-D sampling replan

Strict success means the returned trajectory is collision/kinematics feasible and the independent full STL monitor reports `updated_tv = +inf`.

Planning time excludes scenario/monitor setup and the redundant final validation. Sampling time includes rule checks performed while searching candidates.

Candidate-check budget per case/planner invocation: 0. Runs used parallel rule cohorts; each planner invocation itself is single-process.

| Cohort | Rule | N | VP success | Sampling success | VP mean (ms) | Sampling mean (ms) | Sampling/VP |
|---|---:|---:|---:|---:|---:|---:|---:|
| rg1 | R_G1 | 100 | 100.0% | 100.0% | 9.49 | 272.12 | 29.9x |
| rg2 | R_G2 | 69 | 95.7% | 98.6% | 6.27 | 196.10 | 34.3x |
| rg3 | R_G3 | 100 | 100.0% | 100.0% | 9.38 | 241.21 | 25.8x |
| rg1_rg3 | R_G1_R_G3 | 87 | 100.0% | 86.2% | 10.53 | 2255.49 | 220.3x |
| in1 | R_IN1 | 91 | 98.9% | 42.9% | 8.78 | 828.35 | 103.3x |
| in3 | R_IN3_hand_draft | 68 | 100.0% | 52.9% | 19.57 | 1703.28 | 137.4x |
| in4 | R_IN4 | 46 | 95.7% | 39.1% | 18.37 | 1409.86 | 114.3x |
| in5 | R_IN5 | 50 | 98.0% | 48.0% | 17.66 | 635.76 | 48.8x |

Overall matched cases: 611; VP success 98.9%; sampling success 75.3%.
Overall core-time mean (available timings): VP 11.61 ms (611 cases), sampling 889.55 ms (595 cases), ratio of means 76.6x.
Sampling failures that exhausted the explicit candidate budget: 0. Error categories (counted as method failures): {'none': 595, 'Exception': 16}.
