# Calibrated run metadata

- Cohort policy: retain the current VP manifests, including the 87 unique
  `R_G1_R_G3` cases; duplicate source rows are counted once by
  `(scenario_id, ego_id, rule)`.
- Python: `/data_linux/conda-envs/repairverse310_gpu/bin/python`
- Sampling planner: vendored Lin et al. (2025) Reactive Planner revision
  `ab96a839` (source version 2024.1).
- Candidate budget: unlimited (`max_rule_candidates=0`).
- R_IN1 strategy: paper batch velocity-keeping configuration.
- Strict success: returned trajectory passes the independent complete STL
  monitor with `updated_tv=+inf`.
- Timing: method core only. Sampling `planning_time` includes candidate rule
  checks; setup and redundant final validation are excluded.
- Execution: rule cohorts and the R_IN1 shards were run in parallel; each
  individual planner invocation remained single-process.
- Total matched unique cases: 611.

Only this calibrated, uncapped full run is retained in the repository.
