# MICP rule-fidelity audit (2026-09-04)

Scope: compare the isolated free-longitudinal/free-lateral MICP adapter with
`comparison/micp/traffic_rule.py`, `comparison/micp/traffic_rule_4d.py`, and
the formulas in `commonroad-stl-monitor/crmonitor/traffic_rules_rtamt.yaml`.

## Confirmed mismatches and fixes

- `R_G1` previously quantified every scene vehicle, used a 6-piece/two-sign
  safe-distance disjunction, enlarged same-lane bounds, and added the `R_G3`
  speed envelope. Lin's implementation uses one monitor-selected vehicle and
  ten conjunctive safe-distance tangents. The isolated implementation now has
  that structure; the speed envelope is only composed for `R_G1_R_G3`.
- `R_G2` previously used a three-sample monitor-trigger window, which was an
  extra simplification. It now uses Lin2025's released formula over the complete
  horizon, including `not_braking`, front/same-lane complements, and the
  safe-distance/relative-braking conjunction.
- `R_G3` previously took the minimum speed limit across every lanelet that
  projected into the CLCS and used a four-sided L1 diamond. It now uses the
  lane assignment at each sample and a rotated 16-sided inner approximation of
  Euclidean speed, with no binary variables.
- `R_IN1` reproduces Lin's one-metre stop-zone temporal structure and its three
  released `phantom_false` branches verbatim.
- `R_IN3/R_IN4/R_IN5` now reproduce Lin's geometric intersection-overlap box,
  the full-horizon constraint, and all 27 released auxiliary predicates (nine
  groups of three). Removing those auxiliaries was algebraically tempting but
  materially changed standard-MICP model size and timing.
- The dynamic model is the full Lin2025 eight-state longitudinal/lateral
  position--velocity--acceleration--jerk chain. The former four-state direct
  acceleration model was an extra simplification and is no longer used.
- Candidate validation now invalidates predicate caches of every vehicle at
  each replaced time step. Clearing only the ego cache was unsound for a
  predicate such as `in_intersection_conflict_area(a1,a0)`.
- Intersection constraints now respect the quantified vehicle's lifetime.
  Predicate lookup after the target's final state previously activated a
  phantom conflict vehicle and made valid tail states infeasible.

## Strict Lin2025 standard-encoding full regression

The four rule workers ran in parallel with one Gurobi thread per worker and a
30 s per-case limit. Times are specification construction + solver setup +
solve time; monitor validation and XML loading are separate. `standard` uses
stlpy's original node-wise binary encoding. No monitor-trigger time window,
reduced dynamics, or collapsed auxiliary formula is used.

| Rule | Success | Rate | Mean (s) | Median (s) | P95 (s) |
|---|---:|---:|---:|---:|---:|
| R_G1 | 98/100 | 98.0% | 0.852 | 0.877 | 0.936 |
| R_G1+R_G3 | 84/87 | 96.6% | 1.650 | 1.688 | 1.809 |
| R_G2 | 16/69 | 23.2% | 0.807 | 1.149 | 1.478 |
| R_G3 | 100/100 | 100.0% | 0.848 | 0.861 | 0.908 |
| R_IN1 | 0/16 | 0.0% | 0.431 | 0.430 | 0.503 |
| R_IN3 | 42/85 | 49.4% | 1.344 | 1.253 | 1.560 |
| R_IN4 | 10/24 | 41.7% | 1.393 | 1.388 | 1.612 |
| R_IN5 | 12/29 | 41.4% | 1.381 | 1.386 | 1.496 |
| **All** | **362/510** | **71.0%** | **1.103** | **1.112** | -- |

The low R_G2/intersection success rate is a characteristic of validating Lin's
released MICP approximation with the current VP monitor. Four rows report data
or validation exceptions; they remain failures rather than being hidden by a
constraint relaxation.

Six generated R_IN5 variants with a 150-step horizon are excluded from this
statistical cohort so that it remains comparable with the standard 20-step IN
cases. Their original rows are retained separately in
`results/excluded_in5_150step_generated_variants_2026-09-04.csv`.

Raw highD results are in `results/lin2025_standard_full_2026-09-04/`. The final
strict intersection results are in `results/lin2025_exact_in_full_2026-09-04/`.
The earlier `lin_balanced_*` results used reduced dynamics/local windows and
must not be cited as the Lin2025 baseline.

## License

The working unrestricted-size license is the academic WLS file under
`autoware-repair-docker/gurobi.lic`; it requires token-server access. The
license under `repair-autoware/lib/gurobi.lic` is restricted-size and rejects
the real regression models.
