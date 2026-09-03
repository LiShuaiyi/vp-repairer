# Halder–Althoff minimum-violation velocity-planning baseline

This directory contains two isolated implementations related to:

> Halder and Althoff (2022), *Minimum-Violation Velocity Planning with
> Temporal Logic Constraints*.

It does not modify `crrepairer` or the authors' checkout. `planner.py` is an
early diagnostic reimplementation and must not be used for paper runtime
comparisons. `author_reference_planner.py` directly executes the search,
vehicle model, node and priority-queue code from the supplied author checkout,
with a project-rule facade.

The versioned result is `halder_full_timing_results_2026-09-03.csv`. It
contains 719 `row_type=case` rows and 10 `row_type=rule_summary` rows,
including single rules, combined R_G1+R_G3, and the 93 MONA R_G1 cases. MONA
cases use the `R_G1_MONA` label. Generated per-case JSON, logs, manifests, and
intermediate reports remain local and are intentionally ignored by Git.

## Rule mapping

| project rule | lattice robustness used here | fidelity |
|---|---|---|
| `R_G1` | leading-vehicle gap minus braking-based safe distance | direct fixed-path encoding |
| `R_G2` | abrupt braking implies an unsafe lead or comparable lead braking | direct fixed-path encoding |
| `R_G3` | minimum of the four time/path-dependent speed limits minus ego speed | direct fixed-path encoding |
| `R_IN1` | stop-line crossing implies a completed standstill interval behind the line | direct past-time encoding |
| `R_IN3`, `R_IN4`, `R_IN5` | ego remains outside the conflict interval while the corresponding precomputed priority/conflict signal is active | fixed-path abstraction |

For `R_IN3`/`R_IN4`/`R_IN5`, route geometry, turning relations, priority and
the target vehicles' predicted conflict occupancy are exogenous to velocity
planning.  They must be extracted with the same CommonRoad monitor used by
the main experiment and supplied in the input as `intersection_rules`.  The
planner expands target occupancy by both temporal clauses in the monitor:
an ego-in-conflict look-ahead and a target-in-conflict clearance time.  This
keeps the velocity-search part faithful while making the boundary between
monitor preprocessing and planning measurable.

## Quick start

From the repository root:

```bash
/data_linux/conda-envs/repairverse310_gpu/bin/python \
  -m halder_althoff_comparison.cli \
  halder_althoff_comparison/examples/rg_in_demo.json \
  --output /tmp/halder_althoff_result.json
```

Run the dependency-free tests with:

```bash
python3 -m unittest discover -s halder_althoff_comparison/tests -v
```

For a batch of already extracted cases:

```bash
python3 -m halder_althoff_comparison.batch \
  halder_althoff_comparison/examples/manifest.csv \
  --output /tmp/halder_althoff_batch.csv \
  --result-dir /tmp/halder_althoff_trajectories
```

The input schema is intentionally method-local.  Important fields are:

- `lattice`: `dt`, `ds`, `dv`, horizon, state/input bounds and optional search
  limit. By default, edges must satisfy the vehicle model to numerical
  tolerance; setting `position_tolerance` explicitly enables an approximate
  snapped-position lattice and must be disclosed in results;
- `initial_state`: path position and velocity;
- `rule_order`: highest priority first, exactly as required by the paper's
  totally ordered rulebook;
- `speed_limits`: scalar or per-step lane/type/FOV/braking limits;
- `lead_vehicles`: predicted rear position, velocity, acceleration and
  same-lane mask per step;
- `rule_parameters.R_IN1`: stop-line path position, ego length, standstill
  tolerance and stop duration (or provide one scalar in `stop_lines`);
- `intersection_rules`: one entry per IN rule, containing its path conflict
  interval, priority mask, and target-conflict occupancy mask.

All arrays accept either one scalar (broadcast over the horizon) or exactly
`horizon_steps + 1` samples.  The output contains the state/action sequence,
one accumulated violation integral per ordered rule, runtime, generated and
expanded nodes, and whether the search exhausted its configured limit.
The standalone diagnostic planner requires the initial velocity on its grid.
The author engine does not: its cubic first transition connects the measured
initial state to reachable grid states, matching the released implementation.

The strict adapter augments the released graph node identity from `(t,s,v)` to
`(t,s,v,memory(rule_1),...,memory(rule_n))`. This is generic rather than tied
to IN1: stateless rules contribute `None`, while a temporal rule supplies its
minimal future-equivalence state. In particular, IN1 stores a bounded
consecutive-stop counter plus the latched `once` result, not its raw floating-
point trajectory. The authors' checkout remains read-only; the augmentation
is implemented in `history_aware_search.py` and is enabled by default by the
author adapter.

## Fair comparison protocol

Use the same fixed reference path, initial state, `dt`, horizon, dynamic
obstacle prediction, acceleration bounds, and CommonRoad monitor for VP and
this baseline. Report preprocessing and search time separately, then also
their sum. Compare only paired `(scenario_id, ego_id, rule)` cases. A finite
minimum-violation answer is not automatically successful: after planning,
the implementation re-evaluates the rule for the `tc` and target vehicle
selected before that repair iteration. It deliberately does not replace that
target with a newly selected worst vehicle; such a vehicle belongs to a later
repair iteration. The output records this as `selected_tc_target` validation.

The lattice resolution materially changes runtime and optimality.  Record
`ds`, `dv`, the acceleration set induced by them, node counts, and any search
limit in every experiment.
