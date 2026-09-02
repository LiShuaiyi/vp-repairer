# VP temporal full result setup (2026-09-02)

This directory records one result row per `(scenario, ego, repair method)` for
the following three setups:

- `repairer_type=vp, sat_solver_mode=domain_dpll`: VP with predicate-domain
  estimation and DomainDPLL guidance, including the acceleration fallback for
  `R_IN3`, `R_IN4`, and `R_IN5`.
- `repairer_type=vp, sat_solver_mode=dpll`: unguided DPLL with failed partial
  model blocking. The IN results use the same deceleration-then-acceleration
  phase structure, but no predicate domains or unsupported-candidate
  pre-rejection. RG rules do not use the acceleration phase.
- `repairer_type=smt`: the previously recorded SMT repairer results. These rows
  were preserved rather than rerun in this refresh. The attempted planner and
  constraint-mode fallback sequence is stored in
  `attempted_smt_configurations`; the batch policy is `(1,2)`, `(2,2)`,
  `(1,1)`, then `(2,1)` until a strict success is found.

## Code and environment

- Repository branch: `feature/repair-all`
- Base revision: `da120f4` (`Record full VP DPLL and SMT batch results`)
- Conda environment: `/data_linux/conda-envs/repairverse310_gpu`
- Batch runner: `examples/batch_test_vp_repairer_all_rules_updated.py`
- Execution: serial (`--max-workers 1`), per-case timeout 300 seconds
- Acceleration reference path: enabled (the runner default)
- Strict VP success: the repaired trajectory is accepted only when the updated
  traffic-rule violation time is positive infinity.
- `core_total_time` excludes the post-repair STL monitor/compliance check.

The plain-DPLL IN refresh was generated from the work tree after revision
`da120f4`. It enables `_supports_acceleration_fallback` for DPLL, so an
exhausted deceleration search starts a fresh acceleration SAT phase from the
original CNF. The work tree also aligns implication anchors and semantic
conflict geometry between DPLL backends, treats only the ego-direction
`in_intersection_conflict_area__0_1` predicate as VP-controllable, and
permissively ignores unsupported extra literals in the unguided DPLL model.
Every returned trajectory still has to pass the complete STL monitor check.

## Commands and source result directories

DomainDPLL (all rule groups):

```bash
PYTHONPATH=.. PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl \
/data_linux/conda-envs/repairverse310_gpu/bin/python \
batch_test_vp_repairer_all_rules_updated.py \
--groups rg1,rg1_mona,rg2,rg3,rg1_rg3,in1,in3,in4,in5 \
--repairers vp --vp-sat-solver-mode domain_dpll \
--output-dir /tmp/vp_domain_dpll_serial_full_20260902 \
--max-workers 1 --timeout 300
```

Plain-DPLL source batch (all groups were run before enabling the experimental
acceleration fallback; the formal CSVs retain its RG rows):

```bash
PYTHONPATH=.. PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl \
/data_linux/conda-envs/repairverse310_gpu/bin/python \
batch_test_vp_repairer_all_rules_updated.py \
--groups rg1,rg1_mona,rg2,rg3,rg1_rg3,in1,in3,in4,in5 \
--repairers vp --vp-sat-solver-mode dpll \
--output-dir /tmp/vp_plain_dpll_serial_full_20260902 \
--max-workers 1 --timeout 300
```

Plain DPLL with acceleration fallback and aligned IN semantics:

```bash
PYTHONPATH=.. PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
MPLCONFIGDIR=/tmp/mpl \
/data_linux/conda-envs/repairverse310_gpu/bin/python \
batch_test_vp_repairer_all_rules_updated.py \
--groups in1,in3,in4,in5 \
--repairers vp --vp-sat-solver-mode dpll \
--output-dir /tmp/vp_plain_dpll_semantic_aligned_serial_in_20260902_v2 \
--max-workers 1 --timeout 300
```

Current IN3 hand-draft VP refresh (49 finite-TV cases after filtering 36 cases
whose current monitor value is positive infinity):

```bash
PYTHONPATH=.. PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
MPLCONFIGDIR=/tmp/mpl \
/data_linux/conda-envs/repairverse310_gpu/bin/python \
batch_test_vp_repairer_all_rules_updated.py \
--groups in3 --repairers vp --in3-rule-variant hand_draft \
--vp-sat-solver-mode domain_dpll \
--max-workers 1 --timeout 300
```

The plain-DPLL run uses the same command with
`--vp-sat-solver-mode dpll`.

The actual runs used a prefiltered 49-case manifest so the 36 known
positive-infinity cases did not repeat the per-case monitor setup. This only
changes the input selection; all repair configurations and internal timers are
the same as in the unified runner.

The formal CSVs were assembled as `DomainDPLL VP`, `plain-DPLL VP`, then `SMT`
for every input row. RG1 paper-level summaries combine the 100 highD `rg1`
rows and the 93 `rg1_mona` rows.

## Result caveats

The refreshed IN plain-DPLL rows use `PYTHONHASHSEED=0`, making their Python-set
branch order reproducible. The retained RG plain-DPLL rows came from the older
all-rule batch above, where `PYTHONHASHSEED` was not fixed. Use the explicit
`successful_repair_mode`, `deceleration_iterations`, and
`acceleration_iterations` columns when attributing an outcome to acceleration.

The main `vp_repairer_in3_batch_result_updated.csv` is the current
`R_IN3_hand_draft` comparison. It contains 49 cases per method: freshly run
serial DomainDPLL VP and plain-DPLL VP rows, plus the matching 49 rows selected
from the previously recorded hand-draft SMT batch. The 36 cases for which the
current monitor returns `tv=inf` are excluded from all three methods.

The separate `vp_repairer_in3_full_batch_result_updated.csv` preserves the
complete-`R_IN3` experiment. It retains all 85 input rows per method, including
36 explicit positive-infinity skip rows. Its VP timings are serial; its SMT run
was a parallel success-rate probe, so those SMT timing fields must not be used
as paper timing measurements.

The direction-aligned DomainDPLL work tree was additionally rerun serially in
`/tmp/vp_domain_dpll_direction_aligned_serial_in_20260902`. Its success,
iteration, phase, and error fields matched the retained DomainDPLL rows for
every IN case, so the existing formal DomainDPLL rows were not replaced merely
for timing noise.
