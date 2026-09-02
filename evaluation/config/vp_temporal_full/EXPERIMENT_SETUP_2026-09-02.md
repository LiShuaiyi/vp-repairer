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
- Base revision: `535b5e2` (`Optimize acceleration VP planning and update results`)
- Conda environment: `/data_linux/conda-envs/repairverse310_gpu`
- Batch runner: `examples/batch_test_vp_repairer_all_rules_updated.py`
- Execution: serial (`--max-workers 1`), per-case timeout 300 seconds
- Acceleration reference path: enabled (the runner default)
- Strict VP success: the repaired trajectory is accepted only when the updated
  traffic-rule violation time is positive infinity.
- `core_total_time` excludes the post-repair STL monitor/compliance check.

The plain-DPLL IN refresh additionally enabled `_supports_acceleration_fallback`
for DPLL, so an exhausted deceleration search starts a fresh acceleration SAT
phase from the original CNF. This experimental change was present in the work
tree when the IN plain-DPLL rows were generated; it was not part of base
revision `535b5e2`.

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

Plain DPLL with acceleration fallback for IN groups:

```bash
PYTHONPATH=.. PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mpl \
/data_linux/conda-envs/repairverse310_gpu/bin/python \
batch_test_vp_repairer_all_rules_updated.py \
--groups in1,in3,in4,in5 \
--repairers vp --vp-sat-solver-mode dpll \
--output-dir /tmp/vp_plain_dpll_acceleration_serial_in_20260902 \
--max-workers 1 --timeout 300
```

The formal CSVs were assembled as `DomainDPLL VP`, `plain-DPLL VP`, then `SMT`
for every input row. RG1 paper-level summaries combine the 100 highD `rg1`
rows and the 93 `rg1_mona` rows.

## Result caveat

`PYTHONHASHSEED` was not fixed for this batch. Plain DPLL uses Python sets while
constructing and traversing partial assignments, so its branch/model order can
vary between isolated subprocesses. In this refresh, one extra IN3 case and one
extra IN4 case succeeded in the deceleration phase relative to the immediately
preceding plain-DPLL run; neither was an acceleration success. Use the explicit
`successful_repair_mode`, `deceleration_iterations`, and
`acceleration_iterations` columns when attributing an outcome to acceleration.
