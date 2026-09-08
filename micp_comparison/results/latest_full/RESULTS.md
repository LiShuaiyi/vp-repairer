# VP-aligned MICP results (2026-09-08)

This result set uses the current CSV files in
`evaluation/config/vp_temporal_full/` as the sole definition of the VP test
population. The unified manifest contains 704 unique `(scenario_id, ego_id,
rule)` cases, and `all_micp_results.csv` contains exactly one MICP result for
each manifest entry.

## Supplemental IN runs

The previous MICP result set was missing 62 VP cases. They were run with the
standard Lin2025/stlpy encoding and `lin2025` rule semantics, using a 30 s
Gurobi time limit, one solver thread, one repeat, and headless plotting.

| Cohort | Cases | Solver feasible | Successful | Success rate | Mean core time | Median core time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R_IN3_hand_draft | 19 | 14 | 12 | 63.16% | 1.689 s | 1.486 s |
| R_IN4 | 22 | 12 | 2 | 9.09% | 1.536 s | 1.397 s |
| R_IN5, standard 20-step | 15 | 15 | 10 | 66.67% | 1.626 s | 1.370 s |
| R_IN5, 150-step variants | 6 | 0 | 0 | 0.00% | 11.200 s | 10.549 s |

The six 150-step generated variants are retained in the per-case CSV and the
central scenario manifest for traceability, but are separated from standard
IN5 statistics. The standard supplemental population is therefore 56 cases;
the complete supplement including variants is 62 cases.

Of the 62 cases, 56 time-shift scenarios previously referenced temporary files
which no longer existed. They were regenerated from the original inD data and
stored persistently below `scenarios/experiment_scenarios/`. Every non-empty
`scenario_path` in the retained result CSV now points into that directory.

## Combined aligned IN results

| Cohort | Cases | Solver feasible | Successful | Success rate | Mean core time | Median core time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R_IN3_hand_draft | 68 | 50 | 28 | 41.18% | 1.572 s | 1.413 s |
| R_IN4 | 46 | 30 | 12 | 26.09% | 1.462 s | 1.390 s |
| R_IN5, standard 20-step | 44 | 44 | 22 | 50.00% | 1.465 s | 1.383 s |
| R_IN5, 150-step variants | 6 | 0 | 0 | 0.00% | 11.200 s | 10.549 s |

One IN3 historical row has no recorded core time, so its timing statistics use
67 of 68 rows.

## Validation-scope caveat

The 62 supplemental rows use the corrected fixed vehicle-pair validation:
success is checked for the recorded ego and fixed other vehicle at the updated
violation time. The 642 historical rows used the earlier full-scene quantified
monitor validation. Re-running all historical cases was outside this
supplemental task, so the combined success rates mix those two validation
protocols. The distinction is explicit in `validation_scope` and the original
result file is recorded in `result_source_csv`; no historical row was silently
overwritten.

## Retained files

- `all_micp_results.csv`: unified 704-row MICP result table.
- `all_summary.csv`: summary for all VP cohorts, including RG and IN rules.
- `RESULTS.md`: methodology, result summary, and validation caveats.
- `scenarios/experiment_scenarios/manifest.csv`: centralized scenario paths,
  original sources, uses, sizes, and SHA-256 checksums.
