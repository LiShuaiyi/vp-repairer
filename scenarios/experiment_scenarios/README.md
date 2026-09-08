# Experiment scenario archive

This directory contains the CommonRoad XML inputs referenced by the retained
VP, SMT, MICP, and sampling experiment tables. Result CSV files use the paths
in this directory instead of machine-local or temporary paths.

The subdirectories preserve source provenance and prevent different files with
the same CommonRoad scenario ID from overwriting one another:

- `highd/` and `mona/`: interstate scenarios;
- `ind_2024/`, `ind_converter_2026/`, and `ind_13/`: inD conversion cohorts;
- `generated_sampling/` and `generated_micp/`: independently regenerated
  time-shift windows used by the two baselines;
- `variants/in3/` and `variants/in5/`: generated rule variants.

`manifest.csv` records the canonical repository path, original source path,
experiment usage, file size, and SHA-256 checksum for every archived XML.
