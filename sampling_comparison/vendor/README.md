# Vendored paper baseline

`reactive_planner_ab96a839/commonroad_rp` and its `LICENSE` are exported
verbatim from:

- repository: `/home/shuaiyi/Documents/Lab/commonroad/reactive-planner`
- commit: `ab96a839`
- source version: `2024.1`

This is the historical Lin et al. (2025) revision that adds
`_check_collisions_rule_compliance` and evaluates each cost-sorted polynomial
with `RuleEvaluator`. The currently installed upstream Reactive Planner does
not contain that integration, so merely assigning `planner.rule_evaluators`
does not reproduce the paper baseline.
