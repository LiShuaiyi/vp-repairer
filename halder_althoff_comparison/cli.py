"""Command-line entry point for a single extracted velocity-planning case."""

from __future__ import annotations

import argparse
import json

from .io import load_problem, write_result
from .planner import MinimumViolationPlanner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="method-local JSON problem")
    parser.add_argument("--output", help="write full trajectory/result JSON")
    args = parser.parse_args(argv)

    lattice, environment, rules, initial_s, initial_v, data = load_problem(args.input)
    planner = MinimumViolationPlanner(lattice, environment, rules)
    result = planner.plan(initial_s, initial_v)
    payload = result.as_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        write_result(args.output, result, data.get("metadata"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
