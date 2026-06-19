import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = REPO_ROOT / "evaluation" / "config" / "ind_in4.csv"
OUTPUT_CSV = REPO_ROOT / "evaluation" / "config" / "ind_in4_smt_p2_c2_dpll_stats.csv"
BATCH_SCRIPT = REPO_ROOT / "examples" / "batch_test_vp_repairer_in4.py"
RESULT_PREFIX = "__BATCH_RESULT__="


def load_rows(csv_path: Path):
    with csv_path.open(newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def run_one_case(scenario_id: str, ego_id: int):
    cmd = [
        sys.executable,
        str(BATCH_SCRIPT),
        "--case",
        scenario_id,
        str(ego_id),
        "dpll",
        "smt",
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )

    result = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            result = json.loads(line[len(RESULT_PREFIX):])
            break

    if result is None:
        result = {
            "success": False,
            "wall_time": 0.0,
            "error": f"missing result payload (exit={completed.returncode})",
        }

    return result, completed


def write_rows(rows, csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = load_rows(INPUT_CSV)
    total = len(rows)
    success_count = 0

    for idx, row in enumerate(rows, start=1):
        scenario_id = row["scenario_id"]
        ego_id = int(row["ego_id"])
        print(f"[{idx}/{total}] scenario={scenario_id}, ego_id={ego_id}", flush=True)

        result, completed = run_one_case(scenario_id, ego_id)

        repairable = 1 if result.get("success") else 0
        total_time = float(result.get("wall_time", 0.0) or 0.0)
        row["smt_p2_c2_repairable"] = repairable
        row["smt_p2_c2_total_time_sec"] = f"{total_time:.6f}"
        row["smt_p2_c2_error"] = result.get("error", "")

        if repairable:
            success_count += 1

        print(
            f"  repairable={repairable}, total_time={total_time:.3f}s, "
            f"exit={completed.returncode}, error={row['smt_p2_c2_error']}"
            ,
            flush=True,
        )
        write_rows(rows, OUTPUT_CSV)

    write_rows(rows, OUTPUT_CSV)
    print(f"Finished: {success_count}/{total} repairable", flush=True)
    print(f"Result table written to: {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
