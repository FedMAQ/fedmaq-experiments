"""Close and resolve the Stage-1 power-mean degree selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis import (
    closure_certificate,
    discover_runs,
    power_mean_stage_one,
    resolve_power_mean_degree,
    select_power_mean_degree_iso_byte,
)
from scripts.report_schema import write_report

DEFAULT_EXPECTED_RUNS = Path("docs/recut/power_mean_expected_runs.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-runs", type=Path, default=DEFAULT_EXPECTED_RUNS)
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=Path("scripts/analysis_output/power_mean_degree_selection.json"),
    )
    parser.add_argument(
        "--resolution-output",
        type=Path,
        default=Path("scripts/analysis_output/power_mean_degree_resolution.json"),
    )
    args = parser.parse_args()

    if not args.expected_runs.is_file():
        raise SystemExit(
            f"missing {args.expected_runs}; generate it with "
            "`uv run python scripts/dump_expected_runs.py --power-mean-recut`"
        )
    manifest = json.loads(args.expected_runs.read_text(encoding="utf-8"))["groups"]
    runs = discover_runs(args.experiments_root)
    stage = power_mean_stage_one()
    closure = closure_certificate(runs, manifest, groups=[stage.experiment_group])
    if not closure["all_closed"]:
        raise SystemExit(
            "power-mean Stage 1 is not closed; inspect its closure certificate before selection: "
            + json.dumps(closure["groups"][stage.experiment_group], indent=2)
        )

    selection = select_power_mean_degree_iso_byte(runs)
    resolution = resolve_power_mean_degree(selection)
    documents = (
        (args.selection_output, "power_mean_degree_selection", selection),
        (args.resolution_output, "power_mean_degree_resolution", resolution),
    )
    for path, report_type, document in documents:
        write_report(path, report_type, document)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
