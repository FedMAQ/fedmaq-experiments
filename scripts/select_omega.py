"""Close and resolve the Stage-1b power-mean omega selection at selected p."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analysis import (
    closure_certificate,
    discover_runs,
    power_mean_stage_one_b,
    resolve_power_mean_omega,
    select_power_mean_omega_iso_byte,
)
from scripts.report_schema import write_report

DEFAULT_EXPECTED_RUNS = Path("docs/recut/power_mean_expected_runs.json")
DEFAULT_STAGE1A_RESOLUTION = Path("scripts/analysis_output/power_mean_degree_resolution.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-runs", type=Path, default=DEFAULT_EXPECTED_RUNS)
    parser.add_argument(
        "--stage1a-resolution",
        type=Path,
        default=DEFAULT_STAGE1A_RESOLUTION,
        help="path to Stage 1a degree resolution JSON to read selected_p from",
    )
    parser.add_argument(
        "--p",
        type=str,
        default=None,
        help="explicit compensation degree p (overrides --stage1a-resolution)",
    )
    parser.add_argument(
        "--selection-output",
        type=Path,
        default=Path("scripts/analysis_output/power_mean_omega_selection.json"),
    )
    parser.add_argument(
        "--resolution-output",
        type=Path,
        default=Path("scripts/analysis_output/power_mean_omega_resolution.json"),
    )
    args = parser.parse_args()

    selected_p: float | str
    if args.p is not None:
        selected_p = args.p if args.p == "min" else float(args.p)
    elif args.stage1a_resolution.is_file():
        stage1a_doc = json.loads(args.stage1a_resolution.read_text(encoding="utf-8"))
        data = stage1a_doc.get("data", stage1a_doc)
        selected_p = data["selected_p"]
    else:
        raise SystemExit(
            f"Stage 1a resolution not found at {args.stage1a_resolution}; "
            "provide --p explicitly or run scripts/select_power_mean.py first"
        )

    if not args.expected_runs.is_file():
        raise SystemExit(
            f"missing {args.expected_runs}; generate it with "
            "`uv run python scripts/dump_expected_runs.py --power-mean-recut`"
        )
    manifest = json.loads(args.expected_runs.read_text(encoding="utf-8"))["groups"]
    runs = discover_runs(args.experiments_root)
    stage_1b = power_mean_stage_one_b()
    closure = closure_certificate(runs, manifest, groups=[stage_1b.experiment_group])
    if not closure["all_closed"]:
        raise SystemExit(
            "power-mean Stage 1b is not closed; inspect its closure certificate before selection: "
            + json.dumps(closure["groups"][stage_1b.experiment_group], indent=2)
        )

    selection = select_power_mean_omega_iso_byte(runs, selected_p=selected_p)
    resolution = resolve_power_mean_omega(selection)
    documents = (
        (args.selection_output, "power_mean_omega_selection", selection),
        (args.resolution_output, "power_mean_omega_resolution", resolution),
    )
    for path, report_type, document in documents:
        write_report(path, report_type, document)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
