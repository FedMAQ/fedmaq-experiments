"""Build a plot-ready Tier 1 memory-ceiling table from experiment logs."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_REQUIRED_COLUMNS = (
    "round",
    "algorithm/fedmaq/avg_q",
    "algorithm/fedmaq/avg_q_k_max",
    "algorithm/fedmaq/tier1_binding_fraction",
)


def _run_label(csv_path: Path) -> str:
    """Identify a run from its canonical output path when available."""
    source = csv_path.resolve()
    try:
        outputs_index = source.parts.index("outputs")
    except ValueError:
        return source.stem

    run_parts = source.parts[outputs_index + 1 : -1]
    return "/".join(run_parts) if run_parts else source.stem


def build_tier1_ceiling_frame(csv_paths: Iterable[Path]) -> pd.DataFrame:
    """Return long-form realized-q versus Tier-1-ceiling data for plotting.

    Logs from non-resource-aware arms have no Tier 1 columns and are skipped.
    This keeps the analysis input honest: an absent ceiling means the arm did
    not apply the modeled memory constraint, rather than a zero-valued ceiling.
    """
    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        source = Path(csv_path)
        frame = pd.read_csv(source)
        tier1_columns = set(_REQUIRED_COLUMNS[2:])
        if not tier1_columns.intersection(frame.columns):
            continue
        missing = set(_REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{source} is missing Tier 1 columns: {sorted(missing)}")
        selected = frame.loc[:, _REQUIRED_COLUMNS].rename(
            columns={
                "algorithm/fedmaq/avg_q": "realized_q",
                "algorithm/fedmaq/avg_q_k_max": "tier1_ceiling_q",
                "algorithm/fedmaq/tier1_binding_fraction": "tier1_binding_fraction",
            }
        )
        selected.insert(0, "run", _run_label(source))
        frames.append(selected)

    columns = ["run", "round", "realized_q", "tier1_ceiling_q", "tier1_binding_fraction"]
    if not frames:
        return pd.DataFrame(columns=columns)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["run", "round"], kind="stable")
        .reset_index(drop=True)
    )


def write_tier1_ceiling_analysis(csv_paths: Iterable[Path], output_prefix: Path) -> pd.DataFrame:
    """Write the analysis CSV and the two-panel Tier 1 figure."""
    frame = build_tier1_ceiling_frame(csv_paths)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_prefix.with_suffix(".csv"), index=False)

    figure, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 7), constrained_layout=True)
    for run, run_frame in frame.groupby("run", sort=False):
        axes[0].plot(run_frame["round"], run_frame["realized_q"], label=f"{run} realized q")
        axes[0].plot(run_frame["round"], run_frame["tier1_ceiling_q"], "--", label=f"{run} Tier 1")
        axes[1].plot(run_frame["round"], run_frame["tier1_binding_fraction"], label=run)
    axes[0].set_ylabel("bits")
    axes[0].set_title("Realized bit-width and modeled Tier 1 ceiling")
    axes[0].legend(fontsize="small", ncol=2)
    axes[1].set_xlabel("round")
    axes[1].set_ylabel("binding fraction")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(fontsize="small")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=150)
    plt.close(figure)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="FedMAQ experiment_log.csv files")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("scripts/analysis_output/tier1_memory_ceiling"),
        help="Output path without extension for the CSV and PNG",
    )
    args = parser.parse_args()
    frame = write_tier1_ceiling_analysis(args.csv, args.output_prefix)
    print(f"Wrote {len(frame)} Tier 1 rows to {args.output_prefix.with_suffix('.csv')}")
    print(f"Wrote Tier 1 figure to {args.output_prefix.with_suffix('.png')}")


if __name__ == "__main__":
    main()
