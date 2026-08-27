"""Snapshot the runs ``conf/matrix/*.yaml`` promises into ``docs/freeze/``.

``scripts/analysis.py``'s closure certificate answers "are the right runs
present", which needs something to compare against. A key count cannot serve:
"105 distinct keys" proves only that 105 keys exist, not that they are the 105
the design calls for, and the failure this exists to catch is precisely a set
that is self-consistent and short. The restored v1 bundle's completeness audit
reported 42 entries and ``all_complete: true`` for the 105-run primary grid.

The expected set is derived, never hand-written. Each matrix is expanded by
``common.expand_matrix`` -- the same expansion ``run_matrix.py`` dispatches from
-- and each run's ``alpha``, ``formulation`` and ``post_process`` are read from a
Hydra composition performed as ``scripts/run.py`` would perform it. That last
part is load-bearing: the ablation arms inherit ``conf/algorithm/fedmaq.yaml``
through their defaults list, so reading ``formulation`` off an arm's own file
finds nothing while the run itself carries Formulation 2.

    uv run python scripts/dump_expected_runs.py          # write docs/freeze/
    uv run python scripts/dump_expected_runs.py --check  # verify it is current

``--check`` exits non-zero when the snapshot is stale, so a matrix edit that
changes what the grid promises is caught rather than shipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import expand_matrix, identity_key

REPO_ROOT = Path(__file__).resolve().parents[1]
CONF_DIR = REPO_ROOT / "conf"
MATRIX_DIR = CONF_DIR / "matrix"
SNAPSHOT_PATH = REPO_ROOT / "docs" / "freeze" / "expected_runs.json"
POWER_MEAN_SNAPSHOT_PATH = REPO_ROOT / "docs" / "recut" / "power_mean_expected_runs.json"

# The matrices whose runs the manuscript reports, and therefore the only ones the
# closure certificate can hold anything to.
#
# ``ci_test`` and ``mobilenetv2_smoke_50r`` are excluded deliberately, and their
# absence is not an oversight to be repaired: the first is a two-run R=2 smoke
# check, the second a superseded single-seed sweep that ADR-0008's protocol
# replaced. Requiring either to be present would make the certificate red on a
# clean checkout; requiring their absence would make it red on any machine that
# has run the CI check locally.
#
# The three ``benchmark_grid*`` files are listed separately and union into one
# group, which is the whole reason they share an ``experiment_group``.
REPORTABLE_MATRICES = (
    "ablation",
    "baseline_tuning",
    "benchmark_grid",
    "benchmark_grid_cifar100",
    "benchmark_grid_femnist",
    "formulation_study",
    "pass2_explore",
    "pass2_factorial",
    "pass3_freeze_confirm",
    "uniform_memory_control",
)

POWER_MEAN_RECUT_MATRICES = ("power_mean_design",)


def _load_matrix(name: str) -> dict:
    return OmegaConf.to_container(OmegaConf.load(MATRIX_DIR / f"{name}.yaml"), resolve=True)


def expected_identities(matrix_names: tuple[str, ...] = REPORTABLE_MATRICES) -> dict[str, dict]:
    """Every run identity the named matrices promise, grouped by experiment group."""
    groups: dict[str, dict] = {}
    cache: dict[tuple, tuple[float, int | str | None, bool]] = {}

    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):

        def resolved(dataset, experiment, het, alg, ovr):
            key = (dataset, experiment, het, alg, ovr)
            if key not in cache:
                # Group selections in the order build_run_command emits them, then
                # the matrix file's own value overrides.
                selections = [f"dataset={dataset}", f"heterogeneity={het}", f"algorithm={alg}"]
                if experiment:
                    selections.append(f"experiment={experiment}")
                cfg = compose(config_name="config", overrides=selections + list(ovr))
                cache[key] = (
                    float(cfg.heterogeneity.alpha),
                    cfg.algorithm.get("formulation"),
                    bool(cfg.algorithm.get("post_process", False)),
                )
            return cache[key]

        for name in matrix_names:
            matrix = _load_matrix(name)
            experiment = matrix.get("experiment")
            for spec in expand_matrix(matrix, name):
                alpha, formulation, post_process = resolved(
                    spec["dataset"],
                    experiment,
                    spec["heterogeneity"],
                    spec["algorithm_config"],
                    tuple(spec["overrides"]),
                )
                group = groups.setdefault(
                    spec["experiment_group"],
                    {"matrices": [], "regimes": {}, "runs": []},
                )
                if name not in group["matrices"]:
                    group["matrices"].append(name)
                # Recorded per (group, algorithm config) rather than per group,
                # because the primary grid runs FedMAQ with the post-processing
                # pipeline and its six baselines without it -- so the regime is
                # single-valued only once the arm is named. That is exactly the
                # granularity that lets identity_key omit both fields: the key
                # already carries the group and the algorithm config.
                regime = group["regimes"].setdefault(
                    spec["algorithm_config"],
                    {"phases": [], "post_process": [], "total_rounds": []},
                )
                for field, value in (
                    ("phases", spec["phase"]),
                    ("post_process", post_process),
                    ("total_rounds", spec["total_rounds"]),
                ):
                    if value not in regime[field]:
                        regime[field].append(value)
                group["runs"].append(
                    identity_key(
                        dataset=spec["dataset"],
                        experiment_group=spec["experiment_group"],
                        algorithm_config=spec["algorithm_config"],
                        variant=spec["variant"],
                        alpha=alpha,
                        formulation=formulation,
                        seed=spec["seed"],
                    )
                )

    for group in groups.values():
        group["runs"].sort()
        group["count"] = len(group["runs"])
    return groups


def _render(groups: dict[str, dict], command: str) -> str:
    document = {
        "note": (
            "GENERATED FILE -- do not edit by hand. Regenerate with "
            f"`{command}`. Derived from "
            "conf/matrix/*.yaml; scripts/analysis.py reads it as the expected "
            "side of the closure certificate."
        ),
        "groups": {name: groups[name] for name in sorted(groups)},
    }
    return json.dumps(document, indent=2) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the committed snapshot differs from the matrices",
    )
    parser.add_argument(
        "--power-mean-recut",
        action="store_true",
        help="write or check the separate expected set for the power-mean re-cut",
    )
    args = parser.parse_args()

    matrix_names = POWER_MEAN_RECUT_MATRICES if args.power_mean_recut else REPORTABLE_MATRICES
    snapshot_path = POWER_MEAN_SNAPSHOT_PATH if args.power_mean_recut else SNAPSHOT_PATH
    command = "uv run python scripts/dump_expected_runs.py --power-mean-recut"
    if not args.power_mean_recut:
        command = "uv run python scripts/dump_expected_runs.py"
    groups = expected_identities(matrix_names)
    rendered = _render(groups, command)

    if args.check:
        if not snapshot_path.is_file():
            print(f"missing snapshot: {snapshot_path.relative_to(REPO_ROOT)}", file=sys.stderr)
            return 1
        if snapshot_path.read_text(encoding="utf-8") != rendered:
            print(
                f"{snapshot_path.relative_to(REPO_ROOT)} is stale; regenerate with `{command}`",
                file=sys.stderr,
            )
            return 1
        print(f"{snapshot_path.relative_to(REPO_ROOT)} is current")
        return 0

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(rendered, encoding="utf-8")
    counts = ", ".join(f"{name} {body['count']}" for name, body in sorted(groups.items()))
    print(f"wrote {snapshot_path.relative_to(REPO_ROOT)} ({counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
