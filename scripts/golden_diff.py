"""Golden-output harness for architecture-deepening Step 2 (ADR-0003).

Captures pre-refactor ``experiment_log.csv`` output for a fixed set of configs,
then — after the ``TrainingSkeleton`` migration — re-runs the same configs and
diffs column-by-column. Per Decision 40, the gate is literal bit-exact equality
on every column except the two real-wall-clock columns.

Usage:
    uv run python scripts/golden_diff.py capture   # run BEFORE touching client_hooks code
    uv run python scripts/golden_diff.py compare   # run AFTER each migration step

The golden set covers one config per ``standard.py`` branch (S2a) plus ``fedkd``,
``feddistill``, and ``cfd`` (S2b). See ADR-0003: a plain-FedAvg-only diff would
never exercise FedProx's post-backward instrumentation or the KD/DAdaQuant/FedMAQ
loss-hook paths.

``fedmd`` is deliberately excluded from this default set (Decision 45): it was
already dropped from the formal baseline stack (Decision 25) and its disk-persisted
multi-phase training (pub/priv pretrain + digest + revisit, up to 4x ``run_epochs``
per round) makes it by far the slowest config here (~9 min for 2 rounds on this
machine) for a baseline unlikely to reappear in future experiments. Its hook code
is untouched by candidate work to date; re-add it to ``GOLDEN_SET`` only for a
refactor that actually touches ``fedmd.py``/``kd_utils.py``'s FedMD path.
"""

import csv
import shutil
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path if not present (matches scripts/run_matrix.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.common import build_run_command, kill_ray_processes

# One config per branch this step's migration touches. Extend as S2b adds
# feddistill/cfd/fedmd.
GOLDEN_SET: list[str] = [
    "fedavg",  # StandardFit, no instrumentation
    "fedprox",  # StandardFit + on_after_backward grad-norm instrumentation + prox loss hook
    "fedpaq",  # StandardFit + real quantizing compressor_hook
    "fedavg_kd",  # StandardFit + ClientKDLossHook
    "dadaquant",  # DAdaQuantFit._pretrain_local_loss + dynamic q
    "fedmaq",  # FedMAQFit._reported_local_loss (last_loss, not mean)
    "fedkd",  # joint student+teacher loop, compress_and_reconstruct only
    "feddistill",  # run_epochs x1 + LogitTracker side-channel, no compression
    "cfd",  # run_epochs x2 (distill phase + CE phase), no compress/reconstruct tail
    # "fedmd" intentionally excluded — see Decision 45 in the module docstring.
]

# Real-wall-clock columns excluded from the bit-exact comparison (Decision 40).
IGNORED_COLUMNS = {"system/wall_time_sec", "system/cumulative_wall_time_sec"}

GOLDEN_ROOT = Path("outputs/golden/step2")
COMPARE_ROOT = Path("outputs/golden/step2_compare")
SEED = 42
SSD = "dirichlet_alpha_0.1"

# fedkd/fedmd persist client/teacher state on disk across rounds by design
# (baselines.md), keyed only by cid -- not by run/output-dir. Left uncleared,
# a later run silently inherits an earlier run's trained weights instead of
# starting cold, which looks like a code regression but isn't one. Must be
# wiped before every capture/compare run so each is a true independent trial.
PERSISTENCE_DIR = Path(".data_partitions/fedmd_models")


def _run(algorithm: str, target_dir: Path) -> None:
    if PERSISTENCE_DIR.exists():
        shutil.rmtree(PERSISTENCE_DIR)
    kill_ray_processes()
    cmd = build_run_command(
        dataset="cifar10",
        heterogeneity=SSD,
        algorithm=algorithm,
        total_rounds=2,
        seed=SEED,
        client_gpus=1.0,
        target_dir=target_dir,
        overrides=["experiment=ci"],
    )
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _diff(golden_rows: list[dict[str, str]], compare_rows: list[dict[str, str]], label: str) -> list[str]:
    problems: list[str] = []
    if len(golden_rows) != len(compare_rows):
        problems.append(f"[{label}] row count differs: golden={len(golden_rows)} compare={len(compare_rows)}")
        return problems

    golden_cols = set(golden_rows[0].keys()) if golden_rows else set()
    compare_cols = set(compare_rows[0].keys()) if compare_rows else set()
    if golden_cols != compare_cols:
        problems.append(
            f"[{label}] column set differs: only-in-golden={golden_cols - compare_cols} "
            f"only-in-compare={compare_cols - golden_cols}"
        )

    for i, (g_row, c_row) in enumerate(zip(golden_rows, compare_rows, strict=False)):
        for col in golden_cols & compare_cols:
            if col in IGNORED_COLUMNS:
                continue
            if g_row.get(col) != c_row.get(col):
                problems.append(
                    f"[{label}] row {i} column {col!r}: golden={g_row.get(col)!r} compare={c_row.get(col)!r}"
                )
    return problems


def capture() -> None:
    for algorithm in GOLDEN_SET:
        target = GOLDEN_ROOT / algorithm
        _run(algorithm, target)
    print(f"\nGolden output captured under {GOLDEN_ROOT}/")


def compare() -> None:
    all_problems: list[str] = []
    for algorithm in GOLDEN_SET:
        golden_csv = GOLDEN_ROOT / algorithm / "experiment_log.csv"
        if not golden_csv.exists():
            all_problems.append(f"[{algorithm}] no golden output found at {golden_csv} — run `capture` first")
            continue

        target = COMPARE_ROOT / algorithm
        _run(algorithm, target)
        compare_csv = target / "experiment_log.csv"

        problems = _diff(_read_csv(golden_csv), _read_csv(compare_csv), algorithm)
        if problems:
            all_problems.extend(problems)
        else:
            print(f"[{algorithm}] OK — bit-exact match (excluding wall-clock columns)")

    if all_problems:
        print("\nGOLDEN DIFF FAILED:")
        for p in all_problems:
            print(f"  {p}")
        sys.exit(1)
    print("\nAll golden diffs passed.")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("capture", "compare"):
        print(__doc__)
        sys.exit(1)
    {"capture": capture, "compare": compare}[sys.argv[1]]()


if __name__ == "__main__":
    main()
