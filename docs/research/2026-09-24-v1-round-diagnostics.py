"""Recompute descriptive round diagnostics from the certified local V1 logs."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO.parent / "campaign-evidence-2026-09-22"
LADDER = (2, 3, 4, 5, 6, 7, 8, 16)


def number(row: dict, key: str) -> float | None:
    value = row.get(key)
    if value in (None, ""):
        return None
    return float(value)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3 or statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return math.nan
    return statistics.correlation(xs, ys)


def analyze(csv_path: Path, comparison: str, arm: str) -> dict:
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        csv_rows = list(csv.DictReader(stream))
    jsonl_path = csv_path.with_suffix(".jsonl")
    with jsonl_path.open(encoding="utf-8") as stream:
        jsonl_rows = [json.loads(line) for line in stream if line.strip()]
    manifest = json.loads((csv_path.parent / "run_manifest.json").read_text(encoding="utf-8"))
    if len(csv_rows) != 101 or len(jsonl_rows) != 101:
        raise ValueError(f"expected 101 CSV/JSONL rows: {csv_path}")
    csv_rounds = {int(row["round"]): row for row in csv_rows}
    jsonl = {int(row["round"]): row for row in jsonl_rows}
    if sorted(csv_rounds) != list(range(101)) or sorted(jsonl) != list(range(101)):
        raise ValueError(f"round sequence is not 0..100: {csv_path}")
    cross_check_keys = [
        "test/accuracy",
        "test/loss",
        "communication/cumulative_mb",
        *(f"algorithm/fedmaq/q_count_{b}" for b in LADDER),
        *(f"algorithm/fedmaq/q_hat_count_{b}" for b in LADDER),
    ]
    for round_number in range(101):
        for key in cross_check_keys:
            csv_value = number(csv_rounds[round_number], key)
            jsonl_value = number(jsonl[round_number], key)
            if (
                csv_value is not None
                and jsonl_value is not None
                and not math.isclose(csv_value, jsonl_value, rel_tol=1e-12, abs_tol=1e-12)
            ):
                raise ValueError(f"CSV/JSONL mismatch for {key} round {round_number}: {csv_path}")
    # JSONL is the complete scalar record; CSV is cross-checked for stable fields.
    rounds = jsonl

    trained = list(range(1, 101))
    early = list(range(1, 11))
    late = list(range(91, 101))
    accuracy = {r: number(rounds[r], "test/accuracy") for r in range(101)}
    gap = {
        r: number(rounds[r], "client/avg_train_acc") - accuracy[r]
        for r in trained
        if number(rounds[r], "client/avg_train_acc") is not None and accuracy[r] is not None
    }
    late_steps = [abs(accuracy[r] - accuracy[r - 1]) for r in late]
    late_acc = [accuracy[r] for r in late if accuracy[r] is not None]
    early_gap = [gap[r] for r in early if r in gap]
    late_gap = [gap[r] for r in late if r in gap]

    kd_loss = {r: number(rounds[r], "algorithm/fedmaq/server_kd_loss") for r in trained}
    kd_skipped = [number(rounds[r], "algorithm/fedmaq/kd_skipped") or 0.0 for r in trained]
    dropped = [number(rounds[r], "algorithm/fedmaq/dropped_teachers") or 0.0 for r in trained]
    kd_x, kd_y = [], []
    for r in range(2, 101):
        if kd_loss[r] is not None and accuracy[r] is not None and accuracy[r - 1] is not None:
            kd_x.append(kd_loss[r])
            kd_y.append(accuracy[r] - accuracy[r - 1])

    applied_total = pre_cap_total = applied_8 = pre_cap_8 = 0
    tv_rounds = []
    for r in trained:
        applied = [int(number(rounds[r], f"algorithm/fedmaq/q_count_{b}") or 0) for b in LADDER]
        pre_cap = [int(number(rounds[r], f"algorithm/fedmaq/q_hat_count_{b}") or 0) for b in LADDER]
        a_sum, h_sum = sum(applied), sum(pre_cap)
        if a_sum and h_sum:
            applied_total += a_sum
            pre_cap_total += h_sum
            applied_8 += applied[LADDER.index(8)]
            pre_cap_8 += pre_cap[LADDER.index(8)]
            # Total-variation distance between round histograms. Without client IDs,
            # this is the minimum moved fraction compatible with the two marginals.
            tv_rounds.append(
                0.5 * sum(abs(a / a_sum - h / h_sum) for a, h in zip(applied, pre_cap, strict=True))
            )

    late_kd = [kd_loss[r] for r in late if kd_loss[r] is not None]
    binding = [number(rounds[r], "algorithm/fedmaq/tier1_binding_fraction") for r in trained]
    binding = [value for value in binding if value is not None]
    mean_q_k_max = [number(rounds[r], "algorithm/fedmaq/avg_q_k_max") for r in trained]
    mean_q_k_max = [value for value in mean_q_k_max if value is not None]
    rounds_max_q_k_15 = sum(
        number(rounds[r], "algorithm/fedmaq/max_q_k_max") == 15 for r in trained
    )
    cumulative = number(rounds[100], "communication/cumulative_mb")
    cumulative_90 = number(rounds[90], "communication/cumulative_mb")

    return {
        "comparison": comparison,
        "dataset": manifest["run"]["dataset"],
        "alpha": manifest["run"]["alpha"],
        "seed": manifest["run"]["seed"],
        "arm": arm,
        "kd_epochs": manifest["config"]["algorithm"]["kd_epochs"],
        "late_accuracy_pct": 100 * mean(late_acc),
        "late_abs_step_mean_pp": 100 * mean(late_steps),
        "late_abs_step_max_pp": 100 * max(late_steps),
        "early_local_train_global_eval_diff_pp": 100 * mean(early_gap),
        "late_local_train_global_eval_diff_pp": 100 * mean(late_gap),
        "local_train_global_eval_diff_drift_pp": 100 * (mean(late_gap) - mean(early_gap)),
        "late_kd_loss_mean": mean(late_kd),
        "kd_skipped_rounds": int(sum(kd_skipped)),
        "dropped_teacher_rounds": sum(value > 0 for value in dropped),
        "dropped_teacher_total": sum(dropped),
        "corr_kd_loss_vs_same_round_accuracy_delta": corr(kd_x, kd_y),
        "applied_q8_pct": 100 * applied_8 / applied_total if applied_total else math.nan,
        "pre_cap_q8_pct": 100 * pre_cap_8 / pre_cap_total if pre_cap_total else math.nan,
        "histogram_min_move_pct": 100 * mean(tv_rounds),
        "tier1_binding_pct": 100 * mean(binding),
        "mean_raw_q_k_max": mean(mean_q_k_max),
        "rounds_with_sampled_q_k_max_15": rounds_max_q_k_15,
        "cumulative_mb_round100": cumulative,
        "late_communication_mb_round90_to100": cumulative - cumulative_90,
    }


def main() -> None:
    files: list[tuple[Path, str, str]] = []
    for path in sorted(
        EVIDENCE.glob("outputs/formal/*/benchmark_grid/fedmaq/**/experiment_log.csv")
    ):
        files.append((path, "benchmark_grid", "fedmaq"))
    ablation = EVIDENCE / "outputs/formal/cifar10_mobilenetv2/ablation"
    for arm in ("fedmaq", "fedmaq_no_kd"):
        for path in sorted((ablation / arm).glob("dirichlet_alpha_*/seed_*/experiment_log.csv")):
            files.append((path, "ablation", arm))
    records = [analyze(path, comparison, arm) for path, comparison, arm in files]
    fields = list(records[0])
    writer = csv.DictWriter(sys.stdout, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)


if __name__ == "__main__":
    main()
