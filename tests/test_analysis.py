"""Unit tests for scripts/analysis.py: baseline-comparison deltas and tie-break rule."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analysis import RunRecord, accuracy_at_round, compare_to_baselines, select_winner


def _df(rounds, accs, mbs):
    return pd.DataFrame(
        {
            "round": rounds,
            "test/accuracy": accs,
            "communication/cumulative_mb": mbs,
        }
    )


def test_accuracy_at_round_returns_matching_round():
    df = _df([1, 2, 100], [0.1, 0.2, 0.87], [1.0, 2.0, 50.0])
    assert accuracy_at_round(df, 100) == pytest.approx(0.87)


def test_accuracy_at_round_falls_back_to_last_row_when_round_missing():
    df = _df([1, 2, 3], [0.1, 0.2, 0.5], [1.0, 2.0, 3.0])
    assert accuracy_at_round(df, 100) == pytest.approx(0.5)


def _write_run(tmp_path, algorithm, formulation, seed, accs, mbs):
    """Write a fake job dir with an experiment_log.csv; return a RunRecord
    pointing at it (dataset/alpha fixed to keep fixtures small)."""
    job_dir = tmp_path / f"{algorithm}_{formulation}_{seed}"
    job_dir.mkdir()
    csv_path = job_dir / "experiment_log.csv"
    _df(list(range(1, len(accs) + 1)), accs, mbs).to_csv(csv_path, index=False)
    return RunRecord(
        job_dir=job_dir,
        dataset="cifar10",
        alpha=0.5,
        algorithm=algorithm,
        formulation=formulation,
        seed=seed,
        csv_path=csv_path,
    )


def test_compare_to_baselines_computes_paired_per_seed_accuracy_delta(tmp_path):
    # FedAvg reference: final acc 0.80 across all 3 seeds -> floor = 0.9*0.80 = 0.72
    fedavg_runs = [
        _write_run(tmp_path, "fedavg", None, s, [0.5, 0.7, 0.80], [10, 20, 30]) for s in (1, 2, 3)
    ]
    # Winning FedMAQ formulation (2): final acc 0.85/0.83/0.81 per seed
    fedmaq_runs = [
        _write_run(tmp_path, "fedmaq", 2, 1, [0.6, 0.75, 0.85], [5, 10, 15]),
        _write_run(tmp_path, "fedmaq", 2, 2, [0.6, 0.74, 0.83], [5, 10, 15]),
        _write_run(tmp_path, "fedmaq", 2, 3, [0.6, 0.73, 0.81], [5, 10, 15]),
    ]
    # Baseline FedPAQ: final acc 0.75/0.78/0.70 per seed
    fedpaq_runs = [
        _write_run(tmp_path, "fedpaq", None, 1, [0.5, 0.65, 0.75], [8, 16, 24]),
        _write_run(tmp_path, "fedpaq", None, 2, [0.5, 0.65, 0.78], [8, 16, 24]),
        _write_run(tmp_path, "fedpaq", None, 3, [0.5, 0.65, 0.70], [8, 16, 24]),
    ]
    all_runs = fedavg_runs + fedmaq_runs + fedpaq_runs

    winner_result = select_winner(all_runs)
    result = compare_to_baselines(all_runs, winner_result)

    key = "cifar10_alpha_0.5_vs_fedpaq"
    assert key in result
    entry = result[key]
    # deltas per seed: 0.85-0.75=0.10, 0.83-0.78=0.05, 0.81-0.70=0.11
    assert entry["mean_delta"] == pytest.approx((0.10 + 0.05 + 0.11) / 3, abs=1e-6)
    assert entry["min_delta"] == pytest.approx(0.05, abs=1e-6)
    assert entry["max_delta"] == pytest.approx(0.11, abs=1e-6)
    assert len(entry["per_seed"]) == 3


def test_compare_to_baselines_reports_rounds_to_target_per_side(tmp_path):
    # floor = 0.9*0.80 = 0.72
    fedavg_runs = [
        _write_run(tmp_path, "fedavg", None, s, [0.5, 0.7, 0.80], [10, 20, 30]) for s in (1, 2, 3)
    ]
    # FedMAQ crosses 0.72 at round 3 (0.85); FedPAQ never crosses (caps at 0.70)
    fedmaq_runs = [
        _write_run(tmp_path, "fedmaq", 0, s, [0.6, 0.70, 0.85], [5, 10, 15]) for s in (1, 2, 3)
    ]
    fedpaq_runs = [
        _write_run(tmp_path, "fedpaq", None, s, [0.5, 0.65, 0.70], [8, 16, 24]) for s in (1, 2, 3)
    ]
    all_runs = fedavg_runs + fedmaq_runs + fedpaq_runs

    winner_result = select_winner(all_runs)
    result = compare_to_baselines(all_runs, winner_result)

    entry = result["cifar10_alpha_0.5_vs_fedpaq"]
    for seed_detail in entry["per_seed"].values():
        assert seed_detail["fedmaq_rounds_to_target"] == 3
        assert seed_detail["baseline_rounds_to_target"] is None


def test_select_winner_near_tie_reselects_by_accuracy(tmp_path):
    """margin_mb (0.5) < pooled stdev of top-2 candidates' crossing MBs (~0.94)
    -> near-tie -> re-select by higher mean accuracy at R=100, even though
    formulation 0 has the lower mean MB."""
    fedavg_runs = [
        _write_run(tmp_path, "fedavg", None, s, [0.5, 0.80], [5, 10]) for s in (1, 2, 3)
    ]
    # floor = 0.9*0.80 = 0.72
    formulation_a = [  # mean crossing mb = 10, mean final acc = 0.80
        _write_run(tmp_path, "fedmaq", 0, 1, [0.5, 0.80], [5, 10]),
        _write_run(tmp_path, "fedmaq", 0, 2, [0.5, 0.81], [5, 11]),
        _write_run(tmp_path, "fedmaq", 0, 3, [0.5, 0.79], [5, 9]),
    ]
    formulation_b = [  # mean crossing mb = 10.5, mean final acc = 0.85 (higher)
        _write_run(tmp_path, "fedmaq", 1, 1, [0.5, 0.85], [5, 10.5]),
        _write_run(tmp_path, "fedmaq", 1, 2, [0.5, 0.84], [5, 11.5]),
        _write_run(tmp_path, "fedmaq", 1, 3, [0.5, 0.86], [5, 9.5]),
    ]
    all_runs = fedavg_runs + formulation_a + formulation_b

    result = select_winner(all_runs)
    entry = result["cifar10_alpha_0.5"]
    assert entry["margin_mb"] == pytest.approx(0.5, abs=1e-6)
    assert entry["winner"] == 1


def test_select_winner_clear_margin_keeps_min_mb_winner(tmp_path):
    """margin_mb (20) >> pooled stdev of top-2 candidates' crossing MBs (~11)
    -> not a near-tie -> the lower-mean-MB formulation still wins even though
    the other formulation has much higher accuracy."""
    fedavg_runs = [
        _write_run(tmp_path, "fedavg", None, s, [0.5, 0.80], [5, 10]) for s in (1, 2, 3)
    ]
    formulation_a = [  # mean crossing mb = 10, mean final acc = 0.80
        _write_run(tmp_path, "fedmaq", 0, 1, [0.5, 0.80], [5, 10]),
        _write_run(tmp_path, "fedmaq", 0, 2, [0.5, 0.81], [5, 11]),
        _write_run(tmp_path, "fedmaq", 0, 3, [0.5, 0.79], [5, 9]),
    ]
    formulation_b = [  # mean crossing mb = 30, mean final acc = 0.95 (higher, but MB gap is real)
        _write_run(tmp_path, "fedmaq", 1, 1, [0.5, 0.95], [5, 30]),
        _write_run(tmp_path, "fedmaq", 1, 2, [0.5, 0.94], [5, 31]),
        _write_run(tmp_path, "fedmaq", 1, 3, [0.5, 0.96], [5, 29]),
    ]
    all_runs = fedavg_runs + formulation_a + formulation_b

    result = select_winner(all_runs)
    entry = result["cifar10_alpha_0.5"]
    assert entry["margin_mb"] == pytest.approx(20.0, abs=1e-6)
    assert entry["winner"] == 0
