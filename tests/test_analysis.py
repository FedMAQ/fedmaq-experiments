"""Unit tests for scripts/analysis.py: baseline-comparison deltas and tie-break rule."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analysis import (
    RunRecord,
    accuracy_at_round,
    build_ablation_table,
    compare_to_baselines,
    exploration_noise_margin,
    select_winner,
)


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
    fedavg_runs = [_write_run(tmp_path, "fedavg", None, s, [0.5, 0.80], [5, 10]) for s in (1, 2, 3)]
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
    fedavg_runs = [_write_run(tmp_path, "fedavg", None, s, [0.5, 0.80], [5, 10]) for s in (1, 2, 3)]
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


def _explore_run(tmp_path, label, seed, final_acc, refinements, alpha=0.3):
    """One exploration-phase run at the held-out skew, with its refinement flags."""
    job_dir = tmp_path / f"{label}_{seed}"
    job_dir.mkdir()
    csv_path = job_dir / "experiment_log.csv"
    _df([1, 2, 50], [0.3, 0.5, final_acc], [10, 20, 30]).to_csv(csv_path, index=False)
    return RunRecord(
        job_dir=job_dir,
        dataset="cifar10",
        alpha=alpha,
        algorithm="fedmaq",
        formulation=3,
        seed=seed,
        csv_path=csv_path,
        refinements=refinements,
        experiment_group="pass2_exploration",
        phase="explore",
    )


OFF = (False, False, False)
SOFT_VOTING_ON = (True, False, False)
EMA_ON = (False, True, False)


def test_exploration_margin_is_scaled_above_sigma_not_equal_to_it(tmp_path):
    """§4.3.1: the margin a delta must clear is sqrt(2)*sigma, not sigma.

    The distinction is the whole point of the rule, so it is asserted against a
    mechanism whose delta sits deliberately *between* sigma and the margin: it
    would be retained under the wrong rule and must be dropped under the right
    one.
    """
    # Unrefined cell: 0.70 / 0.72 / 0.74 -> sigma = 0.02, margin = 0.0283.
    runs = [
        _explore_run(tmp_path, "off", s, acc, OFF)
        for s, acc in zip((0, 42, 123), (0.70, 0.72, 0.74))
    ]
    # Mean 0.7450 -> delta 0.0250. Above sigma (0.02), below the margin (0.0283).
    runs += [
        _explore_run(tmp_path, "sv", s, acc, SOFT_VOTING_ON)
        for s, acc in zip((0, 42, 123), (0.735, 0.745, 0.755))
    ]
    # Mean 0.7800 -> delta 0.0600, clears the margin.
    runs += [
        _explore_run(tmp_path, "ema", s, acc, EMA_ON)
        for s, acc in zip((0, 42, 123), (0.77, 0.78, 0.79))
    ]

    result = exploration_noise_margin(runs)

    assert result["sigma_unrefined"] == pytest.approx(0.02, abs=1e-9)
    assert result["noise_margin"] == pytest.approx(0.02 * 2**0.5, abs=1e-9)
    assert result["verdicts"]["soft_voting"]["retained"] is False
    assert result["verdicts"]["ema_student"]["retained"] is True
    assert result["surviving_refinement_set"] == ["ema_student"]
    assert result["discarded"] == ["soft_voting"]


def test_exploration_margin_flags_contamination_from_reported_skews(tmp_path):
    """A run at a confirmatory skew must be reported, not averaged in.

    This is the guard behind §4.3.1's held-out-skew claim. conf/matrix has been
    wrong about this before: it ran exploration at alpha 0.1 and 1.0, exactly the
    two skews the thesis reports on.
    """
    runs = [
        _explore_run(tmp_path, "off", s, acc, OFF)
        for s, acc in zip((0, 42, 123), (0.70, 0.72, 0.74))
    ]
    runs.append(_explore_run(tmp_path, "leak", 0, 0.71, OFF, alpha=0.1))

    result = exploration_noise_margin(runs)

    assert result["other_skews_present"] == [0.1]
    # The contaminating run must not move sigma.
    assert result["sigma_unrefined"] == pytest.approx(0.02, abs=1e-9)


def test_exploration_margin_refuses_to_guess_when_reference_is_underpowered(tmp_path):
    """No sigma means no keep-or-drop call; §4.3.1 requires three seeds."""
    runs = [_explore_run(tmp_path, "off", 0, 0.70, OFF)]
    result = exploration_noise_margin(runs)
    assert "error" in result
    assert "sigma" not in result


# --- §4.3.7 / §5.4 ablation matrix -------------------------------------------

ALL_REFINEMENTS = (True, True, True)


def _ablation_run(
    tmp_path,
    algorithm_config,
    seed,
    acc,
    mb,
    *,
    alpha=0.1,
    group="ablation",
    formulation=3,
    refinements=ALL_REFINEMENTS,
    post_process=False,
):
    """One ablation-grid run. ``algorithm`` is 'fedmaq' for every FedMAQ arm,
    exactly as the shipped configs declare it -- that collision is the thing the
    arm-identity plumbing has to survive."""
    algorithm = "fedmaq" if algorithm_config.startswith("fedmaq") else algorithm_config
    job_dir = tmp_path / f"{group}_{algorithm_config}_f{formulation}_{alpha}_{seed}"
    job_dir.mkdir()
    csv_path = job_dir / "experiment_log.csv"
    _df([1, 50, 100], [acc - 0.2, acc - 0.1, acc], [mb / 4, mb / 2, mb]).to_csv(
        csv_path, index=False
    )
    return RunRecord(
        job_dir=job_dir,
        dataset="cifar10",
        alpha=alpha,
        algorithm=algorithm,
        formulation=formulation,
        seed=seed,
        csv_path=csv_path,
        refinements=refinements,
        algorithm_config=algorithm_config,
        experiment_group=group,
        post_process=post_process,
    )


def _full_ablation_grid(tmp_path, include_anchor=True, **overrides):
    """All eight configurations at both skews and three seeds.

    ``include_anchor`` also emits the formulation study's Formulation 1 runs,
    which Configuration 4 is compared against under §4.3.7's fallback rule. They
    are part of a valid grid; §4.5 orders the calendar so they exist first.
    """
    specs = {
        "fedmaq_no_resource": (0.70, 40.0, 3, ALL_REFINEMENTS),
        "fedmaq_no_data": (0.72, 38.0, 3, ALL_REFINEMENTS),
        "fedmaq_no_state": (0.71, 39.0, 1, ALL_REFINEMENTS),
        "fedmaq_no_kd": (0.66, 36.0, 3, (False, True, True)),
        "fedavg_kd": (0.74, 90.0, None, (False, True, False)),
        "fedmaq": (0.76, 37.0, 3, ALL_REFINEMENTS),
        "fedmaq_no_refinements": (0.73, 37.5, 3, (False, False, False)),
    }
    runs = []
    for alpha in (0.1, 1.0):
        for cfg, (acc, mb, formulation, refinements) in specs.items():
            kwargs = {
                "alpha": alpha,
                "formulation": formulation,
                "refinements": refinements,
                **overrides.get(cfg, {}),
            }
            for seed in (0, 42, 123):
                runs.append(_ablation_run(tmp_path, cfg, seed, acc + seed * 1e-4, mb, **kwargs))
            # Configuration 1 is inherited from the primary grid, not re-run.
        for seed in (0, 42, 123):
            runs.append(
                _ablation_run(
                    tmp_path,
                    "fedavg",
                    seed,
                    0.80 + seed * 1e-4,
                    120.0,
                    alpha=alpha,
                    group="benchmark_grid",
                    formulation=None,
                    refinements=(False, False, False),
                )
            )
        if include_anchor:
            for seed in (0, 42, 123):
                runs.append(
                    _ablation_run(
                        tmp_path,
                        "fedmaq",
                        seed,
                        0.78 + seed * 1e-4,
                        33.0,
                        alpha=alpha,
                        group=None,
                        formulation=1,
                    )
                )
    return runs


def test_ablation_table_is_buildable_from_telemetry(tmp_path):
    """C5: the §4.3.7 table must be constructible from what the runs actually log.

    Every column the manuscript's build note asks for -- the eight arms, both
    skews, accuracy and communication as mean +- seed SD, each arm's formulation
    so Configuration 4's fallback is visible, and the pipeline regime -- has to
    come out of the telemetry join without a manual step.
    """
    table = build_ablation_table(_full_ablation_grid(tmp_path))

    assert table["parity"]["attributable"], table["parity"]["violations"]
    assert sorted(table["configurations"]) == [1, 2, 3, 4, 5, 6, 7, 8]

    for config_num, entry in table["configurations"].items():
        assert sorted(entry["cells"]) == ["alpha_0.1", "alpha_1.0"], config_num
        for cell in entry["cells"].values():
            assert cell["seeds"] == [0, 42, 123]
            assert cell["accuracy_r100"]["mean"] is not None
            assert cell["accuracy_r100"]["sd"] is not None
            assert cell["cumulative_mb"]["mean"] is not None

    # The formulation column: Configuration 4 alone falls back to Formulation 1,
    # and Configuration 6 has no formulation at all.
    assert table["configurations"][4]["formulation"] == 1
    assert table["configurations"][7]["formulation"] == 3
    assert table["configurations"][6]["formulation"] is None

    assert table["configurations"][1]["inherited"] is True
    assert table["configurations"][7]["inherited"] is False


def test_ablation_table_separates_configuration_7_from_the_primary_grid(tmp_path):
    """Configuration 7 and FedMAQ's primary-grid rows share every field the
    analysis reads except the group. Picking the wrong one silently swaps a
    pipeline-free anchor for a pipelined run, which is the confound §4.3.7 exists
    to prevent."""
    runs = _full_ablation_grid(tmp_path)
    runs += [
        _ablation_run(
            tmp_path,
            "fedmaq",
            seed,
            0.79,
            12.0,  # the pipeline compresses hard; that is how you spot the mix-up
            alpha=alpha,
            group="benchmark_grid",
            post_process=True,
        )
        for alpha in (0.1, 1.0)
        for seed in (0, 42, 123)
    ]

    table = build_ablation_table(runs)

    assert table["parity"]["attributable"], table["parity"]["violations"]
    config7 = table["configurations"][7]
    assert config7["post_process"] is False
    assert config7["cells"]["alpha_0.1"]["cumulative_mb"]["mean"] == pytest.approx(37.0)


def test_ablation_table_refuses_a_pipelined_arm(tmp_path):
    runs = _full_ablation_grid(tmp_path, fedmaq_no_data={"post_process": True})
    table = build_ablation_table(runs)
    assert not table["parity"]["attributable"]
    assert any("post-processing" in v for v in table["parity"]["violations"])


def test_ablation_table_refuses_a_broken_refinement_layer(tmp_path):
    """§5.4 requires the parity check before any delta is attributed. An arm that
    quietly drops a refinement is not one removal from Configuration 7."""
    runs = _full_ablation_grid(tmp_path, fedmaq_no_resource={"refinements": (True, False, True)})
    table = build_ablation_table(runs)
    assert not table["parity"]["attributable"]
    assert any("ema_student" in v for v in table["parity"]["violations"])


def test_ablation_arms_never_enter_the_formulation_study(tmp_path):
    """The arms declare ``name: fedmaq`` and sit on the winner's formulation, so
    an algorithm-level filter counts each of them as a formulation candidate and
    lets one stand in for FedMAQ in the headline comparison."""
    grid = _full_ablation_grid(tmp_path, include_anchor=False)
    fedavg = [r for r in grid if r.algorithm_config == "fedavg"]
    arms = [r for r in grid if r.experiment_group == "ablation"]
    study = [
        _ablation_run(tmp_path, "fedmaq", s, 0.85, 30.0, group=None, formulation=f)
        for f in (1, 3)
        for s in (0, 42, 123)
    ]

    clean = select_winner(fedavg + study)
    contaminated = select_winner(fedavg + study + arms)

    key = "cifar10_alpha_0.1"
    assert contaminated[key]["formulations"].keys() == clean[key]["formulations"].keys()
    assert contaminated[key]["winner"] == clean[key]["winner"]


def test_ablation_table_resolves_configuration_4_to_its_formulation_study_anchor(tmp_path):
    """§4.3.7's fallback rule: when the winner cannot express state-awareness
    removal, Configuration 4 runs on another formulation and is compared against
    that formulation's own full-FedMAQ runs. Reading it against Configuration 7
    would price the formulation change instead of the removed signal."""
    table = build_ablation_table(_full_ablation_grid(tmp_path))

    assert table["parity"]["attributable"], table["parity"]["violations"]
    anchor = table["configurations"][4]["parity_anchor"]
    assert anchor["formulation"] == 1
    assert anchor["cells"]["alpha_0.1"]["seeds"] == [0, 42, 123]
    # 0.78 plus the per-seed offsets the fixture applies (0, 42e-4, 123e-4).
    assert anchor["cells"]["alpha_0.1"]["accuracy_r100"]["mean"] == pytest.approx(0.7855, abs=1e-6)
    # Configuration 7 is on Formulation 3 and is not the anchor.
    assert "parity_anchor" not in table["configurations"][7]


def test_ablation_table_flags_a_fallback_arm_with_no_anchor(tmp_path):
    """The anchor is what the calendar ordering in §4.5 exists to guarantee. If
    the formulation study has not produced Formulation 1 runs, Configuration 4 has
    nothing valid to be compared against and the table must say so."""
    table = build_ablation_table(_full_ablation_grid(tmp_path, include_anchor=False))
    assert not table["parity"]["attributable"]
    assert any("parity anchor" in v for v in table["parity"]["violations"])


def test_exploration_margin_ignores_confirmatory_runs_entirely(tmp_path):
    """The contamination warning names conf/matrix/pass2_explore.yaml, so it must
    only fire on exploration runs. Every confirmatory FedMAQ run sits at a
    reported skew and declares ``name: fedmaq``; if those counted, the warning
    would fire on a correct grid and be trained away as noise."""
    runs = [
        _explore_run(tmp_path, "off", s, acc, OFF)
        for s, acc in zip((0, 42, 123), (0.70, 0.72, 0.74), strict=True)
    ]
    runs += _full_ablation_grid(tmp_path)

    result = exploration_noise_margin(runs)

    assert result["other_skews_present"] == []
    assert result["unrefined_seeds"] == 3
