"""Unit tests for scripts/analysis.py: baseline-comparison deltas and tie-break rule."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analysis import (
    EXPLORATION_GROUP,
    FORMULATION_STUDY_GROUP,
    GRID_GROUP,
    RunRecord,
    accuracy_at_round,
    build_ablation_table,
    compare_to_baselines,
    discover_runs,
    exploration_noise_margin,
    resolve_frozen_formulation,
    select_winner,
)
from common import get_canonical_output_dir

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _write_run(tmp_path, algorithm, formulation, seed, accs, mbs, group=None, alpha=0.5):
    """Write a fake job dir with an experiment_log.csv; return a RunRecord
    pointing at it (dataset/alpha fixed to keep fixtures small).

    ``experiment_group`` is not decoration: ``select_winner`` reads formulation
    candidates from the study group alone and the accuracy floor from the grid's
    FedAvg rows alone (Decision 71), so a fixture that leaves the group unset is
    testing a run that no matrix could have produced. FedMAQ therefore defaults
    to the study group and everything else to the grid.
    """
    group = group or (FORMULATION_STUDY_GROUP if algorithm == "fedmaq" else GRID_GROUP)
    # The group belongs in the path for the same reason it belongs in the record:
    # the study and the grid run the same algorithm at the same formulation, skew
    # and seed, and are separated by nothing else.
    job_dir = tmp_path / f"{group}_{algorithm}_{formulation}_{seed}_{alpha}"
    job_dir.mkdir()
    csv_path = job_dir / "experiment_log.csv"
    _df(list(range(1, len(accs) + 1)), accs, mbs).to_csv(csv_path, index=False)
    return RunRecord(
        job_dir=job_dir,
        dataset="cifar10",
        alpha=alpha,
        algorithm=algorithm,
        formulation=formulation,
        seed=seed,
        csv_path=csv_path,
        experiment_group=group,
    )


def test_compare_to_baselines_computes_paired_per_seed_accuracy_delta(tmp_path):
    # FedAvg reference: final acc 0.80 across all 3 seeds -> floor = 0.9*0.80 = 0.72
    fedavg_runs = [
        _write_run(tmp_path, "fedavg", None, s, [0.5, 0.7, 0.80], [10, 20, 30]) for s in (1, 2, 3)
    ]
    # The grid's frozen FedMAQ rows (formulation 2): final acc 0.85/0.83/0.81 per
    # seed. Grid group, because the headline table is a grid-internal contrast.
    fedmaq_runs = [
        _write_run(tmp_path, "fedmaq", 2, 1, [0.6, 0.75, 0.85], [5, 10, 15], group=GRID_GROUP),
        _write_run(tmp_path, "fedmaq", 2, 2, [0.6, 0.74, 0.83], [5, 10, 15], group=GRID_GROUP),
        _write_run(tmp_path, "fedmaq", 2, 3, [0.6, 0.73, 0.81], [5, 10, 15], group=GRID_GROUP),
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
        _write_run(tmp_path, "fedmaq", 0, s, [0.6, 0.70, 0.85], [5, 10, 15], group=GRID_GROUP)
        for s in (1, 2, 3)
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


def test_exploration_fedmaq_runs_never_become_formulation_candidates(tmp_path):
    """Decision 71. ``pass2_factorial`` and ``pass3_freeze_confirm`` dispatch
    ``algorithm=fedmaq`` at the held-out alpha = 0.3, and ``fedmaq.yaml`` carries
    ``formulation: 3``, so on algorithm and formulation alone they are
    indistinguishable from study candidates.

    Admitting them manufactures a verdict at a skew the study never ran, where no
    FedAvg reference exists by design -- so the failure is not a subtly wrong
    winner but a ``ValueError`` out of the floor, taking the whole analysis run
    down at the point in the dispatch order where the freeze is written.
    """
    fedavg = [_write_run(tmp_path, "fedavg", None, s, [0.5, 0.80], [5, 10]) for s in (1, 2, 3)]
    study = [
        _write_run(tmp_path, "fedmaq", f, s, [0.5, 0.80 + 0.01 * f], [5, 10 + f])
        for f in (0, 3)
        for s in (1, 2, 3)
    ]
    exploration = [
        _write_run(
            tmp_path,
            "fedmaq",
            3,
            s,
            [0.5, 0.82],
            [5, 4],  # far cheaper: would win outright if it were admitted
            group=EXPLORATION_GROUP,
            alpha=0.3,
        )
        for s in (4, 5, 6)
    ]

    result = select_winner(fedavg + study + exploration)

    assert set(result) == {"cifar10_alpha_0.5"}, (
        "an entry at alpha=0.3 means the exploration factorial was read as a "
        "formulation-study arm; there is no FedAvg reference at the held-out skew"
    )
    assert set(result["cifar10_alpha_0.5"]["formulations"][3]["seeds"]) == {1, 2, 3}


def test_grid_fedmaq_rows_never_enter_the_formulation_study(tmp_path):
    """Decision 71. The grid runs FedMAQ at the study's own dataset, skews and
    seeds, differing only by ``algorithm.post_process=true``.

    That flag is the whole point: §4.3.6 withholds the pipeline so the formulas
    are judged on their mathematical merit, and the pipeline's payload savings
    land in exactly the cumulative-MB figure the winner rule minimizes. Pooling
    the grid's rows into Formulation 3's cell would credit the incumbent
    formulation with savings the study exists to exclude -- and Formulation 3 is
    the incumbent that ships in ``fedmaq.yaml``, so the contamination flatters
    the status quo rather than perturbing it randomly.
    """
    fedavg = [_write_run(tmp_path, "fedavg", None, s, [0.5, 0.80], [5, 10]) for s in (1, 2, 3)]
    study = [  # formulation 0 wins pipeline-free: 10 MB vs formulation 3's 20 MB
        *[_write_run(tmp_path, "fedmaq", 0, s, [0.5, 0.80], [5, 10]) for s in (1, 2, 3)],
        *[_write_run(tmp_path, "fedmaq", 3, s, [0.5, 0.80], [5, 20]) for s in (1, 2, 3)],
    ]
    grid = [  # same seeds, same skew, pipeline on -> 2 MB
        _write_run(tmp_path, "fedmaq", 3, s, [0.5, 0.80], [1, 2], group=GRID_GROUP)
        for s in (1, 2, 3)
    ]

    entry = select_winner(fedavg + study + grid)["cifar10_alpha_0.5"]

    assert entry["winner"] == 0, (
        "formulation 3 can only win here by absorbing the grid's pipeline-on "
        "payloads, which chapter_4.tex:312 withholds from the study"
    )
    assert entry["formulations"][3]["mean_cumulative_mb"] == pytest.approx(20.0)


def test_baseline_comparison_pairs_grid_fedmaq_not_the_pipeline_free_study(tmp_path):
    """Decision 71. Both sides of the headline table must carry the pipeline.

    The study and the grid share ``(dataset, alpha, seed, formulation)``, so a
    ``{seed: run}`` map built without a group filter takes twelve candidates for
    six slots and resolves by iteration order. The wrong resolution reports
    pipeline-free FedMAQ against pipeline-era baselines, which is the one
    direction chapter_4.tex:312's regime rule exists to forbid.
    """
    fedavg = [_write_run(tmp_path, "fedavg", None, s, [0.5, 0.80], [5, 10]) for s in (1, 2, 3)]
    study = [  # pipeline-free: final accuracy 0.60, deliberately far lower
        _write_run(tmp_path, "fedmaq", 3, s, [0.5, 0.60], [5, 40]) for s in (1, 2, 3)
    ]
    grid = [
        _write_run(tmp_path, "fedmaq", 3, s, [0.5, 0.90], [1, 4], group=GRID_GROUP)
        for s in (1, 2, 3)
    ]
    fedpaq = [_write_run(tmp_path, "fedpaq", None, s, [0.5, 0.70], [8, 16]) for s in (1, 2, 3)]

    result = compare_to_baselines(
        fedavg + study + grid + fedpaq, select_winner(fedavg + study + fedpaq)
    )

    entry = result["cifar10_alpha_0.5_vs_fedpaq"]
    assert entry["mean_delta"] == pytest.approx(0.20, abs=1e-6), (
        "0.90 - 0.70 is the grid contrast; -0.10 means the pipeline-free study "
        "run stood in for FedMAQ's grid row"
    )
    assert entry["formulation_matches_freeze"] is True


def test_baseline_comparison_flags_a_grid_that_ran_a_different_formulation(tmp_path):
    """§4.3.1's tag forbids editing a frozen config downstream of it, so a grid
    row carrying a formulation the study did not select is a pre-registration
    breach and must surface in the report rather than be silently averaged."""
    fedavg = [_write_run(tmp_path, "fedavg", None, s, [0.5, 0.80], [5, 10]) for s in (1, 2, 3)]
    study = [_write_run(tmp_path, "fedmaq", 3, s, [0.5, 0.80], [5, 10]) for s in (1, 2, 3)]
    grid = [
        _write_run(tmp_path, "fedmaq", 1, s, [0.5, 0.90], [1, 4], group=GRID_GROUP)
        for s in (1, 2, 3)
    ]
    fedpaq = [_write_run(tmp_path, "fedpaq", None, s, [0.5, 0.70], [8, 16]) for s in (1, 2, 3)]

    result = compare_to_baselines(fedavg + study + grid + fedpaq, select_winner(fedavg + study))

    entry = result["cifar10_alpha_0.5_vs_fedpaq"]
    assert entry["frozen_formulation"] == 3
    assert entry["fedmaq_formulation"] == 1
    assert entry["formulation_matches_freeze"] is False


def test_select_winner_near_tie_reselects_by_accuracy(tmp_path):
    """margin_mb (0.5) < max of the top-2 candidates' own crossing-MB stdevs (1.0)
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
    """margin_mb (20) >> max of the top-2 candidates' own crossing-MB stdevs (1.0)
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


def test_near_tie_threshold_is_within_candidate_spread_not_pooled_spread(tmp_path):
    """Decision 66. The threshold must be each candidate's own seed-to-seed
    spread, never the spread of both candidates' values concatenated.

    Pooling folds the between-candidate separation into the threshold, so it
    grows with the very margin it is judging: with within-candidate sd ``s`` and
    separation ``d``, the combined sample variance is ``(4s^2 + 1.5d^2)/5`` and
    the rule fires whenever ``d < 1.069s`` rather than ``d < s``. That band is
    exactly what this fixture sits in. Both formulations have sd 1.0 and are
    separated by 1.03, so the pooled stdev is ~1.058 and the old rule would call
    a near-tie and hand the win to formulation 1 on accuracy; the published rule
    is that 1.03 exceeds either candidate's own spread, so the lower-MB
    formulation 0 wins on the scalar rule as written.
    """
    fedavg_runs = [_write_run(tmp_path, "fedavg", None, s, [0.5, 0.80], [5, 10]) for s in (1, 2, 3)]
    # floor = 0.9*0.80 = 0.72; crossing MB is the second row in each run.
    formulation_a = [  # crossing mbs [10, 11, 9] -> mean 10.0, stdev 1.0
        _write_run(tmp_path, "fedmaq", 0, 1, [0.5, 0.80], [5, 10]),
        _write_run(tmp_path, "fedmaq", 0, 2, [0.5, 0.81], [5, 11]),
        _write_run(tmp_path, "fedmaq", 0, 3, [0.5, 0.79], [5, 9]),
    ]
    formulation_b = [  # crossing mbs [11.03, 12.03, 10.03] -> mean 11.03, stdev 1.0
        _write_run(tmp_path, "fedmaq", 1, 1, [0.5, 0.85], [5, 11.03]),
        _write_run(tmp_path, "fedmaq", 1, 2, [0.5, 0.84], [5, 12.03]),
        _write_run(tmp_path, "fedmaq", 1, 3, [0.5, 0.86], [5, 10.03]),
    ]

    entry = select_winner(fedavg_runs + formulation_a + formulation_b)["cifar10_alpha_0.5"]

    assert entry["margin_mb"] == pytest.approx(1.03, abs=1e-6)
    assert entry["winner"] == 0, (
        "1.03 MB exceeds either candidate's own seed-to-seed spread (1.0), so this "
        "is not a near-tie and the accuracy tie-break must not fire. A winner of 1 "
        "means the threshold was computed from the two candidates' pooled values, "
        "which is self-referential -- see Decision 66."
    )


def _verdict(alpha, winner, formulations=None, floor=0.72):
    """One per-skew entry shaped like ``select_winner``'s output."""
    return {
        "dataset": "cifar10",
        "alpha": alpha,
        "target_accuracy_floor": floor,
        "formulations": formulations or {},
        "winner": winner,
        "margin_mb": None,
    }


def _winner_result(severe, moderate, severe_detail=None):
    return {
        "cifar10_alpha_0.1": _verdict(0.1, severe, severe_detail),
        "cifar10_alpha_1.0": _verdict(1.0, moderate),
    }


def test_agreeing_skews_freeze_that_formulation(tmp_path):
    """Decision 64, rule 1. The common case needs no tie-break."""
    out = resolve_frozen_formulation(_winner_result(severe=3, moderate=3))
    assert out["frozen_formulation"] == 3
    assert out["rule"] == "agreement"
    assert out["skews_agree"] is True
    # Formulation 3 is the one the refinement layer was selected under, so the
    # reserved recheck of conf/matrix/formulation_study.yaml does not fire.
    assert out["recheck_required"] is False


def test_diverging_skews_freeze_the_severe_skew_winner(tmp_path):
    """Decision 64, rule 2. §4.3.6 promises a skew-dependent winner is a finding;
    the freeze still takes one scalar, and alpha = 0.1 is the regime the thesis's
    claims are staked on."""
    out = resolve_frozen_formulation(_winner_result(severe=1, moderate=3))
    assert out["frozen_formulation"] == 1
    assert out["rule"] == "divergence_severe_skew_breaks"
    assert out["skews_agree"] is False
    assert out["alpha_0.1_winner"] == 1 and out["alpha_1.0_winner"] == 3
    # Not Formulation 3, so the layer must be re-tested where it now has to live.
    assert out["recheck_required"] is True


def test_one_skew_disqualifying_its_whole_field_defers_to_the_other(tmp_path):
    """Decision 65, rule 3. Rule 2 does not apply: there is only one valid
    verdict, so there is nothing to break a tie between."""
    out = resolve_frozen_formulation(_winner_result(severe=None, moderate=2))
    assert out["frozen_formulation"] == 2
    assert out["rule"] == "one_sided_disqualification"
    assert out["surviving_alpha"] == 1.0
    assert out["contribution_withdrawn"] is False


def test_total_disqualification_falls_back_to_accuracy_and_withdraws_the_claim(tmp_path):
    """Decision 65, rule 4. The accuracy-floor guard catching all five is a live
    outcome, not a hypothetical: the floor is 90% of *uncompressed* FedAvg and
    the study runs FedMAQ quantized with the post-processing pipeline withheld.

    A winner is still produced, because ``fedmaq.yaml`` takes a number either way
    and freezing the incumbent by default would settle the thesis's primary
    methodological contribution with a default value. What is withdrawn is the
    contribution *claim*, not the configuration.
    """
    detail = {
        0: {"mean_accuracy_r100": 0.61, "disqualified": True},
        3: {"mean_accuracy_r100": 0.68, "disqualified": True},
        4: {"mean_accuracy_r100": 0.55, "disqualified": True},
    }
    out = resolve_frozen_formulation(_winner_result(None, None, severe_detail=detail))

    assert out["frozen_formulation"] == 3, "highest mean top-1 at R=100 at alpha=0.1"
    assert out["rule"] == "total_disqualification_accuracy_fallback"
    assert out["contribution_withdrawn"] is True, (
        "§4.3.6 frames formulation selection as the thesis's primary methodological "
        "contribution. A field in which nothing reached the FedAvg-relative target "
        "does not support that framing, and the fallback must say so rather than "
        "letting a rescued winner paper over it."
    )
    assert out["fallback_mean_accuracy_r100"] == pytest.approx(0.68)


def test_freeze_rule_refuses_a_partial_sweep(tmp_path):
    """The formulation study runs both skews by design. Resolving from one of
    them would silently apply rule 3 to a sweep that simply had not finished."""
    partial = {"cifar10_alpha_0.1": _verdict(0.1, 3)}
    with pytest.raises(ValueError, match="missing alpha"):
        resolve_frozen_formulation(partial)


def _explore_run(tmp_path, label, seed, final_acc, refinements, alpha=0.3, group=EXPLORATION_GROUP):
    """One exploration-phase run at the held-out skew, with its refinement flags.

    Defaults to the factorial's group because that is the stage that makes the
    keep-or-drop calls. The screening sweep and the R=100 confirmation are
    separate groups and are never pooled with it.
    """
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
        experiment_group=group,
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
        for s, acc in zip((0, 42, 123), (0.70, 0.72, 0.74), strict=True)
    ]
    # Mean 0.7450 -> delta 0.0250. Above sigma (0.02), below the margin (0.0283).
    runs += [
        _explore_run(tmp_path, "sv", s, acc, SOFT_VOTING_ON)
        for s, acc in zip((0, 42, 123), (0.735, 0.745, 0.755), strict=True)
    ]
    # Mean 0.7800 -> delta 0.0600, clears the margin.
    runs += [
        _explore_run(tmp_path, "ema", s, acc, EMA_ON)
        for s, acc in zip((0, 42, 123), (0.77, 0.78, 0.79), strict=True)
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
        for s, acc in zip((0, 42, 123), (0.70, 0.72, 0.74), strict=True)
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


SV_AND_EMA_ON = (True, True, False)


def test_surviving_set_is_a_measured_cell_and_never_a_union(tmp_path):
    """The union of clearing cells can name a combination the factorial never ran.

    Here soft_voting alone and ema_student alone both clear, but the cell that
    holds *both* does not. Unioning the clearing cells would freeze the pair --
    shipping a configuration whose only measurement says it fails to clear.
    """
    runs = [
        _explore_run(tmp_path, "off", s, acc, OFF)
        for s, acc in zip((0, 42, 123), (0.70, 0.72, 0.74), strict=True)
    ]
    # sigma = 0.02, margin = 0.0283, baseline mean = 0.72.
    runs += [  # mean 0.78 -> delta 0.06, clears
        _explore_run(tmp_path, "sv", s, acc, SOFT_VOTING_ON)
        for s, acc in zip((0, 42, 123), (0.77, 0.78, 0.79), strict=True)
    ]
    runs += [  # mean 0.80 -> delta 0.08, clears by more
        _explore_run(tmp_path, "ema", s, acc, EMA_ON)
        for s, acc in zip((0, 42, 123), (0.79, 0.80, 0.81), strict=True)
    ]
    runs += [  # mean 0.73 -> delta 0.01, does NOT clear
        _explore_run(tmp_path, "both", s, acc, SV_AND_EMA_ON)
        for s, acc in zip((0, 42, 123), (0.72, 0.73, 0.74), strict=True)
    ]

    result = exploration_noise_margin(runs)

    assert result["verdicts"]["soft_voting+ema_student"]["retained"] is False
    # Both singletons clear; the tie on size breaks toward the larger delta.
    assert result["surviving_refinement_set"] == ["ema_student"]
    assert result["surviving_cell"] == "ema_student"


def test_surviving_set_prefers_the_smallest_clearing_cell(tmp_path):
    """Parsimony, not the highest score: a bigger cell must beat a smaller one on
    more than noise to justify the extra mechanism, and the margin rule alone does
    not test that. The smallest cell that clears is the one that ships."""
    runs = [
        _explore_run(tmp_path, "off", s, acc, OFF)
        for s, acc in zip((0, 42, 123), (0.70, 0.72, 0.74), strict=True)
    ]
    runs += [  # delta 0.05, clears
        _explore_run(tmp_path, "sv", s, acc, SOFT_VOTING_ON)
        for s, acc in zip((0, 42, 123), (0.76, 0.77, 0.78), strict=True)
    ]
    runs += [  # delta 0.09, clears by more -- but costs a second mechanism
        _explore_run(tmp_path, "both", s, acc, SV_AND_EMA_ON)
        for s, acc in zip((0, 42, 123), (0.80, 0.81, 0.82), strict=True)
    ]

    result = exploration_noise_margin(runs)

    assert result["surviving_refinement_set"] == ["soft_voting"]
    assert result["discarded"] == ["ema_student"]


def test_nothing_clearing_freezes_unrefined_rather_than_crowning_a_best(tmp_path):
    """conf/matrix/pass3_freeze_confirm.yaml pre-registers the empty set as a real
    outcome. Selecting the highest scorer when none clears is the exact failure
    the margin exists to prevent."""
    runs = [
        _explore_run(tmp_path, "off", s, acc, OFF)
        for s, acc in zip((0, 42, 123), (0.70, 0.72, 0.74), strict=True)
    ]
    runs += [  # delta 0.02 -- above sigma, below the margin
        _explore_run(tmp_path, "sv", s, acc, SOFT_VOTING_ON)
        for s, acc in zip((0, 42, 123), (0.73, 0.74, 0.75), strict=True)
    ]

    result = exploration_noise_margin(runs)

    assert result["surviving_refinement_set"] == []
    assert result["surviving_cell"] is None


def test_exploration_stages_are_refused_rather_than_pooled(tmp_path):
    """The three exploration matrices differ in round budget and share the
    unrefined cell, so pooling them computes sigma from a mixture of horizons
    with seed 0 counted twice. Scoping is by group, and a miss is an error."""
    factorial = [
        _explore_run(tmp_path, "off", s, acc, OFF)
        for s, acc in zip((0, 42, 123), (0.70, 0.72, 0.74), strict=True)
    ]
    confirm = [
        _explore_run(tmp_path, "c_off", s, acc, OFF, group="pass3_freeze_confirm")
        for s, acc in zip((0, 42, 123), (0.80, 0.86, 0.92), strict=True)
    ]

    result = exploration_noise_margin(factorial + confirm)
    # The R=100 runs have a far wider spread; if they were pooled in, sigma moves.
    assert result["sigma_unrefined"] == pytest.approx(0.02, abs=1e-9)
    assert result["unrefined_seeds"] == 3

    other = exploration_noise_margin(factorial + confirm, experiment_group="pass3_freeze_confirm")
    assert other["sigma_unrefined"] == pytest.approx(0.06, abs=1e-9)

    missing = exploration_noise_margin(factorial, experiment_group="nonexistent_group")
    assert "error" in missing
    assert "pass2_factorial" in missing["groups_present"]


def test_margin_reports_its_own_uncertainty_and_the_multiplicity_it_carries(tmp_path):
    """§4.3.1 calls the margin "measured rather than asserted"; a measurement
    without its uncertainty is nearer an assertion. Deepening the reference cell
    is what pays for a usable interval, so the interval has to be visible."""
    deep = [
        _explore_run(tmp_path, "off", s, acc, OFF)
        for s, acc in zip((0, 42, 123, 7, 21), (0.70, 0.71, 0.72, 0.73, 0.74), strict=True)
    ]
    deep += [
        _explore_run(tmp_path, "sv", s, acc, SOFT_VOTING_ON)
        for s, acc in zip((0, 42, 123), (0.77, 0.78, 0.79), strict=True)
    ]

    result = exploration_noise_margin(deep)

    ci = result["sigma_confidence_interval"]
    assert ci["n"] == 5
    assert ci["low"] < result["sigma_unrefined"] < ci["high"]
    # n=5 keeps the interval inside a fivefold span; n=3 spans roughly twelvefold.
    assert ci["high"] / ci["low"] < 5.0

    mult = result["multiplicity"]
    assert mult["comparisons"] == 1
    assert 0.0 < mult["family_wise_false_positive_rate"] < 1.0
    # With n_ref=5 against n_cell=3 the threshold sits near 1.94 standard errors.
    assert result["verdicts"]["soft_voting"]["margin_in_standard_errors"] == pytest.approx(
        1.936, abs=1e-3
    )


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
                        group=FORMULATION_STUDY_GROUP,
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
        _ablation_run(
            tmp_path, "fedmaq", s, 0.85, 30.0, group=FORMULATION_STUDY_GROUP, formulation=f
        )
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


# --------------------------------------------------------------------------
# End-to-end readback: matrix file -> canonical path -> discover_runs -> margin.
#
# Every other fixture in this file constructs RunRecord directly, which skips
# the path-composition and path-parsing seam entirely. That is exactly how three
# exploration matrices shipped for months dispatching every cell of a stage into
# one directory (Decision 76): nothing here ever built a path or read one back.
# --------------------------------------------------------------------------


def _write_discoverable_run(root, out_dir, *, seed, refinements, accuracy, alpha=0.3):
    """Write the artifacts discover_runs actually requires, at a real path."""
    job_dir = root / out_dir
    (job_dir / ".hydra").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        OmegaConf.create(
            {
                "dataset": {"name": "cifar10"},
                "heterogeneity": {"alpha": alpha},
                "seed": seed,
                "algorithm": {
                    "name": "fedmaq",
                    "soft_voting": refinements[0],
                    "ema_student": refinements[1],
                    "grad_norm_ema": refinements[2],
                    "post_process": False,
                },
            }
        ),
        job_dir / ".hydra" / "config.yaml",
    )
    OmegaConf.save(
        OmegaConf.create({"hydra": {"runtime": {"choices": {"algorithm": "fedmaq"}}}}),
        job_dir / ".hydra" / "hydra.yaml",
    )
    _df([1, 25, 50], [0.1, 0.2, accuracy], [1.0, 2.0, 3.0]).to_csv(
        job_dir / "experiment_log.csv", index=False
    )


def _expand_matrix(name):
    """Expand a shipped matrix exactly as run_matrix.py does (seed-major)."""
    matrix = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / "conf" / "matrix" / f"{name}.yaml"), resolve=True
    )
    matrix_seeds = [int(s) for s in matrix.get("seeds", [0])]

    def seeds_for(run):
        return [int(s) for s in run.get("seeds", matrix_seeds)]

    all_seeds = list(matrix_seeds)
    for run in matrix["runs"]:
        for seed in seeds_for(run):
            if seed not in all_seeds:
                all_seeds.append(seed)

    tasks = []
    for het in matrix["heterogeneities"]:
        for seed in all_seeds:
            for run in matrix["runs"]:
                if seed not in seeds_for(run):
                    continue
                flags = dict(o.split("=", 1) for o in run.get("overrides", []))
                tasks.append(
                    {
                        "label": run.get("label", run["alg"]),
                        "seed": seed,
                        "refinements": tuple(
                            flags.get(f"algorithm.{n}", "false") == "true"
                            for n in ("soft_voting", "ema_student", "grad_norm_ema")
                        ),
                        "out_dir": get_canonical_output_dir(
                            phase=matrix["phase"],
                            dataset=matrix["dataset"],
                            model=matrix["model"],
                            exp_group=matrix["experiment_group"],
                            algorithm=run["alg"],
                            heterogeneity=het,
                            seed=seed,
                            variant=run.get("variant", ""),
                        ),
                    }
                )
    return matrix, tasks


def test_factorial_on_disk_reads_back_as_eight_distinct_cells(tmp_path):
    """The shipped pass2_factorial layout must survive a round trip to disk.

    Asserting directory uniqueness (test_simulation.py) proves the runs do not
    overwrite each other. It does not prove analysis.py can still *find* them:
    ``phase_and_group_of`` keys on a path length of exactly 7 and reads the group
    from ``parts[3]``, so the ``fedmaq__<variant>`` segment introduced by
    Decision 76 is only safe while it stays one path component. This walks the
    real matrix file to real paths to a real margin.

    It reads conf/matrix/pass2_factorial.yaml rather than a fixture on purpose:
    delete a ``variant:`` and this fails, which is the failure that cost Stage
    1.1 and would have cost the 26-run factorial.
    """
    matrix, tasks = _expand_matrix("pass2_factorial")
    assert len(tasks) == 26, f"factorial expanded to {len(tasks)} tasks, §4.3.1 dispatches 26"

    # Unrefined cell gets real spread (so sigma > 0); one cell clears decisively.
    unrefined_accs = {0: 0.500, 42: 0.510, 123: 0.490, 7: 0.505, 21: 0.495}
    for task in tasks:
        if task["refinements"] == (False, False, False):
            acc = unrefined_accs[task["seed"]]
        elif task["refinements"] == (True, True, True):
            acc = 0.600
        else:
            acc = 0.500
        _write_discoverable_run(
            tmp_path,
            task["out_dir"],
            seed=task["seed"],
            refinements=task["refinements"],
            accuracy=acc,
        )

    runs = discover_runs(tmp_path)
    assert len(runs) == 26, (
        f"discover_runs found {len(runs)} of 26 written runs. Cells are colliding "
        "on disk or the path no longer parses as an exploration layout."
    )
    assert {r.experiment_group for r in runs} == {"pass2_factorial"}, (
        "the fedmaq__<variant> segment broke group parsing; phase_and_group_of "
        "reads the group from parts[3] of a 7-part path."
    )
    assert {r.algorithm for r in runs} == {"fedmaq"}
    assert {r.phase for r in runs} == {"explore"}

    by_cell = {}
    for run in runs:
        by_cell.setdefault(run.refinements, []).append(run.seed)
    assert len(by_cell) == 8, f"read back {len(by_cell)} distinct cells, the 2^3 factorial has 8"
    assert sorted(by_cell[(False, False, False)]) == [0, 7, 21, 42, 123], (
        "the unrefined reference must read back at all five seeds -- it is the "
        "cell sigma is measured from"
    )

    result = exploration_noise_margin(runs, alpha=0.3, experiment_group="pass2_factorial")
    assert "error" not in result, f"margin refused the shipped layout: {result.get('error')}"
    assert result["unrefined_seeds"] == 5
    assert result["sigma_unrefined"] > 0
    assert len(result["verdicts"]) == 7, "seven non-reference cells are judged against the margin"
    assert result["surviving_cell"] == "soft_voting+ema_student+grad_norm_ema"
