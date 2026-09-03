"""Unit tests for scripts/analysis.py: baseline-comparison deltas and tie-break rule."""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analysis import (
    BASELINE_TUNING_GROUP,
    BASELINE_TUNING_WIDE_GROUP,
    EXPLORATION_GROUP,
    FEDPAQ_PIPELINE_GROUP,
    FORMULATION_STUDY_GROUP,
    GRID_GROUP,
    POWER_MEAN_DESIGN_GROUP,
    POWER_MEAN_OMEGA_GROUP,
    RunRecord,
    accuracy_at_budget,
    accuracy_at_round,
    baseline_tuning_margin,
    baseline_tuning_specs,
    build_ablation_table,
    closure_certificate,
    compare_fedpaq_pipeline_iso_byte,
    compare_to_baselines,
    compare_to_baselines_iso_byte,
    discover_runs,
    exploration_noise_margin,
    fedavg_at_fedmaq_budget,
    first_crossing,
    frozen_refinement_layer,
    iso_byte_scores,
    power_mean_stage_one,
    power_mean_stage_one_b,
    resolve_frozen_formulation,
    resolve_metrics_frame,
    resolve_power_mean_degree,
    resolve_power_mean_omega,
    round_at_budget,
    round_completeness,
    run_identity,
    select_power_mean_degree_iso_byte,
    select_power_mean_omega_iso_byte,
    select_winner,
    select_winner_iso_byte,
    sustained_crossing,
    write_baseline_tuning_plots,
)
from common import get_canonical_output_dir
from dump_expected_runs import POWER_MEAN_RECUT_MATRICES, expected_identities

from fedmaq.core.run_identity import parse_run_directory
from tests.run_fixtures import write_run

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_run_tree_carries_identity_and_provenance(canonical_run_tree, tmp_path):
    run = canonical_run_tree.write_run(
        "fedmaq",
        2,
        7,
        _df([1, 100], [0.4, 0.8], [2.0, 20.0]),
        group=FORMULATION_STUDY_GROUP,
    )

    parsed = parse_run_directory(run.job_dir, tmp_path)
    manifest = json.loads((run.job_dir / "run_manifest.json").read_text(encoding="utf-8"))

    assert parsed is not None
    assert parsed.algorithm_path == "fedmaq"
    assert parsed.variant == "f2"
    assert manifest["run"]["algorithm_config"] == "fedmaq"
    assert manifest["run"]["seed"] == 7
    assert set(manifest["git"]) >= {"commit", "branch", "tag", "dirty"}
    assert manifest["source_root"] == run.job_dir.resolve().as_posix()
    assert len(manifest["config_sha256"]) == 64


def test_selection_accepts_pre_resolved_metrics_frames(tmp_path, monkeypatch):
    """Analysis can use an in-memory frame without reopening a CSV artifact."""
    fedavg_path = tmp_path / "fedavg.csv"
    fedmaq_path = tmp_path / "fedmaq.csv"
    fedavg = RunRecord(
        job_dir=tmp_path / "fedavg",
        dataset="cifar10",
        alpha=0.5,
        algorithm="fedavg",
        formulation=None,
        seed=1,
        csv_path=fedavg_path,
        experiment_group=GRID_GROUP,
        promotable=True,
    )
    fedmaq = RunRecord(
        job_dir=tmp_path / "fedmaq",
        dataset="cifar10",
        alpha=0.5,
        algorithm="fedmaq",
        formulation=0,
        seed=1,
        csv_path=fedmaq_path,
        experiment_group=FORMULATION_STUDY_GROUP,
        promotable=True,
    )
    frames = {
        fedavg_path: _df([1, 2], [0.5, 0.8], [5.0, 10.0]),
        fedmaq_path: _df([1, 2], [0.5, 0.8], [4.0, 8.0]),
    }

    monkeypatch.setattr(
        "analysis.load_round_metrics",
        lambda _path: pytest.fail("the supplied metrics frames should be sufficient"),
    )

    assert resolve_metrics_frame(fedmaq, frames) is frames[fedmaq_path]
    result = select_winner([fedavg, fedmaq], frames=frames)

    assert result["cifar10_alpha_0.5"]["winner"] == 0


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


def _write_run(
    tmp_path,
    algorithm,
    formulation,
    seed,
    accs,
    mbs,
    group=None,
    alpha=0.5,
    dataset="cifar10",
    variant="",
):
    """Write one canonical fixture run and return its ``RunRecord``.

    ``experiment_group`` is not decoration: ``select_winner`` reads formulation
    candidates from the study group alone and the accuracy floor from the grid's
    FedAvg rows alone (Decision 71), so a fixture that leaves the group unset is
    testing a run that no matrix could have produced. FedMAQ therefore defaults
    to the study group and everything else to the grid.
    """
    return write_run(
        tmp_path,
        algorithm,
        formulation,
        seed,
        accs,
        mbs,
        group=group,
        alpha=alpha,
        dataset=dataset,
        variant=variant,
    )


def test_compare_to_baselines_computes_paired_per_seed_accuracy_delta(tmp_path):
    fedavg_runs = [
        _write_run(tmp_path, "fedavg", None, s, [0.5, 0.7, 0.80], [10, 20, 30]) for s in (1, 2, 3)
    ]
    fedmaq_runs = [
        _write_run(tmp_path, "fedmaq", 2, 1, [0.6, 0.75, 0.85], [5, 10, 15], group=GRID_GROUP),
        _write_run(tmp_path, "fedmaq", 2, 2, [0.6, 0.74, 0.83], [5, 10, 15], group=GRID_GROUP),
        _write_run(tmp_path, "fedmaq", 2, 3, [0.6, 0.73, 0.81], [5, 10, 15], group=GRID_GROUP),
    ]
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
    assert entry["mean_delta"] == pytest.approx((0.10 + 0.05 + 0.11) / 3, abs=1e-6)
    assert entry["min_delta"] == pytest.approx(0.05, abs=1e-6)
    assert entry["max_delta"] == pytest.approx(0.11, abs=1e-6)
    assert len(entry["per_seed"]) == 3


def test_compare_to_baselines_reports_rounds_to_target_per_side(tmp_path):
    fedavg_runs = [
        _write_run(tmp_path, "fedavg", None, s, [0.5, 0.7, 0.80], [10, 20, 30]) for s in (1, 2, 3)
    ]
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


def test_a_fired_recheck_carries_whether_it_is_owed_or_discharged(tmp_path):
    """`recheck_required: true` alone reads as an outstanding obligation. Decision
    85 discharged it because the frozen layer is empty, so the artifact has to say
    which of the two it is."""
    owed = resolve_frozen_formulation(
        _winner_result(severe=1, moderate=3), refinement_layer={"soft_voting"}
    )
    assert owed["recheck_required"] is True
    assert owed["recheck_discharged"] is False
    assert "soft_voting" in owed["recheck_note"]

    discharged = resolve_frozen_formulation(
        _winner_result(severe=1, moderate=3), refinement_layer=set()
    )
    assert discharged["recheck_required"] is True
    assert discharged["recheck_discharged"] is True

    # Absent a layer the function must not guess in the reassuring direction.
    assert (
        resolve_frozen_formulation(_winner_result(severe=1, moderate=3))["recheck_discharged"]
        is None
    )


def test_frozen_refinement_layer_reads_the_shipped_config(tmp_path):
    """The discharge above is only sound while the shipped layer is empty."""
    cfg = tmp_path / "fedmaq.yaml"
    cfg.write_text("soft_voting: false\nema_student: true\ngrad_norm_ema: false\n")
    assert frozen_refinement_layer(cfg) == {"ema_student"}
    assert frozen_refinement_layer() == set()


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
    return write_run(
        tmp_path,
        "fedmaq",
        3,
        seed,
        [0.3, 0.5, final_acc],
        [10, 20, 30],
        rounds=[1, 2, 50],
        group=group,
        alpha=alpha,
        variant=label,
        refinements=refinements,
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


def test_exploration_noise_margin_rejects_test_split_data(tmp_path):
    """Fails closed on test-split data, like every other margin function."""
    runs = [
        _explore_run(tmp_path, "off", s, acc, OFF)
        for s, acc in zip((0, 42, 123), (0.70, 0.72, 0.74), strict=True)
    ]
    runs[0].split = "test"

    with pytest.raises(ValueError, match="requires validation-split inputs"):
        exploration_noise_margin(runs)


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
    return write_run(
        tmp_path,
        algorithm,
        formulation,
        seed,
        [acc - 0.2, acc - 0.1, acc],
        [mb / 4, mb / 2, mb],
        rounds=[1, 50, 100],
        group=group,
        alpha=alpha,
        algorithm_config=algorithm_config,
        refinements=refinements,
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
# Most fixtures use the shared canonical writer through small domain-specific
# helpers. The end-to-end case below still exercises path composition, parsing,
# and discovery against a shipped matrix.
# --------------------------------------------------------------------------


def _write_discoverable_run(root, out_dir, *, seed, refinements, accuracy, alpha=0.3):
    """Write the artifacts discover_runs actually requires, at a real path."""
    run = write_run(
        root,
        "fedmaq",
        3,
        seed,
        [0.1, 0.2, accuracy],
        [1.0, 2.0, 3.0],
        rounds=[1, 25, 50],
        group=Path(out_dir).parts[3],
        alpha=alpha,
        refinements=refinements,
        phase="explore",
        output_dir=Path(out_dir),
    )
    manifest_path = run.job_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    config.update({"protocol": "replacement-v1", "protocol_stage": "stage_1a"})
    from fedmaq.core.protocol import register_protocol
    from fedmaq.core.run_identity import config_sha256

    manifest["config_sha256"] = config_sha256(config)
    manifest["git"] = {"commit": "fixture-commit", "dirty": False}
    manifest["protocol"] = register_protocol(config, manifest["git"]).as_dict()
    manifest["run"]["loader_used"] = manifest["protocol"]["split"]
    manifest["run"]["split"] = manifest["protocol"]["split"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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

    Asserting directory uniqueness (test_config_and_dispatch.py) proves the runs do not
    overwrite each other. It does not prove analysis.py can still *find* them:
    ``parse_run_directory`` keys on a path length of exactly 7 and reads the group
    from the canonical record, so the ``fedmaq__<variant>`` segment introduced by
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
        "the fedmaq__<variant> segment broke group parsing; the canonical parser "
        "must recover the group from the 7-part path."
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


# --------------------------------------------------------------------------
# Amended primary criterion (Decision 83): accuracy at the minimum common
# cumulative-MB budget. These tests pin the two properties the amendment was
# adopted for -- that the budget is data-determined rather than chosen, and
# that there is no wipeout branch -- plus the concrete pathology that motivated
# it, so a future edit cannot quietly restore bytes-to-target as the selector.
# --------------------------------------------------------------------------


def test_accuracy_at_budget_reads_the_last_round_within_the_budget():
    df = _df([1, 2, 3, 4], [0.10, 0.40, 0.55, 0.60], [10.0, 20.0, 30.0, 40.0])
    # 25 MB buys two rounds, not three: the third round's bytes are already spent
    # by the time its accuracy exists.
    assert accuracy_at_budget(df, 25.0) == pytest.approx(0.40)
    assert accuracy_at_budget(df, 30.0) == pytest.approx(0.55)


def test_accuracy_at_budget_is_none_when_even_round_one_overruns_the_budget():
    df = _df([1, 2], [0.10, 0.40], [10.0, 20.0])
    assert accuracy_at_budget(df, 5.0) is None


def test_sustained_crossing_ignores_a_transient_spike():
    # Touches the floor at round 2 and falls back -- the exact shape that made
    # Formulation 3 the pre-registered winner while finishing last (Decision 82).
    df = _df([1, 2, 3, 4, 5], [0.60, 0.75, 0.66, 0.67, 0.68], [10.0, 20.0, 30.0, 40.0, 50.0])
    assert first_crossing(df, 0.72)[0] == 2
    assert sustained_crossing(df, 0.72, 3) == (None, None)


def test_sustained_crossing_reports_the_start_of_the_first_qualifying_run():
    df = _df([1, 2, 3, 4, 5], [0.60, 0.75, 0.73, 0.74, 0.68], [10.0, 20.0, 30.0, 40.0, 50.0])
    assert sustained_crossing(df, 0.72, 3) == (2, 20.0)


def test_round_completeness_keeps_one_entry_per_run_not_per_directory_name(tmp_path):
    """Identity, rather than a directory name, keeps each arm independently visible.

    The explicit directory override recreates the collision shape that a broken
    output convention would leave behind, while the run identities remain distinct.
    """
    runs = []
    for formulation, n_rounds in ((0, 100), (1, 100), (2, 84)):
        for seed in (1, 2):
            r = _write_run(
                tmp_path,
                "fedmaq",
                formulation,
                seed,
                [0.5] * n_rounds,
                list(range(1, n_rounds + 1)),
            )
            r.job_dir = tmp_path  # force the collision the real output tree produces
            runs.append(r)

    report = round_completeness(runs, expected_round=100)

    assert len(report["runs"]) == 6, "one entry per run, not per directory name"
    assert len(report["incomplete_runs"]) == 2, "both seeds of the short arm"
    assert report["all_complete"] is False


def test_round_completeness_flags_a_run_that_died_before_the_budget(tmp_path):
    """The round check itself, and the identity the report is keyed by.

    Keys are :func:`run_identity`, so a run is looked up by what identifies it
    under ADR-0009 rather than by a string this test happens to know how to
    spell. The literal below is asserted once, here, so that a silent change to
    the serialization -- which would desynchronize the closure certificate from
    the expected-run manifest without failing anything else -- has somewhere to
    fail loudly.
    """
    full = _write_run(tmp_path, "fedmaq", 3, 1, [0.5] * 100, list(range(1, 101)))
    short = _write_run(tmp_path, "fedmaq", 3, 2, [0.5] * 84, list(range(1, 85)))

    report = round_completeness([full, short], expected_round=100)

    assert run_identity(full) == "cifar10|formulation_study|fedmaq|f3|a0.5|f3|s1"
    assert report["all_complete"] is False
    assert report["incomplete_runs"] == [run_identity(short)]
    assert report["runs"][run_identity(full)]["max_round"] == 100
    assert report["runs"][run_identity(short)]["max_round"] == 84
    assert report["duplicate_runs"] == {}


def _without_variant(r):
    """The run key proposed before ``variant`` was restored to it.

    ADR-0009's identity table names ``variant`` as the field that separates cells
    differing only by an override. The three tests below each build a population
    that is a single point under this projection and several runs under
    :func:`run_identity`, so they fail loudly if the field is ever dropped again
    rather than silently auditing a fraction of their input.
    """
    return (r.dataset, r.experiment_group, r.algorithm_config, r.alpha, r.formulation, r.seed)


def test_run_identity_separates_baseline_tuning_variants(tmp_path):
    """Stage 1b sweeps one knob per baseline at a single skew and seed set, so its
    cells differ in nothing a RunRecord records except ``variant`` -- FedProx at
    mu 1.0, 0.1 and 0.01 are one point under every other field. Fifteen cells fold
    onto five without it, and the fold reads as a pass."""
    runs = [
        _write_run(
            tmp_path,
            "fedprox",
            None,
            0,
            [0.5] * 100,
            list(range(1, 101)),
            group=BASELINE_TUNING_GROUP,
            alpha=0.3,
            variant=variant,
        )
        for variant in ("mu1p0", "mu0p1", "mu0p01")
    ]

    report = round_completeness(runs, expected_round=100)

    assert len({_without_variant(r) for r in runs}) == 1
    assert len(report["runs"]) == 3
    assert report["duplicate_runs"] == {}


def test_run_identity_separates_the_three_dataset_grid(tmp_path):
    """§4.5's 105 primary-grid runs live in three matrix files sharing one
    ``experiment_group``, so the dataset is the only thing between CIFAR-10's
    FedMAQ row and CIFAR-100's. This is the collision the restored bundle hit:
    42 entries reported ``all_complete`` for a 105-run grid."""
    runs = [
        _write_run(
            tmp_path,
            "fedmaq",
            2,
            0,
            [0.5] * 100,
            list(range(1, 101)),
            group=GRID_GROUP,
            alpha=1.0,
            dataset=dataset,
        )
        for dataset in ("cifar10", "cifar100", "femnist")
    ]

    report = round_completeness(runs, expected_round=100)

    assert len(report["runs"]) == 3
    assert len({(r.experiment_group, r.algorithm_config, r.alpha, r.seed) for r in runs}) == 1


def test_run_identity_separates_the_grid_from_the_uniform_memory_control(tmp_path):
    """The control arm runs FedMAQ at the grid's own dataset, skews and seeds --
    Decision 75 keeps KD identically configured on both sides so the contrast
    prices the memory ceiling alone. Only ``experiment_group`` tells them apart,
    and pooling them would score the control's rows as the grid's."""
    runs = [
        _write_run(
            tmp_path,
            "fedmaq",
            2,
            0,
            [0.5] * 100,
            list(range(1, 101)),
            group=group,
            alpha=0.1,
        )
        for group in (GRID_GROUP, "uniform_memory_control")
    ]

    report = round_completeness(runs, expected_round=100)

    assert len(report["runs"]) == 2
    assert len({(r.dataset, r.algorithm_config, r.alpha, r.formulation, r.seed) for r in runs}) == 1


def test_round_completeness_reports_duplicate_run_identities(tmp_path):
    """Two directories resolving to one identity must be named, not silently
    reduced. ``report[key] = ...`` in a loop cannot report a collision it has
    already overwritten, so this is unobservable on any shape that does not
    accumulate before it reduces -- the same defect one level up from the one the
    identity keying closes."""
    first = _write_run(tmp_path, "fedmaq", 3, 0, [0.5] * 100, list(range(1, 101)))
    second = _write_run(tmp_path, "fedmaq", 3, 42, [0.5] * 100, list(range(1, 101)))
    # Two distinct output directories carrying one run identity, which is what a
    # re-dispatch under a changed directory convention actually leaves behind.
    second.seed = 0

    report = round_completeness([first, second], expected_round=100)

    assert len(report["runs"]) == 1
    assert report["duplicate_runs"] == {
        run_identity(first): [str(first.job_dir), str(second.job_dir)]
    }


def _expected_runs_manifest():
    with open(REPO_ROOT / "docs" / "freeze" / "expected_runs.json", encoding="utf-8") as f:
        return json.load(f)["groups"]


def test_expected_runs_manifest_holds_105_primary_grid_identities():
    """The ticket's headline arithmetic, proved from tracked config and nothing
    else: no telemetry, no extraction, no fixtures. 42 CIFAR-10 + 42 CIFAR-100 +
    21 FEMNIST, and every one of them a distinct identity."""
    grid = _expected_runs_manifest()["benchmark_grid"]

    assert grid["count"] == 105
    assert len(set(grid["runs"])) == 105
    assert sum(1 for r in grid["runs"] if r.startswith("femnist|")) == 21


def test_closure_certificate_flags_the_42_of_105_shortfall(tmp_path):
    """The restored bundle's failure, reproduced against the tracked manifest.

    Dispatching only ``benchmark_grid.yaml`` delivers CIFAR-10's 42 runs of the
    105 the grid promises. Every one of them logged round 100, so the old audit
    had nothing to report and said so; the shortfall is invisible to any check
    that only asks whether the runs it found are finished.

    The expected side comes from the committed manifest rather than a list
    written here, which is what makes this a test of the serialization contract
    as well as of the diff: the identities are built from RunRecords and the
    manifest's from Hydra composition over conf/matrix/*.yaml, and a formatting
    disagreement on any field would show up as 105 missing and 42 unexpected
    instead of the clean 63.
    """
    runs = [
        _write_run(
            tmp_path,
            algorithm,
            2 if algorithm == "fedmaq" else None,
            seed,
            [0.5] * 100,
            list(range(1, 101)),
            group=GRID_GROUP,
            alpha=alpha,
        )
        for algorithm in (
            "fedavg",
            "fedprox",
            "fedpaq",
            "dadaquant",
            "feddistill",
            "fedkd",
            "fedmaq",
        )
        for alpha in (0.1, 1.0)
        for seed in (0, 42, 123)
    ]
    assert len(runs) == 42

    certificate = closure_certificate(runs, _expected_runs_manifest(), groups=["benchmark_grid"])
    grid = certificate["groups"]["benchmark_grid"]

    assert grid["all_complete"] is True, "every run present did finish -- that is the trap"
    assert grid["observed"] == 42
    assert grid["expected"] == 105
    assert grid["unexpected"] == [], "a serialization mismatch would surface here first"
    assert len(grid["missing"]) == 63
    assert grid["closed"] is False
    assert certificate["all_closed"] is False


def test_closure_certificate_closes_a_complete_fifty_round_group(tmp_path):
    """``pass2_factorial`` runs 50 rounds, and it must be able to close.

    ``phase: explore`` covers both this group and the 100-round formulation
    study, so a certificate that scored every group at R=100 would mark all 26 of
    these runs incomplete on a fully dispatched sweep and put ``all_closed``
    permanently out of reach. A gate that cannot report success is not a gate,
    and it fails in the direction nobody investigates -- the same shape as the
    ungrouped-run case one test below.
    """
    manifest = _expected_runs_manifest()
    runs = []
    for identity in manifest["pass2_factorial"]["runs"]:
        _, group, algorithm, variant, alpha, formulation, seed = identity.split("|")
        runs.append(
            _write_run(
                tmp_path,
                algorithm,
                None if formulation == "fnone" else int(formulation[1:]),
                int(seed[1:]),
                [0.5] * 50,
                list(range(1, 51)),
                group=group,
                alpha=float(alpha[1:]),
                variant=variant,
            )
        )

    certificate = closure_certificate(runs, manifest, groups=["pass2_factorial"])
    factorial = certificate["groups"]["pass2_factorial"]

    assert len(runs) == 26
    assert factorial["expected_round"] == 50
    assert factorial["missing"] == [] and factorial["unexpected"] == []
    assert factorial["closed"] is True
    assert certificate["all_closed"] is True


def test_closure_certificate_names_a_group_the_manifest_does_not_hold():
    """A mistyped or renamed group must say so. Certifying against an absent
    expected set is the one outcome worse than raising: it reports zero missing."""
    with pytest.raises(ValueError, match="no expected-run manifest entry"):
        closure_certificate([], _expected_runs_manifest(), groups=["benchmark-grid"])


def test_closure_certificate_ignores_runs_belonging_to_no_group(tmp_path):
    """A bare scripts/run.py invocation lands outside the canonical 7-part path,
    so ``parse_run_directory`` gives it no group and ``discover_runs`` still finds
    it. Differencing globally would file it as ``unexpected`` and leave the
    certificate red on any checkout that has ever run a smoke test."""
    stray = _write_run(tmp_path, "fedmaq", 2, 0, [0.5] * 100, list(range(1, 101)))
    stray.experiment_group = None

    certificate = closure_certificate([stray], _expected_runs_manifest(), groups=["ablation"])

    assert certificate["groups"]["ablation"]["unexpected"] == []
    assert certificate["groups"]["ablation"]["observed"] == 0


def _pathology_cell(tmp_path):
    """FedAvg finishes at 0.80, so the floor is 0.72.

    Formulation 3 spikes over the floor at round 2 and falls back to 0.68.
    Formulation 1 climbs past it and stays, finishing at 0.74. Both spend the
    same 15 MB. This is the α=1.0 shape from Decision 82 reduced to three
    rounds: the cheap transient crossing beats the better final model under
    bytes-to-target.
    """
    fedavg = [
        _write_run(tmp_path, "fedavg", None, s, [0.5, 0.7, 0.80], [10, 20, 30]) for s in (1, 2, 3)
    ]
    form3 = [
        _write_run(tmp_path, "fedmaq", 3, s, [0.50, 0.75, 0.68], [5, 10, 15]) for s in (1, 2, 3)
    ]
    form1 = [
        _write_run(tmp_path, "fedmaq", 1, s, [0.60, 0.71, 0.74], [5, 10, 15]) for s in (1, 2, 3)
    ]
    return fedavg + form3 + form1


def test_bytes_to_target_prefers_the_transient_spike_over_the_better_model(tmp_path):
    # Characterisation of the superseded rule, not an endorsement: this is what
    # Decision 82 recorded and what Decision 83 amends away from.
    result = select_winner(_pathology_cell(tmp_path))["cifar10_alpha_0.5"]

    assert result["winner"] == 3
    assert (
        result["formulations"][3]["mean_accuracy_r100"]
        < result["formulations"][1]["mean_accuracy_r100"]
    )


def test_iso_byte_selection_prefers_the_better_model_at_equal_spend(tmp_path):
    result = select_winner_iso_byte(_pathology_cell(tmp_path))["cifar10_alpha_0.5"]

    assert result["winner"] == 1
    assert result["budget_mb"] == pytest.approx(15.0)
    assert result["formulations"][1]["mean_accuracy_at_budget"] == pytest.approx(0.74)
    assert result["formulations"][3]["mean_accuracy_at_budget"] == pytest.approx(0.68)


def test_power_mean_degree_selection_keeps_the_p_ladder_distinct(tmp_path):
    runs = []
    winners = {0.1: "p-1", 1.0: "p0"}
    stage = power_mean_stage_one()
    for alpha, winning_variant in winners.items():
        for variant in stage.degrees_by_variant:
            final_accuracy = 0.9 if variant == winning_variant else 0.7
            for seed in stage.seeds:
                run = _write_run(
                    tmp_path,
                    "fedmaq",
                    "power_mean",
                    seed,
                    [0.4, 0.6, final_accuracy],
                    [5.0, 10.0, 15.0],
                    group=POWER_MEAN_DESIGN_GROUP,
                    alpha=alpha,
                    variant=variant,
                )
                run.algorithm_config = stage.algorithm_config
                runs.append(run)

    selection = select_power_mean_degree_iso_byte(runs)
    assert selection["cifar10_alpha_0.1"]["winner"] == "p-1"
    assert selection["cifar10_alpha_1.0"]["winner"] == "p0"
    assert set(selection["cifar10_alpha_0.1"]["groups"]) == set(stage.degrees_by_variant)

    resolution = resolve_power_mean_degree(selection)
    assert resolution["selected_p"] == -1.0
    assert resolution["omega"] == 0.5
    assert resolution["rule"] == "severe-skew tie-break"


def test_power_mean_degree_selection_evaluates_descriptive_controls_safely(tmp_path):
    """Structural controls (F0, F3, F4) remain descriptive and do not fail if out of support."""
    stage = power_mean_stage_one()
    runs = []
    # 7 eligible variants finishing at 15 MB
    for variant in stage.degrees_by_variant:
        acc = 0.85 if variant == "p-1" else 0.70
        for seed in stage.seeds:
            run = _write_run(
                tmp_path,
                "fedmaq",
                "power_mean",
                seed,
                [0.4, 0.6, acc],
                [5.0, 10.0, 15.0],
                group=POWER_MEAN_DESIGN_GROUP,
                alpha=0.1,
                variant=variant,
            )
            run.algorithm_config = stage.algorithm_config
            runs.append(run)

    # Descriptive control F0 finishing early at 10.0 MB (out of support at 15.0 MB)
    for seed in stage.seeds:
        run_f0 = _write_run(
            tmp_path,
            "fedmaq",
            0,
            seed,
            [0.3, 0.5, 0.6],
            [3.0, 6.0, 10.0],
            group=POWER_MEAN_DESIGN_GROUP,
            alpha=0.1,
            variant="resource-only",
        )
        run_f0.algorithm_config = stage.algorithm_config
        runs.append(run_f0)

    # Descriptive control F3
    for seed in stage.seeds:
        run_f3 = _write_run(
            tmp_path,
            "fedmaq",
            3,
            seed,
            [0.4, 0.6, 0.75],
            [5.0, 10.0, 16.0],
            group=POWER_MEAN_DESIGN_GROUP,
            alpha=0.1,
            variant="f3-kappa-0.5",
        )
        run_f3.algorithm_config = stage.algorithm_config
        runs.append(run_f3)

    selection = select_power_mean_degree_iso_byte(runs)["cifar10_alpha_0.1"]
    assert selection["winner"] == "p-1"
    assert "resource-only" not in selection["groups"]
    assert "resource-only" in selection["descriptive_controls"]
    # F0 is out of support at 15 MB budget, so accuracy_at_budget is None
    assert selection["descriptive_controls"]["resource-only"]["accuracy_at_budget"] is None
    # F3 was in support, so its accuracy was interpolated
    assert selection["descriptive_controls"]["f3-kappa-0.5"]["accuracy_at_budget"] is not None


def test_power_mean_omega_selection_joins_stage1a_and_neutral_first_tie_break(tmp_path):
    """Stage 1b joins Stage 1a omega=0.5 runs with Stage 1b omega in {0.25, 0.75}
    and applies neutral-first tie breaking.
    """
    stage_1a = power_mean_stage_one()
    stage_1b = power_mean_stage_one_b()
    selected_p = -1.0
    runs = []

    # Stage 1a runs (omega=0.5 at p=-1, variant="p-1")
    for alpha in (0.1, 1.0):
        for seed in stage_1b.seeds:
            r1a = _write_run(
                tmp_path,
                "fedmaq",
                "power_mean",
                seed,
                [0.4, 0.6, 0.75],
                [5.0, 10.0, 15.0],
                group=POWER_MEAN_DESIGN_GROUP,
                alpha=alpha,
                variant="p-1",
            )
            r1a.algorithm_config = stage_1a.algorithm_config
            r1a.p = selected_p
            runs.append(r1a)

        # Stage 1b runs (omega=0.25 and omega=0.75 at equal accuracy 0.75)
        for w_var in ("omega0.25", "omega0.75"):
            for seed in stage_1b.seeds:
                r1b = _write_run(
                    tmp_path,
                    "fedmaq",
                    "power_mean",
                    seed,
                    [0.4, 0.6, 0.75],
                    [5.0, 10.0, 15.0],
                    group=POWER_MEAN_OMEGA_GROUP,
                    alpha=alpha,
                    variant=w_var,
                )
                r1b.algorithm_config = stage_1b.algorithm_config
                r1b.p = selected_p
                runs.append(r1b)

    selection = select_power_mean_omega_iso_byte(runs, selected_p=selected_p)
    assert selection["cifar10_alpha_0.1"]["selected_p"] == -1.0
    # Neutral-first tie-break selects 0.5 when accuracies are identical
    assert selection["cifar10_alpha_0.1"]["winner"] == 0.5
    assert selection["cifar10_alpha_1.0"]["winner"] == 0.5
    assert selection["cifar10_alpha_0.1"]["ranking"] == [0.5, 0.25, 0.75]

    resolution = resolve_power_mean_omega(selection)
    assert resolution["selected_p"] == -1.0
    assert resolution["selected_omega"] == 0.5
    assert resolution["rule"] == "agreement"


def test_power_mean_omega_selection_rejects_missing_manifest_p(tmp_path):
    """Stage 1b must not infer the selected degree from a variant name alone."""
    stage_1b = power_mean_stage_one_b()
    run = _write_run(
        tmp_path,
        "fedmaq",
        "power_mean",
        0,
        [0.4, 0.6, 0.75],
        [5.0, 10.0, 15.0],
        group=POWER_MEAN_OMEGA_GROUP,
        alpha=0.1,
        variant="omega0.25",
    )
    run.algorithm_config = stage_1b.algorithm_config

    with pytest.raises(ValueError, match="recorded p=None"):
        select_power_mean_omega_iso_byte([run], selected_p=-1.0)


def test_compare_fedpaq_pipeline_iso_byte(tmp_path):
    """compare_fedpaq_pipeline_iso_byte isolates coding pipeline treatment at common budgets."""
    runs = []
    seeds = (0, 42, 123)
    for dataset, alpha in (
        ("cifar10", 0.1),
        ("cifar10", 1.0),
        ("cifar100", 0.1),
        ("cifar100", 1.0),
        ("femnist", 1.0),
    ):
        for seed in seeds:
            # Ordinary fedpaq (higher byte spend, lower accuracy)
            r_fedpaq = _write_run(
                tmp_path,
                "fedpaq",
                None,
                seed,
                [0.4, 0.5, 0.65],
                [10.0, 20.0, 30.0],
                group=GRID_GROUP,
                alpha=alpha,
                dataset=dataset,
            )
            # FedPAQ pipeline (moderate spend, higher accuracy)
            r_pipe = _write_run(
                tmp_path,
                "fedpaq_pipeline",
                None,
                seed,
                [0.4, 0.6, 0.72],
                [6.0, 12.0, 18.0],
                group=FEDPAQ_PIPELINE_GROUP,
                alpha=alpha,
                dataset=dataset,
            )
            # FedMAQ
            r_fedmaq = _write_run(
                tmp_path,
                "fedmaq",
                "power_mean",
                seed,
                [0.4, 0.65, 0.78],
                [5.0, 10.0, 15.0],
                group=GRID_GROUP,
                alpha=alpha,
                dataset=dataset,
            )
            for run in (r_fedpaq, r_pipe, r_fedmaq):
                run.split = "test"
            runs.extend([r_fedpaq, r_pipe, r_fedmaq])

    result = compare_fedpaq_pipeline_iso_byte(runs)
    assert "cifar10_alpha_0.1" in result
    assert "cifar10_alpha_1.0" in result
    cell = result["cifar10_alpha_0.1"]
    assert set(cell["groups"].keys()) == {"fedpaq", "fedpaq_pipeline", "fedmaq"}
    assert cell["winner"] == "fedmaq"
    assert cell["ranking"] == ["fedmaq", "fedpaq_pipeline", "fedpaq"]


def test_power_mean_selectors_reject_test_split_data(tmp_path):
    """Selectors fail closed if given test-split data."""
    stage = power_mean_stage_one()
    run = _write_run(
        tmp_path,
        "fedmaq",
        "power_mean",
        0,
        [0.4, 0.6, 0.8],
        [5.0, 10.0, 15.0],
        group=POWER_MEAN_DESIGN_GROUP,
        alpha=0.1,
        variant="p1",
    )
    run.algorithm_config = stage.algorithm_config
    run.split = "test"

    with pytest.raises(ValueError, match="requires validation-split inputs"):
        select_power_mean_degree_iso_byte([run])

    with pytest.raises(ValueError, match="requires validation-split inputs"):
        select_power_mean_omega_iso_byte([run], selected_p=1.0)


def test_power_mean_recut_expected_set_has_all_96_identities():
    groups = expected_identities(POWER_MEAN_RECUT_MATRICES)

    design = groups[POWER_MEAN_DESIGN_GROUP]
    assert design["count"] == 84
    assert len(set(design["runs"])) == 84
    assert sum("|fpower_mean|" in identity for identity in design["runs"]) == 42

    omega = groups[POWER_MEAN_OMEGA_GROUP]
    assert omega["count"] == 12
    assert len(set(omega["runs"])) == 12


def test_iso_byte_budget_is_the_minimum_final_spend_across_arms(tmp_path):
    """The budget is read off the data. Nobody picks it -- that is what keeps
    the amendment from relocating the free parameter it rejects k-consecutive
    for (Decision 83)."""
    fedavg = [
        _write_run(tmp_path, "fedavg", None, s, [0.5, 0.7, 0.80], [10, 20, 30]) for s in (1, 2, 3)
    ]
    cheap = [
        _write_run(tmp_path, "fedmaq", 0, s, [0.40, 0.50, 0.60], [4, 8, 12]) for s in (1, 2, 3)
    ]
    dear = [
        _write_run(tmp_path, "fedmaq", 2, s, [0.45, 0.55, 0.65], [20, 40, 60]) for s in (1, 2, 3)
    ]

    result = select_winner_iso_byte(fedavg + cheap + dear)["cifar10_alpha_0.5"]

    # min over every compared run's final spend: 12 MB, not 60.
    assert result["budget_mb"] == pytest.approx(12.0)
    # The expensive arm has bought exactly nothing by 12 MB, so it cannot be
    # scored there and does not win by default.
    assert result["formulations"][2]["mean_accuracy_at_budget"] is None
    assert result["winner"] == 0


def test_iso_byte_selection_has_no_wipeout_branch(tmp_path):
    """Every arm failing the accuracy floor collapses the pre-registered rule to
    a null winner and fires Rule 4. The amended criterion still ranks them,
    which is the second reason it replaces bytes-to-target."""
    fedavg = [
        _write_run(tmp_path, "fedavg", None, s, [0.5, 0.7, 0.90], [10, 20, 30]) for s in (1, 2, 3)
    ]
    # Floor is 0.81; nothing here comes close.
    form0 = [
        _write_run(tmp_path, "fedmaq", 0, s, [0.30, 0.40, 0.50], [5, 10, 15]) for s in (1, 2, 3)
    ]
    form3 = [
        _write_run(tmp_path, "fedmaq", 3, s, [0.30, 0.40, 0.55], [5, 10, 15]) for s in (1, 2, 3)
    ]
    runs = fedavg + form0 + form3

    assert select_winner(runs)["cifar10_alpha_0.5"]["winner"] is None

    amended = select_winner_iso_byte(runs)["cifar10_alpha_0.5"]
    assert amended["winner"] == 3
    assert amended["ranking"] == [3, 0]


def test_iso_byte_report_carries_all_three_crossing_verdicts(tmp_path):
    """The robustness table travels with the amended verdict so the superseded
    rule stays auditable (Decision 83)."""
    result = select_winner_iso_byte(_pathology_cell(tmp_path), k_consecutive=2)["cifar10_alpha_0.5"]

    f3 = result["formulations"][3]
    assert f3["qualifies_first_touch"] is True
    assert f3["qualifies_sustained_2"] is False
    assert f3["qualifies_final_round_gate"] is False

    f1 = result["formulations"][1]
    assert f1["qualifies_first_touch"] is True
    assert f1["qualifies_final_round_gate"] is True


def test_round_at_budget_reports_where_the_accuracy_came_from():
    df = _df([1, 2, 3], [0.30, 0.40, 0.55], [10, 20, 30])

    assert round_at_budget(df, 25.0) == 2
    assert accuracy_at_budget(df, 25.0) == pytest.approx(0.40)
    assert round_at_budget(df, 5.0) is None


def test_iso_byte_scores_applies_one_rule_whatever_key_it_is_grouped_on(tmp_path):
    """The core scorer is key-agnostic by design: the formulation study, the
    baseline table and the ablation must not each get their own implementation
    of Decision 83's rule to drift apart on."""
    cheap = [_write_run(tmp_path, "fedmaq", 2, s, [0.40, 0.55, 0.62], [4, 8, 12]) for s in (1, 2)]
    dear = [
        _write_run(tmp_path, "fedpaq", None, s, [0.50, 0.58, 0.70], [10, 20, 30]) for s in (1, 2)
    ]

    scored = iso_byte_scores(cheap + dear, lambda r: r.algorithm_config)["cifar10_alpha_0.5"]

    assert scored["budget_mb"] == pytest.approx(12.0)
    assert scored["budget_set_by"]["group"] == "fedmaq"
    # FedPAQ is one round into a 30 MB run at that budget, not finished.
    assert scored["groups"]["fedpaq"]["seeds"][1]["round_at_budget"] == 1
    assert scored["groups"]["fedpaq"]["mean_accuracy_at_budget"] == pytest.approx(0.50)
    assert scored["groups"]["fedmaq"]["mean_accuracy_at_budget"] == pytest.approx(0.62)
    assert scored["winner"] == "fedmaq"
    # The seed spread travels with the mean: at a mid-climb round it is the only
    # thing that says whether the number is a plateau or a lucky sample.
    assert scored["groups"]["fedpaq"]["accuracy_at_budget"]["sd"] == pytest.approx(0.0)


def _grid_pair(tmp_path, fedmaq_mbs, baseline_mbs, baseline="fedpaq", seeds=(1, 2, 3)):
    fedavg = [
        _write_run(tmp_path, "fedavg", None, s, [0.5, 0.7, 0.80], [40, 80, 120]) for s in (1, 2, 3)
    ]
    fedmaq = [
        _write_run(tmp_path, "fedmaq", 2, s, [0.50, 0.65, 0.78], fedmaq_mbs, group=GRID_GROUP)
        for s in (1, 2, 3)
    ]
    baselines = [
        _write_run(tmp_path, baseline, None, s, [0.45, 0.60, 0.70], baseline_mbs) for s in seeds
    ]
    return fedavg + fedmaq + baselines


def test_iso_byte_baseline_delta_is_read_at_the_budget_not_at_equal_rounds(tmp_path):
    """The equal-round delta charges FedMAQ for accuracy and credits it nothing
    for the bytes it saved, which is the mechanism under study (Decision 83)."""
    runs = _grid_pair(tmp_path, fedmaq_mbs=[5, 10, 15], baseline_mbs=[10, 40, 60])

    row = compare_to_baselines_iso_byte(runs, frozen_formulation=2)["cifar10_alpha_0.5_vs_fedpaq"]

    assert row["budget_mb"] == pytest.approx(15.0)
    assert row["budget_set_by"]["group"] == "fedmaq"
    # FedPAQ has bought one round by 15 MB; FedMAQ has finished.
    assert row["per_seed"][1]["baseline"]["round_at_budget"] == 1
    assert row["delta_at_budget"]["mean"] == pytest.approx(0.78 - 0.45)
    # The superseded rule, on the same runs, reads the gap as a third of that.
    equal_round = compare_to_baselines(runs, select_winner(runs))["cifar10_alpha_0.5_vs_fedpaq"]
    assert equal_round["mean_delta"] == pytest.approx(0.78 - 0.70)


def test_iso_byte_baseline_budget_is_per_row_not_per_table(tmp_path):
    """A baseline that transmits less than FedMAQ sets the budget itself, and
    FedMAQ is then scored mid-run for that row alone. Nothing in this table may
    be averaged across rows."""
    runs = _grid_pair(tmp_path, fedmaq_mbs=[5, 20, 30], baseline_mbs=[3, 6, 9])

    row = compare_to_baselines_iso_byte(runs, frozen_formulation=2)["cifar10_alpha_0.5_vs_fedpaq"]

    assert row["budget_mb"] == pytest.approx(9.0)
    assert row["budget_set_by"]["group"] == "baseline"
    assert row["per_seed"][1]["fedmaq"]["round_at_budget"] == 1
    assert row["delta_at_budget"]["mean"] == pytest.approx(0.50 - 0.70)


def test_iso_byte_baseline_intersects_seeds_before_pairing(tmp_path):
    """Three FedMAQ seeds against two baseline seeds is not a method effect."""
    runs = _grid_pair(tmp_path, [5, 10, 15], [10, 40, 60], seeds=(1, 2))

    row = compare_to_baselines_iso_byte(runs, frozen_formulation=2)["cifar10_alpha_0.5_vs_fedpaq"]

    assert row["seeds"] == [1, 2]
    assert row["delta_at_budget"]["n"] == 2
    assert row["fedmaq_accuracy_at_budget"]["n"] == 2


def test_fedavg_at_fedmaq_budget_scores_the_control_mid_climb(tmp_path):
    """The number chapters 5 and 6 are framed around. FedAvg is scored at the
    frozen formulation's spend, which lands it short of its own final round."""
    fedavg = [
        _write_run(tmp_path, "fedavg", None, s, [0.30, 0.55, 0.62], [10, 40, 120])
        for s in (0, 42, 123)
    ]
    frozen = [
        _write_run(tmp_path, "fedmaq", 2, s, [0.20, 0.40, 0.52], [10, 20, 30]) for s in (0, 42, 123)
    ]
    # A losing formulation from the same study must not enter the comparison.
    other = [
        _write_run(tmp_path, "fedmaq", 4, s, [0.10, 0.15, 0.18], [5, 10, 15]) for s in (0, 42, 123)
    ]

    cell = fedavg_at_fedmaq_budget(fedavg + frozen + other, formulation=2)["cifar10_alpha_0.5"]

    assert sorted(cell["groups"]) == ["fedavg", "fedmaq_formulation_2"]
    assert cell["budget_mb"] == pytest.approx(30.0)
    assert cell["groups"]["fedavg"]["seeds"][0]["round_at_budget"] == 1
    assert cell["groups"]["fedavg"]["mean_accuracy_at_budget"] == pytest.approx(0.30)
    assert cell["mean_delta_vs_fedavg"] == pytest.approx(0.52 - 0.30)


def test_ablation_iso_byte_scores_every_arm_at_the_stingiest_arm_budget(tmp_path):
    """The §4.3.7 arms differ in spend, so an equal-round table credits each one
    with accuracy it bought with bandwidth. The budget is the cheapest arm's
    total, including when that is not the anchor -- Configuration 7 gets scored
    mid-run here, which is the rule working rather than an exception to it."""
    table = build_ablation_table(_full_ablation_grid(tmp_path))
    assert table["iso_byte"]["all_complete"] is True
    cell = table["iso_byte"]["cells"]["cifar10_alpha_0.1"]

    # Configuration 5 (fedmaq_no_kd) is the cheapest arm in the fixture at 36 MB.
    assert cell["budget_mb"] == pytest.approx(36.0)
    assert cell["budget_set_by"]["group"] == 5
    assert sorted(cell["groups"]) == [1, 2, 3, 4, 5, 6, 7, 8]
    # Configuration 1 spends 120 MB across rounds 1/50/100, so the uncompressed
    # control has reached round 1 on the budget the compressed arms finish on.
    assert cell["groups"][1]["seeds"][0]["round_at_budget"] == 1
    assert cell["groups"][7]["seeds"][0]["round_at_budget"] == 50
    assert cell["groups"][5]["seeds"][0]["round_at_budget"] == 100


def test_ablation_iso_byte_refuses_a_budget_read_off_a_half_finished_arm(tmp_path):
    """An arm at round 50 of 100 reports half its true spend, sets B for the
    whole cell, and drags every other arm back to it. Mid-sweep is the ordinary
    way to hit this, so the table has to say so rather than look plausible."""
    runs = _full_ablation_grid(tmp_path)
    truncated = next(
        r
        for r in runs
        if r.algorithm_config == "fedmaq_no_resource" and r.alpha == 0.1 and r.seed == 0
    )
    # Half the rounds, and so half the spend: cheaper than every finished arm.
    _df([1, 25, 50], [0.35, 0.45, 0.55], [4.5, 9.0, 18.0]).to_csv(truncated.csv_path, index=False)

    table = build_ablation_table(runs)

    assert table["iso_byte"]["all_complete"] is False
    assert table["iso_byte"]["incomplete_runs"] == [run_identity(truncated)]
    assert not table["parity"]["attributable"]
    assert any("never logged round 100" in v for v in table["parity"]["violations"])


# --- Stage 1b: baseline matched-tuning margin (§4.3.2, Decision 81) ---


def _tuning_run(
    tmp_path, algorithm, variant, seed, final_acc, alpha=0.3, group=BASELINE_TUNING_GROUP
):
    """One Stage 1b run. The cells of a baseline's sweep differ in nothing the
    record holds *except* ``variant`` -- same algorithm, config name, group and
    skew -- which is why the field exists."""
    return write_run(
        tmp_path,
        algorithm,
        None,
        seed,
        [0.3, 0.5, final_acc],
        [10, 20, 30],
        rounds=[1, 50, 100],
        group=group,
        alpha=alpha,
        variant=variant,
        phase="explore",
    )


def _cell(tmp_path, algorithm, variant, accs, group=BASELINE_TUNING_GROUP):
    return [
        _tuning_run(tmp_path, algorithm, variant, seed, acc, group=group)
        for seed, acc in zip(range(len(accs)), accs, strict=True)
    ]


def _five_seeds_with(mean, sigma):
    """Five accuracies whose fmean and stdev are exactly ``mean`` and ``sigma``.

    Decision 81 recorded each reference cell's summary statistics but not its raw
    per-seed values, so the fixtures reconstruct a cell consistent with what was
    published rather than inventing unrelated numbers. Spacing ``[-2d, -d, 0, d,
    2d]`` has stdev ``d*sqrt(2.5)``.
    """
    d = sigma / math.sqrt(2.5)
    return [mean + k * d for k in (-2, -1, 0, 1, 2)]


def test_baseline_tuning_margin_reproduces_decision_81_fedprox(tmp_path):
    """FedProx's two challengers both clear; the larger delta is adopted.

    Reconstructed from Decision 81's own published figures: reference mu=1.0 at
    0.5423 +- 0.0124 (n=5), challengers mu=0.01 at 0.5690 and mu=0.1 at 0.5677.
    """
    runs = _cell(tmp_path, "fedprox", "mu1p0", _five_seeds_with(0.5423, 0.0124))
    runs += _cell(tmp_path, "fedprox", "mu0p01", [0.5690] * 3)
    runs += _cell(tmp_path, "fedprox", "mu0p1", [0.5677] * 3)

    cell = baseline_tuning_margin(runs)["baselines"]["fedprox"]

    assert cell["reference"]["mean"] == pytest.approx(0.5423)
    assert cell["reference"]["sigma"] == pytest.approx(0.0124)
    assert cell["margin"] == pytest.approx(0.0175, abs=5e-5)
    assert cell["challengers"]["mu0p01"]["delta"] == pytest.approx(0.0267, abs=5e-5)
    assert cell["challengers"]["mu0p1"]["delta"] == pytest.approx(0.0254, abs=5e-5)
    assert cell["challengers"]["mu0p01"]["clears_margin"]
    assert cell["challengers"]["mu0p1"]["clears_margin"]
    # Both clear, so the tie-break decides -- and it decides on delta, which is
    # the only reason conf/algorithm/fedprox.yaml ships 0.01 rather than 0.1.
    assert cell["adopted_variant"] == "mu0p01"
    assert cell["retained_shipped_value"] is False
    assert {row["variant"]: row["is_adopted"] for row in cell["table"]} == {
        "mu1p0": False,
        "mu0p01": True,
        "mu0p1": False,
    }


def test_baseline_tuning_margin_reproduces_decision_81_fedpaq_retention(tmp_path):
    """FedPAQ's challengers both score *higher* and neither is adopted.

    The retention case is the one the rule exists for: q=4 and q=16 beat the
    shipped q=8 by 0.40pp and 0.38pp against a 3.08pp margin. Adopting a
    higher-scoring arm here is exactly the manufactured winner §4.3.1 refuses.
    """
    runs = _cell(tmp_path, "fedpaq", "q8", _five_seeds_with(0.5629, 0.0218))
    runs += _cell(tmp_path, "fedpaq", "q4", [0.5669] * 3)
    runs += _cell(tmp_path, "fedpaq", "q16", [0.5667] * 3)

    cell = baseline_tuning_margin(runs)["baselines"]["fedpaq"]

    assert cell["margin"] == pytest.approx(0.0308, abs=5e-5)
    assert not any(c["clears_margin"] for c in cell["challengers"].values())
    assert all(c["delta"] > 0 for c in cell["challengers"].values())
    assert cell["adopted_variant"] is None
    assert cell["retained_shipped_value"] is True
    # Nothing clears, so the reference retains the mark rather than every row
    # reading False -- the distinction "no data" and "verdict: keep shipped" must
    # not collapse to the same table.
    assert {row["variant"]: row["is_adopted"] for row in cell["table"]} == {
        "q8": True,
        "q4": False,
        "q16": False,
    }


def test_baseline_tuning_margin_refuses_an_underpowered_reference_cell(tmp_path):
    """A margin estimated from two runs would still decide two adoptions."""
    runs = _cell(tmp_path, "fedkd", "t0p95", [0.40, 0.41])
    runs += _cell(tmp_path, "fedkd", "t0p85", [0.50] * 3)

    cell = baseline_tuning_margin(runs)["baselines"]["fedkd"]

    assert "error" in cell
    assert "adopted_variant" not in cell


def test_exploration_noise_margin_cannot_analyse_stage_1b(tmp_path):
    """The defect baseline_tuning_margin exists to fix, pinned.

    conf/matrix/baseline_tuning.yaml named exploration_noise_margin as its
    analyser until 2026-08-06. That function filters ``algorithm == "fedmaq"``,
    so it reports a fully-completed 55-run stage as no runs at all -- silently, if
    nobody reads the error string.
    """
    runs = _cell(tmp_path, "fedprox", "mu1p0", [0.54] * 5)

    assert "error" in exploration_noise_margin(runs, experiment_group=BASELINE_TUNING_GROUP)
    assert baseline_tuning_margin(runs)["baselines"]["fedprox"]["reference"]["n"] == 5


def test_baseline_tuning_margin_flags_a_contaminating_skew(tmp_path):
    """§4.3.1 holds every exploration stage at the held-out alpha."""
    runs = _cell(tmp_path, "fedprox", "mu1p0", [0.54] * 5)
    runs.append(_tuning_run(tmp_path, "fedprox", "mu1p0", 99, 0.54, alpha=0.1))

    result = baseline_tuning_margin(runs)

    assert result["other_skews_present"] == [0.1]
    assert result["baselines"]["fedprox"]["reference"]["n"] == 5


def test_wide_baseline_tuning_report_marks_values_and_writes_curves(tmp_path):
    runs = []
    for variant, value in (
        ("qmax4", 0.51),
        ("qmax6", 0.52),
        ("qmax8", 0.53),
        ("qmax16", 0.54),
    ):
        seeds = (0, 42, 123, 7, 21)
        runs.extend(
            _tuning_run(
                tmp_path,
                "fedmaq",
                variant,
                seed,
                value + seed / 100000,
                group=BASELINE_TUNING_WIDE_GROUP,
            )
            for seed in seeds
        )

    report = baseline_tuning_margin(runs, experiment_group=BASELINE_TUNING_WIDE_GROUP)
    cell = report["baselines"]["fedmaq"]
    assert cell["knob"] == "q_max"
    assert cell["paper_default_variant"] is None
    assert "no source-paper default" in cell["paper_default_note"]
    assert cell["shipped_adopted_variant"] == "qmax16"
    assert cell["adopted_variant"] is None
    assert cell["retained_shipped_value"] is True
    assert [entry["value"] for entry in cell["table"]] == [4, 6, 8, 16]
    assert sum(entry["is_adopted"] for entry in cell["table"]) == 1
    assert len(cell["curves"]) == 5

    paths = write_baseline_tuning_plots(report, tmp_path / "plots")
    assert paths == [tmp_path / "plots" / "fedmaq_q_max.png"]
    assert paths[0].is_file()


def test_wide_baseline_tuning_metadata_does_not_invent_source_defaults():
    specs = baseline_tuning_specs(BASELINE_TUNING_WIDE_GROUP)

    assert specs["fedpaq"]["paper_default_variant"] is None
    assert "level counts" in specs["fedpaq"]["paper_default_note"]
    assert specs["fedkd"]["paper_default_variant"] is None
    assert "does not report numeric defaults" in specs["fedkd"]["paper_default_note"]
    assert specs["fedmaq"]["paper_default_variant"] is None
    assert "no source-paper default" in specs["fedmaq"]["paper_default_note"]


def test_wide_baseline_tuning_report_withholds_partial_verdict(tmp_path):
    runs = _cell(
        tmp_path,
        "fedmaq",
        "qmax16",
        [0.54] * 5,
        group=BASELINE_TUNING_WIDE_GROUP,
    )
    runs += _cell(
        tmp_path,
        "fedmaq",
        "qmax4",
        [0.60] * 3,
        group=BASELINE_TUNING_WIDE_GROUP,
    )

    cell = baseline_tuning_margin(runs, experiment_group=BASELINE_TUNING_WIDE_GROUP)["baselines"][
        "fedmaq"
    ]

    assert "error" in cell
    assert "adopted_variant" not in cell
    assert cell["missing_variants"] == ["qmax6", "qmax8"]


def test_run_directory_parser_round_trips_the_variant(tmp_path):
    job_dir = tmp_path / get_canonical_output_dir(
        "explore",
        "cifar10",
        "mobilenetv2",
        "baseline_tuning",
        "fedprox",
        "dirichlet_alpha_0.3",
        0,
        variant="mu0p01",
    )
    assert parse_run_directory(job_dir, tmp_path).variant == "mu0p01"


def test_run_directory_parser_returns_empty_variant_when_matrix_set_none(tmp_path):
    job_dir = tmp_path / get_canonical_output_dir(
        "formal", "cifar10", "mobilenetv2", "ablation", "fedmaq_no_kd", "dirichlet_alpha_0.1", 0
    )
    assert parse_run_directory(job_dir, tmp_path).variant == ""
