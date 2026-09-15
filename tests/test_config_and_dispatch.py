"""Configuration, matrix-dispatch, and run-provenance contract tests."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import MissingMandatoryValue, OmegaConf
from torch.utils.data import TensorDataset

CONF_DIR = str((Path(__file__).parent.parent / "conf").resolve())

ALGORITHM_CONFIGS = [
    "fedavg",
    "fedprox",
    "fedpaq",
    "fedpaq_pipeline",
    "power_mean",
    "dadaquant",
    "fedmd",
    "fedkd",
    "fedavg_kd",
    "fedmaq",
    "fedmaq_no_kd",
    "fedmaq_no_state",
    "fedmaq_no_data",
    "fedmaq_no_resource",
    "fedmaq_no_refinements",
    "feddistill",
    "cfd",
]


@pytest.fixture
def mock_dataset(monkeypatch):
    """Mock torchvision dataset loading with 100 MNIST-like samples."""
    mock_data = torch.randn(100, 1, 28, 28)
    mock_labels = torch.randint(0, 10, (100,))
    mock_ds = TensorDataset(mock_data, mock_labels)
    mock_ds.targets = mock_labels
    monkeypatch.setattr("fedmaq.core.partitioning.load_dataset", lambda name, train=True: mock_ds)
    return mock_ds


def _algorithm_cfg(algorithm):
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"algorithm={algorithm}"])
    return OmegaConf.to_container(cfg.algorithm, resolve=True)


REFINEMENTS = ("soft_voting", "ema_student", "grad_norm_ema")


def _frozen_refinements():
    """The refinement mechanisms ``fedmaq.yaml`` currently ships as active.

    This is the freeze itself: conf/matrix/pass3_freeze_confirm.yaml writes the
    surviving set here, and everything downstream is derived from it rather than
    restated. Before that write it holds the pre-freeze defaults, which is why
    nothing below asserts a specific membership -- the point is that the arms
    agree with whatever is frozen, not that a particular set was frozen.
    """
    full = _algorithm_cfg("fedmaq")
    return {flag for flag in REFINEMENTS if full[flag]}


def _ablation_arm_diffs():
    frozen = _frozen_refinements()
    # Dual-formulation compatibility bridge: fedmaq_no_data and fedmaq_no_state
    # express signal removal for both Formulation 2 (gamma2/gamma1) and the power-mean
    # family (omega per ADR-0021 D4), so the removal holds under both formulations.
    return {
        "fedmaq_no_resource": {"resource_aware"},
        "fedmaq_no_data": {"gamma2", "omega"},
        "fedmaq_no_state": {"gamma1", "omega"},
        "fedmaq_no_kd": {"kd_epochs"} | ({"soft_voting"} & frozen),
        "fedmaq_no_refinements": set(frozen),
    }


@pytest.mark.parametrize("arm,expected_diff", sorted(_ablation_arm_diffs().items()))
def test_ablation_arm_is_one_removal_from_full_fedmaq(arm, expected_diff):
    """Each §4.3.7 arm must differ from full FedMAQ in exactly its declared keys.

    Both directions matter. An *extra* difference makes the arm's delta
    unattributable; a *missing* one means the arm no longer removes what it
    claims to and is silently a duplicate of Configuration 7.
    """
    full = _algorithm_cfg("fedmaq")
    arm_cfg = _algorithm_cfg(arm)

    assert arm_cfg.keys() == full.keys(), (
        f"{arm} has a different key set than fedmaq.yaml; parity is only "
        f"checkable when both carry every knob explicitly. "
        f"Only in arm: {arm_cfg.keys() - full.keys()}. "
        f"Missing from arm: {full.keys() - arm_cfg.keys()}"
    )

    actual_diff = {k for k in full if full[k] != arm_cfg[k]}
    assert actual_diff == expected_diff, (
        f"{arm} is not one removal from full FedMAQ.\n"
        f"  unexpected differences: {actual_diff - expected_diff}\n"
        f"  declared but absent:    {expected_diff - actual_diff}\n"
        f"If a new difference is intentional, it must be justified in "
        f"chapter_4.tex §4.3.7 first, then declared in ABLATION_ARM_DIFFS."
    )


def test_ablation_arms_share_one_refinement_layer():
    """§4.3.7 requires an identical refinement layer across every arm.

    The exceptions are recorded rather than silently disabled, so this asserts
    the exception list itself: only the two mechanisms that have no signal to act
    on may deviate, and only in the arms that remove that signal.
    """
    full = _algorithm_cfg("fedmaq")
    for arm in ("fedmaq_no_resource", "fedmaq_no_data", "fedmaq_no_state"):
        arm_cfg = _algorithm_cfg(arm)
        for flag in REFINEMENTS:
            assert arm_cfg[flag] == full[flag], (
                f"{arm} deviates from the shared refinement layer on {flag}; "
                f"§4.3.7 requires the difference between arms to be the "
                f"awareness signal alone."
            )

    fedavg_kd = _algorithm_cfg("fedavg_kd")
    # ema_student is quantization-independent, so the no-quantization arm carries
    # whatever the freeze decided. Asserted against fedmaq.yaml rather than a
    # literal: pinning it to True here would mean a freeze that drops ema_student
    # leaves Configuration 6 as the one arm still running it, turning §4.3.7's
    # shared layer into a second difference on the arm that can least afford one.
    assert fedavg_kd["ema_student"] == full["ema_student"], (
        "fedavg_kd (Configuration 6) must carry the frozen ema_student setting; "
        "it removes quantization, which ema_student does not depend on."
    )
    # soft_voting and grad_norm_ema stay literal. Both act on a quantization
    # signal this arm never produces, so they are INAPPLICABLE here regardless of
    # what the freeze decides -- deriving them would assert nothing.
    assert fedavg_kd["soft_voting"] is False
    assert fedavg_kd["grad_norm_ema"] is False


def test_configuration_8_exists_only_while_there_is_a_layer_to_remove():
    """conf/matrix/pass3_freeze_confirm.yaml pre-registers the empty surviving set
    as a real outcome: if nothing clears the margin at R=100, FedMAQ freezes
    unrefined and Configuration 8 has nothing left to remove.

    An arm that removes nothing is not an ablation arm -- it is a second copy of
    Configuration 7 dispatched under a different label, and its delta would be
    pure noise reported as a component's contribution. This is the check that
    makes the pre-registered branch enforceable rather than merely written down.
    """
    frozen = _frozen_refinements()
    arm_present = any(run["alg"] == "fedmaq_no_refinements" for run in _matrix("ablation")["runs"])

    if frozen:
        assert arm_present, (
            f"fedmaq.yaml freezes {sorted(frozen)}, so §4.3.7's Configuration 8 "
            "must be dispatched to price that layer."
        )
    else:
        assert not arm_present, (
            "fedmaq.yaml carries no active refinements, so Configuration 8 "
            "removes nothing and duplicates Configuration 7. Drop it from "
            "conf/matrix/ablation.yaml and drop the chapter 6 contribution "
            "bullet that rests on its contrast."
        )


def test_frozen_config_snapshot_is_current():
    """docs/freeze/resolved_configs.yaml must match what the configs compose to.

    The §4.3.7 arms inherit fedmaq.yaml and state only their own removal, so
    reading an arm no longer tells you what it runs. That generated snapshot is
    what chapter 6 §6.2's "recoverable frozen configuration" promise now rests on,
    and a stale one is worse than none: it describes a configuration the tag does
    not contain. The freeze runbook says to regenerate it, but an instruction in a
    runbook is not a guard -- this is.
    """
    import subprocess
    import sys

    repo_root = Path(__file__).parent.parent
    result = subprocess.run(
        [sys.executable, "scripts/dump_frozen_configs.py", "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{result.stdout}{result.stderr}\n"
        "Run `uv run python scripts/dump_frozen_configs.py` and commit the result."
    )


def _assert_expected_runs_snapshot_is_current(*mode_flags: str):
    import subprocess
    import sys

    repo_root = Path(__file__).parent.parent
    result = subprocess.run(
        [sys.executable, "scripts/dump_expected_runs.py", *mode_flags, "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    command = " ".join(["uv run python scripts/dump_expected_runs.py", *mode_flags])
    assert result.returncode == 0, (
        f"{result.stdout}{result.stderr}\nRun `{command}` and commit the result."
    )


def test_expected_runs_snapshot_is_current():
    """docs/freeze/expected_runs.json must match what conf/matrix/*.yaml promises.

    It is the expected side of the closure certificate, so a stale snapshot does
    not merely go out of date -- it silently redefines what "complete" means. A
    matrix that gains a seed and a manifest that does not is a grid the
    certificate certifies as closed while it is short.

    `just check` and CI invoke the generator in this mode only. Every other mode
    writes a snapshot this one does not read, so each is covered by its own test
    below or by nothing at all.
    """
    _assert_expected_runs_snapshot_is_current()


def test_power_mean_recut_expected_runs_snapshot_is_current():
    """The re-cut's separate expected set must not silently redefine v1 closure."""
    _assert_expected_runs_snapshot_is_current("--power-mean-recut")


def test_baseline_tuning_wide_expected_runs_snapshot_is_current():
    """The widened tuning stage has its own expected set, outside v1 closure."""
    _assert_expected_runs_snapshot_is_current("--baseline-tuning-wide")


def test_fedpaq_pipeline_expected_runs_snapshot_is_current():
    """The three FedPAQ-pipeline datasets share one expected set, outside v1 closure."""
    _assert_expected_runs_snapshot_is_current("--fedpaq-pipeline")


def test_memory_sensitivity_expected_runs_snapshot_is_current():
    """The memory-sensitivity sweep has its own expected set, outside v1 closure."""
    _assert_expected_runs_snapshot_is_current("--memory-sensitivity")


def test_each_reportable_arm_carries_one_regime():
    """``identity_key`` omits ``phase`` and ``post_process``, which ADR-0009 also
    lists as identity fields. That is admissible only while both are functions of
    ``(experiment_group, algorithm_config)`` -- two fields the key does carry --
    and this is what makes that a checked property rather than an assumption.

    Not at group granularity: the primary grid runs FedMAQ with the §4.3
    post-processing pipeline and its six baselines without it, so ``benchmark_grid``
    spans both regimes and only the arm resolves them. A future matrix that
    dispatched one arm into two regimes under one group would make the omitted
    fields load-bearing again, and the certificate would start pairing two real
    runs onto one identity.

    ``total_rounds`` is asserted here for a different reason: the closure
    certificate scores each group against the budget the manifest declares, so a
    group spanning two budgets would leave half of it certified against the wrong
    one. It is a per-matrix property and not a per-phase one -- ``explore`` covers
    both the 50-round factorial passes and the 100-round formulation study.
    """
    manifest = json.loads(
        (Path(__file__).parent.parent / "docs" / "freeze" / "expected_runs.json").read_text(
            encoding="utf-8"
        )
    )
    offenders = {
        f"{group}/{arm}": regime
        for group, body in manifest["groups"].items()
        for arm, regime in body["regimes"].items()
        if any(len(regime[field]) != 1 for field in ("phases", "post_process", "total_rounds"))
    }
    assert not offenders, (
        f"{offenders} dispatch one arm under one group into more than one phase, "
        "post-processing regime or round budget. scripts/common.identity_key leaves "
        "the first two out of the run key on the grounds that this cannot happen, and "
        "closure_certificate scores a group against the third; restore them there, or "
        "split the group."
    )

    budgets = {
        group: {n for regime in body["regimes"].values() for n in regime["total_rounds"]}
        for group, body in manifest["groups"].items()
    }
    assert budgets["pass2_factorial"] == {50} and budgets["formulation_study"] == {100}, (
        "Both groups carry phase 'explore' and they disagree on the round budget, "
        "which is why the certificate reads it per group instead of assuming 100. "
        f"Got {budgets}."
    )


def test_configuration_8_can_express_any_freeze():
    """fedmaq_no_refinements must hold every mechanism off, not merely the ones
    that happen to be frozen today. Its job is to be the layer's absence, so a
    freeze that later turns on a mechanism this arm leaves enabled would silently
    make Configuration 8 a partial removal."""
    arm = _algorithm_cfg("fedmaq_no_refinements")
    still_on = [flag for flag in REFINEMENTS if arm[flag]]
    assert not still_on, (
        f"fedmaq_no_refinements leaves {still_on} enabled. Configuration 8 must "
        "disable all three so it removes whatever the freeze turns on."
    )


def _matrix(name):
    return OmegaConf.to_container(
        OmegaConf.load(Path(CONF_DIR).parent / "conf" / "matrix" / f"{name}.yaml"),
        resolve=True,
    )


def _post_process_overrides(matrix):
    """Map run label -> the post_process value that run's overrides set, if any."""
    found = {}
    for run in matrix["runs"]:
        for override in run.get("overrides") or []:
            key, _, value = override.partition("=")
            if key.strip() == "algorithm.post_process":
                found[run["label"]] = value.strip().lower() == "true"
    return found


# §4.5's 105 primary-grid runs are split across three matrix files because
# run_matrix.py reads ``dataset`` as a scalar and because FEMNIST differs on
# client count, heterogeneity, and model too. They share one ``experiment_group``
# so analysis.py reads them as a single grid.
PRIMARY_GRID_MATRICES = (
    "benchmark_grid",
    "benchmark_grid_cifar100",
    "benchmark_grid_femnist",
)


@pytest.mark.parametrize("name", PRIMARY_GRID_MATRICES)
def test_primary_grid_turns_the_post_process_pipeline_on(name):
    """§4.3 applies difference coding, error compensation, and lossless encoding
    to FedMAQ for primary benchmarking.

    ``post_process`` ships false in every conf/algorithm/*.yaml, deliberately, so
    the only thing standing between the manuscript's promise and a grid that
    silently reports uncompressed payload numbers is this override. A comment is
    not a guard; a plausible-looking communication table is exactly the kind of
    defect nobody questions after the fact.

    Parametrized over all three primary-grid files rather than checked on
    CIFAR-10 alone. The CIFAR-100 and FEMNIST halves had no matrix file at all
    until 2026-07-30 and were dispatched from a commented raw ``--multirun`` in
    conf/config.yaml that omitted this flag, so FedMAQ's communication rows would
    have been measured pipeline-free on two of three datasets and pipeline-on on
    the third, in one table, with nothing failing.
    """
    enabled = _post_process_overrides(_matrix(name))
    assert enabled.get("fedmaq") is True, (
        f"conf/matrix/{name}.yaml must run FedMAQ with "
        "algorithm.post_process=true. Without it the primary grid measures "
        "communication without the §4.3 pipeline and reports it as if it had."
    )


def _protocol_document():
    return OmegaConf.to_container(
        OmegaConf.load(Path(CONF_DIR).parent / "conf" / "protocol" / "replacement-v1.yaml"),
        resolve=True,
    )


def _comparable(value):
    """Compare a preregistered support entry against an override token by value.

    Overrides reach ``expand_matrix`` as ``"algorithm.p=-0.5"`` strings while the
    support lists hold YAML scalars, so ``"0" in [0, 0.5, "min"]`` is False for a
    degree that was preregistered.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def test_registered_matrices_only_dispatch_preregistered_p_and_omega():
    """Every `p` and `omega` a preregistered matrix dispatches lies in its support.

    `selection_domains` is the independent dispatch authority: it is hashed into
    `preregistration_sha256`, so regenerating a ledger from a widened matrix cannot
    make the widening look preregistered. Nothing until now compared it against what
    the matrices actually expand to.

    Scoped to `p` and `omega`, which are the axes selection ranges over. The other
    two domains do not describe this dispatch surface and asserting them here would
    fail on correct configuration: `seeds` is stage-scoped -- `baseline_tuning_wide`
    runs five by design, pinned at `scripts/stage_manifest.py` -- and `q_ladder` names
    a quantity no matrix sweeps, since matrices set `algorithm.q` (FedPAQ's fixed
    width, which reaches 32) and `algorithm.q_max` (FedMAQ's ceiling), documented as
    different quantities in `conf/algorithm/dadaquant.yaml`.

    Arms carrying no `p` are outside the power-mean sweep by construction:
    formulations 0, 3, and 4 parameterise resources, `kappa`, or `tau_*` instead. The
    `???` sentinel is admitted because it is the unresolved state
    `test_stage_1b_p_is_explicit_and_fails_closed_until_selection` requires, and
    dispatch already refuses it.
    """
    from scripts.common import expand_matrix

    protocol = _protocol_document()
    domains = protocol["selection_domains"]
    supports = {
        "algorithm.p": {_comparable(value) for value in domains["p_support"]},
        "algorithm.omega": {_comparable(value) for value in domains["omega_support"]},
    }

    checked = dict.fromkeys(supports, 0)
    for name in sorted(protocol["matrix_contracts"]):
        for spec in expand_matrix(_matrix(name), name):
            for override in spec["overrides"]:
                key, _, value = override.partition("=")
                support = supports.get(key.strip())
                if support is None or value.strip() == "???":
                    continue
                checked[key.strip()] += 1
                assert _comparable(value.strip()) in support, (
                    f"conf/matrix/{name}.yaml run {spec['label']!r} sets "
                    f"{key.strip()}={value.strip()}, which is outside the "
                    f"preregistered support in conf/protocol/replacement-v1.yaml. "
                    "Widen the support there and re-declare the preregistration, or "
                    "drop the arm -- a ledger regenerated from this matrix would "
                    "otherwise report an unpreregistered treatment as preregistered."
                )

    unexercised = sorted(key for key, count in checked.items() if not count)
    assert not unexercised, (
        f"No registered matrix dispatched {', '.join(unexercised)}, so this test "
        "asserted nothing about that axis while passing on the other. Both skip "
        "branches above fail open -- an override key this test does not name is "
        "silently ignored, and `???` is skipped wholesale -- so the axis is counted "
        "per key rather than in total. Repair the key or the matrix registration; "
        "do not drop the axis."
    )


def test_no_matrix_dispatches_two_runs_into_one_output_directory():
    """Every task in every matrix must key a distinct output directory.

    ``get_canonical_output_dir`` keys on the *algorithm*, not the run label, so a
    matrix that sweeps an override across one ``alg`` writes every cell into the
    same directory and keeps only whichever finished last. Nothing fails: the
    sweep exits 0, reports its full task count, and writes a ``sweep_status.json``
    with no failures. The loss surfaces only at analysis time, as cells that
    appear to have never run.

    ``variant`` is the disambiguator, and the cost of omitting it is measured in
    allocation hours rather than minutes: all three exploration matrices shipped
    without it, which silently collapsed ``pass2_factorial``'s 26 runs onto 5
    directories and left ``exploration_noise_margin`` unable to measure the sigma
    that gates every keep-or-drop call downstream.

    Asserted across every matrix rather than the three that were wrong, because
    the next matrix to sweep an override on one algorithm has no reason to know
    this rule exists.
    """
    from scripts.common import expand_matrix, get_canonical_output_dir

    for path in sorted((Path(CONF_DIR).parent / "conf" / "matrix").glob("*.yaml")):
        seen: dict[str, str] = {}
        for spec in expand_matrix(_matrix(path.stem), path.stem):
            label = str(spec["label"])
            out = str(
                get_canonical_output_dir(
                    phase=spec["phase"],
                    dataset=spec["dataset"],
                    model=spec["model"],
                    exp_group=spec["experiment_group"],
                    algorithm=spec["algorithm_config"],
                    heterogeneity=spec["heterogeneity"],
                    seed=spec["seed"],
                    variant=spec["variant"],
                )
            )
            assert out not in seen, (
                f"conf/matrix/{path.name}: runs {seen[out]!r} and {label!r} "
                f"both dispatch into {out}. Give each a distinct `variant:` "
                "-- otherwise only the last one to finish survives, and the "
                "sweep reports success either way."
            )
            seen[out] = label


def test_primary_grid_files_dispatch_all_105_runs():
    """§4.5: main benchmark 2 datasets x 2 alpha x 7 algorithms x 3 seeds = 84,
    plus FEMNIST 7 x 3 = 21.

    The arithmetic is the point. A matrix file that silently covers one dataset
    of three still runs to completion and still produces a table; the shortfall
    shows up only as absent rows, months later. This asserts the three files sum
    to what the manuscript promises, and that they agree on the group analysis.py
    reads them by.
    """
    total = 0
    for name in PRIMARY_GRID_MATRICES:
        matrix = _matrix(name)
        assert matrix["experiment_group"] == "benchmark_grid", (
            f"conf/matrix/{name}.yaml declares experiment_group "
            f"{matrix['experiment_group']!r}. All three primary-grid files must "
            "share one group or analysis.py splits the headline table."
        )
        assert len(matrix["runs"]) == 7, (
            f"conf/matrix/{name}.yaml dispatches {len(matrix['runs'])} algorithms; "
            "§4.5 compares 7 (six baselines plus FedMAQ) on every dataset."
        )
        total += len(matrix["runs"]) * len(matrix["heterogeneities"]) * len(matrix["seeds"])
    assert total == 105, (
        f"The primary grid dispatches {total} runs; §4.5 accounts for 105 "
        "(84 CIFAR-10/100 + 21 FEMNIST)."
    )


def test_femnist_grid_selects_its_own_experiment_group():
    """Table 4.1 note (a): FEMNIST runs at K = 200, one real LEAF writer each.

    conf/experiment/default.yaml is K = 100 with MobileNetV2GN throughput
    constants. Without the group override the FEMNIST grid composes from that
    default and runs a differently-sized federation than the one the manuscript
    describes, at the wrong telemetry, while looking entirely normal in the log.
    """
    matrix = _matrix("benchmark_grid_femnist")
    assert matrix.get("experiment") == "femnist", (
        "conf/matrix/benchmark_grid_femnist.yaml must set `experiment: femnist` "
        "to pick up num_clients=200 and the SimpleCNN throughput constants."
    )
    assert matrix["heterogeneities"] == ["femnist"], (
        "§4.1 exempts FEMNIST from the Dirichlet sweep; its writer partition is "
        "already non-IID and layering synthetic skew on it departs from LEAF."
    )


def test_ablation_grid_never_turns_the_post_process_pipeline_on():
    """The mirror guard, and the one that actually costs six runs.

    The pipeline is downstream of every awareness signal. If any ablation arm
    carried it while its reference did not (or the reverse), that arm would be
    two removals from its reference rather than one, and its delta -- especially
    on the communication axis -- would attribute to the coding pipeline rather
    than to the signal the arm names. Configuration 7 is listed in the ablation
    matrix rather than inherited from the benchmark grid for exactly this reason.
    """
    matrix = _matrix("ablation")
    enabled = _post_process_overrides(matrix)
    assert not any(enabled.values()), (
        f"conf/matrix/ablation.yaml enables the post-processing pipeline on "
        f"{sorted(k for k, v in enabled.items() if v)}. Every §4.3.7 arm must "
        f"run in the same regime as its parity anchor."
    )
    assert any(run["alg"] == "fedmaq" for run in matrix["runs"]), (
        "Configuration 7 (full FedMAQ) must be dispatched by the ablation "
        "matrix. Inheriting it from benchmark_grid.yaml reintroduces the "
        "pipeline as a second removal in every other arm."
    )
    # Configurations 2-7 are always net-new; Configuration 8 (fedmaq_no_refinements)
    # only exists while there is a frozen refinement layer to remove (Decision 60,
    # test_configuration_8_exists_only_while_there_is_a_layer_to_remove). Deriving
    # the expected count from that same condition means this assertion doesn't need
    # hand-editing in lockstep with a freeze the way a literal would.
    expected_arms = 7 if _frozen_refinements() else 6
    assert len(matrix["runs"]) == expected_arms, (
        f"§4.3.7 dispatches {expected_arms} net-new arms ({expected_arms * 6} runs) "
        f"given the current freeze; only Configuration 1, uncompressed FedAvg, is "
        f"inherited from the primary grid."
    )


def test_uniform_memory_control_arm_matches_its_comparison_partner():
    """§4.1's control arm is compared against FedMAQ's primary-grid rows.

    Third site of the same defect. The arm holds memory constant to isolate
    server-side KD's recovery from memory-driven quantization; its partner is
    FedMAQ's own variable-memory rows in the benchmark grid, which carry the §4.3
    coding pipeline. Without the pipeline here the contrast silently becomes
    "uniform memory and no pipeline" against "variable memory with one".

    The rule, everywhere: match the regime of whatever the run is compared
    against, not the regime of the algorithm config it composes from.
    """
    matrix = _matrix("uniform_memory_control")
    enabled = _post_process_overrides(matrix)
    assert all(enabled.get(run["label"]) is True for run in matrix["runs"]), (
        "conf/matrix/uniform_memory_control.yaml must set "
        "algorithm.post_process=true; its comparison partner is the primary grid."
    )
    # Six runs: both alphas, three seeds. Split into per-alpha heterogeneity
    # configs because output dirs key on the config name, so one file overridden
    # twice would collide.
    assert len(matrix["heterogeneities"]) == 2
    assert len(matrix["runs"]) * len(matrix["heterogeneities"]) * len(matrix["seeds"]) == 6


def test_formulation_study_never_turns_the_post_process_pipeline_on():
    """§4.3.6 judges the five formulations on their mathematical merit alone.

    Fourth site of the same rule, and the one with a second dependency: the
    Formulation 1 cell is Ablation Configuration 4's parity anchor under §4.3.7's
    fallback rule, so if the pipeline appeared here that arm would be compared
    against a run in a different regime from every other arm in its own study.
    """
    matrix = _matrix("formulation_study")
    enabled = _post_process_overrides(matrix)
    assert not any(enabled.values()), (
        f"conf/matrix/formulation_study.yaml enables the post-processing pipeline "
        f"on {sorted(k for k, v in enabled.items() if v)}."
    )
    formulations = {
        int(o.partition("=")[2])
        for run in matrix["runs"]
        for o in run.get("overrides") or []
        if o.startswith("algorithm.formulation=")
    }
    assert formulations == {0, 1, 2, 3, 4}, (
        "§4.3.6 evaluates all five candidate formulations; the study's winner "
        "rule is not pre-registrable over a subset chosen after the fact."
    )
    assert len(matrix["runs"]) * len(matrix["heterogeneities"]) * len(matrix["seeds"]) == 30


def test_baseline_tuning_gives_every_baseline_the_same_budget_as_fedmaq():
    """Decision 67. §4.3.2's fairness claim is procedural, so the file has to
    hold its shape or the claim is an assertion again.

    Until 2026-08-01 no matrix file tuned a baseline at all, while FedMAQ had a
    38-run exploration phase and a 30-run formulation study. `You tuned yours and
    not theirs` is the most predictable attack on the grid, and the only answer
    that survives is uniform treatment: every baseline with a knob gets one, at
    the same seed depth, under the same rule, at the same held-out skew.
    """
    matrix = _matrix("baseline_tuning")
    assert matrix["phase"] == "explore", (
        "These runs select a configuration and therefore cannot sit in the "
        "confirmatory grid they configure. They are not among the 183."
    )
    assert matrix["heterogeneities"] == ["dirichlet_alpha_0.3"], (
        "Baselines must be tuned at the same held-out skew as FedMAQ's own "
        "mechanisms, or a baseline is selected on a skew it is later reported on."
    )
    assert matrix["total_rounds"] == 100, (
        "R=50 would be a truncated-horizon pick shipped straight into the "
        "reported grid: unlike the factorial, this stage has no confirmation "
        "stage behind it to catch one."
    )

    enabled = _post_process_overrides(matrix)
    assert not any(enabled.values()), (
        "Every baseline ships post_process: false and the primary grid enables "
        "the pipeline on FedMAQ only, so tuning must happen in the regime the "
        "baselines are actually reported in."
    )

    by_alg: dict[str, list] = {}
    for run in matrix["runs"]:
        by_alg.setdefault(run["alg"], []).append(run)

    assert "fedavg" not in by_alg, (
        "FedAvg is the uncompressed control and has no key hyperparameter; "
        "including it would tune the reference the target floor is defined from."
    )
    assert set(by_alg) == {"fedprox", "fedpaq", "dadaquant", "feddistill", "fedkd"}, (
        f"Tuned baselines are {sorted(by_alg)}. A baseline dropped from this file "
        "reintroduces the selective-sweep shape Decision 67 rejected: the person "
        "whose algorithm benefits deciding whose knobs deserve tuning."
    )

    default_seeds = list(matrix["seeds"])
    total = 0
    for alg, runs in by_alg.items():
        assert len(runs) == 3, (
            f"{alg} has {len(runs)} cells; the budget is one reference + two challengers."
        )
        deep = [r for r in runs if len(r.get("seeds") or default_seeds) == 5]
        assert len(deep) == 1, (
            f"{alg} must deepen exactly its shipped-value reference cell to five "
            "seeds. That cell's spread is the sigma its two challengers are judged "
            "against, so it sets the precision of two decisions while each "
            "challenger's sets none (§4.4, Decision 61 applied per baseline)."
        )
        assert deep[0]["label"].endswith("-ref")
        total += sum(len(r.get("seeds") or default_seeds) for r in runs)

    assert total * len(matrix["heterogeneities"]) == 55, (
        f"conf/matrix/baseline_tuning.yaml dispatches {total} runs; the design is "
        "55 (5 baselines x (5 + 3 + 3)) per ADR-0011. This test is where that "
        "count is pinned -- docs/agents/execution-model.md states the shape, not "
        "the number, so it cannot drift out of agreement with the matrix."
    )


def test_wide_baseline_tuning_adds_fedmaq_and_four_challengers():
    matrix = _matrix("baseline_tuning_wide")
    assert matrix["experiment_group"] == "baseline_tuning_wide"
    assert matrix["phase"] == "explore"
    assert matrix["heterogeneities"] == ["dirichlet_alpha_0.3"]
    assert matrix["total_rounds"] == 100

    by_alg: dict[str, list] = {}
    for run in matrix["runs"]:
        by_alg.setdefault(run["alg"], []).append(run)

    assert set(by_alg) == {
        "fedprox",
        "fedpaq",
        "dadaquant",
        "feddistill",
        "fedkd",
        "fedmaq",
    }
    assert all(len(runs) == 5 for alg, runs in by_alg.items() if alg != "fedmaq")
    assert len(by_alg["fedmaq"]) == 4
    assert (
        sum(len(run.get("seeds") or matrix["seeds"]) for runs in by_alg.values() for run in runs)
        == 145
    )

    for algorithm, runs in by_alg.items():
        references = [run for run in runs if run["label"].endswith("-ref")]
        assert len(references) == 1
        assert len(references[0].get("seeds") or matrix["seeds"]) == 5
        assert all(
            len(run.get("seeds") or matrix["seeds"]) == 5 for run in runs if run not in references
        )
        post_process = _post_process_overrides({"runs": runs})
        if algorithm == "fedmaq":
            assert len(post_process) == 4
            assert {
                "algorithm.q_max=4",
                "algorithm.q_max=6",
                "algorithm.q_max=8",
                "algorithm.q_max=16",
            } <= {
                override
                for run in runs
                for override in run["overrides"]
                if override.startswith("algorithm.q_max=")
            }
        else:
            assert not post_process


def test_wide_baseline_tuning_adopted_variant_matches_shipped_config():
    """``tuning.<algo>.adopted_variant`` is provenance metadata read by
    scripts/analysis.py for reporting, not a computational input -- but it must
    still name the value each algorithm actually ships, or the report labels a
    baseline as tuned to a reference cell that was never run. This caught a
    real drift (fedprox, feddistill, dadaquant) resolved alongside this test."""
    matrix = _matrix("baseline_tuning_wide")
    for algorithm, spec in matrix["tuning"].items():
        adopted_variant = spec["adopted_variant"]
        assert adopted_variant in spec["values"], (
            f"{algorithm}.adopted_variant={adopted_variant!r} is not a key in "
            f"{algorithm}.values={spec['values']!r}"
        )
        shipped = _algorithm_cfg(algorithm)[spec["knob"]]
        assert float(spec["values"][adopted_variant]) == float(shipped), (
            f"{algorithm}.adopted_variant={adopted_variant!r} names "
            f"{spec['values'][adopted_variant]!r} but conf/algorithm/{algorithm}.yaml "
            f"ships {spec['knob']}={shipped!r} -- update adopted_variant to match."
        )


@pytest.mark.parametrize(
    "matrix_name",
    [
        "benchmark_grid",
        "ablation",
        "uniform_memory_control",
        "formulation_study",
        "baseline_tuning",
    ],
)
def test_matrix_runs_never_collide_on_an_output_directory(matrix_name):
    """Two runs writing to one directory means one of them is silently discarded.

    The canonical path keys on the algorithm config, not the run label, so any
    matrix listing the same ``alg`` twice needs a ``variant`` to separate them.
    This has bitten twice: the uniform-memory control arm (worked around by
    splitting the heterogeneity config per alpha) and the formulation study's five
    formulations of ``fedmaq``. Neither failed loudly -- the sweep completes and
    the missing runs simply are not there at analysis time.
    """
    import sys

    sys.path.insert(0, str(Path(CONF_DIR).parent))
    from scripts.common import get_canonical_output_dir

    matrix = _matrix(matrix_name)
    dirs = [
        get_canonical_output_dir(
            phase=matrix.get("phase", "smoke"),
            dataset=matrix.get("dataset", "cifar10"),
            model=matrix.get("model", "mobilenetv2"),
            exp_group=matrix.get("experiment_group", matrix_name),
            algorithm=run["alg"],
            heterogeneity=het,
            seed=seed,
            variant=run.get("variant", ""),
        )
        for het in matrix["heterogeneities"]
        for seed in matrix["seeds"]
        for run in matrix["runs"]
    ]
    duplicates = {d for d in dirs if dirs.count(d) > 1}
    assert not duplicates, (
        f"conf/matrix/{matrix_name}.yaml maps more than one run onto "
        f"{sorted(str(d) for d in duplicates)}. Give the colliding runs a "
        f"`variant`, or the later run overwrites the earlier one."
    )


def test_run_manifest_hashes_the_resolved_config(tmp_path):
    """§4.3.1: config content is hashed into the run manifest for verification.

    The hash must key on what a run *executed*, so an override that changes the
    algorithm must change the digest, while a cosmetic re-ordering must not.
    """
    from fedmaq.core.manifest import MANIFEST_FILENAME, config_sha256, write_run_manifest

    base = _algorithm_cfg("fedmaq")
    reordered = dict(reversed(list(base.items())))
    assert config_sha256(base) == config_sha256(reordered), (
        "digest must be order-independent, or the same configuration reached by "
        "different override spellings would look like two configurations"
    )
    assert config_sha256(base) != config_sha256(_algorithm_cfg("fedmaq_no_data"))

    path = write_run_manifest({"algorithm": base, "seed": 42}, tmp_path)
    assert path == tmp_path / MANIFEST_FILENAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["run"]["seed"] == 42
    assert len(manifest["config_sha256"]) == 64
    # Provenance fields §4.3.1's freeze depends on being able to check.
    assert "commit" in manifest["git"] and "dirty" in manifest["git"]


def test_manifest_ignores_generated_artifacts_but_not_source_changes(monkeypatch, tmp_path):
    """Formal result files must not make later grid tasks look unreproducible."""
    import fedmaq.core.manifest as manifest_module

    class CompletedProcess:
        def __init__(self, stdout):
            self.stdout = stdout

    def provenance_for(status):
        def fake_run(command, **_kwargs):
            if command[1] == "status":
                return CompletedProcess(status)
            if command[1:] == ["rev-parse", "HEAD"]:
                return CompletedProcess("commit")
            if command[1:] == ["rev-parse", "--abbrev-ref", "HEAD"]:
                return CompletedProcess("main")
            if command[1:] == ["describe", "--tags", "--exact-match"]:
                return CompletedProcess("pre-registration")
            raise AssertionError(f"unexpected git command: {command}")

        monkeypatch.setattr(manifest_module.subprocess, "run", fake_run)
        return manifest_module._git_provenance(tmp_path)

    generated_only = "?? outputs/\0?? scripts/analysis_output/\0"
    assert provenance_for(generated_only)["dirty"] is False
    assert provenance_for(generated_only + " M src/fedmaq/simulation.py\0")["dirty"] is True


def test_final_global_model_is_written_only_on_the_last_round(tmp_path):
    """§5.2.1's t-SNE plots are built from the trained global model after the grid.

    Nothing persisted that model until 2026-07-30, so the figure would have been
    unbuildable once all 183 runs finished. Guard both halves of the contract:
    the last round writes a loadable state_dict, and no earlier round writes at
    all (a per-round checkpoint would multiply the grid's disk cost by 100).
    """
    import torch

    from fedmaq.core.checkpoint import FINAL_MODEL_FILENAME, write_final_global_model

    model = torch.nn.Linear(4, 3)
    total_rounds = 100

    assert write_final_global_model(model, tmp_path, 1, total_rounds) is None
    assert write_final_global_model(model, tmp_path, 99, total_rounds) is None
    assert not (tmp_path / FINAL_MODEL_FILENAME).exists()

    path = write_final_global_model(model, tmp_path, total_rounds, total_rounds)
    assert path == tmp_path / FINAL_MODEL_FILENAME

    loaded = torch.load(path, map_location="cpu")
    reconstructed = torch.nn.Linear(4, 3)
    reconstructed.load_state_dict(loaded)
    assert torch.allclose(reconstructed.weight, model.weight)


def test_run_completion_keys_on_the_final_round_checkpoint(tmp_path):
    """Resume must not mistake a killed run for a finished one.

    Every other per-run artifact appears before round 1, so the final-round
    checkpoint is the only usable completion sentinel. Also pin the sweep-group
    directory as an ancestor of the canonical run directory, since
    ``sweep_status.json`` is written there.
    """
    import sys

    sys.path.insert(0, str(Path(CONF_DIR).parent))
    from fedmaq.core.checkpoint import FINAL_MODEL_FILENAME
    from fedmaq.core.manifest import MANIFEST_FILENAME
    from scripts.common import (
        get_canonical_output_dir,
        get_sweep_group_dir,
        is_run_complete,
    )

    assert not is_run_complete(tmp_path)
    # Written before round 1, so it must not read as completion.
    (tmp_path / MANIFEST_FILENAME).write_text(
        json.dumps({"run": {"total_rounds": 1}}), encoding="utf-8"
    )
    (tmp_path / "experiment_log.csv").write_text(
        "round,train/loss,communication/cumulative_mb\n1,0.5,1.0\n", encoding="utf-8"
    )
    (tmp_path / "experiment_log.jsonl").write_text(
        '{"round": 1, "communication/cumulative_mb": 1.0}\n', encoding="utf-8"
    )
    assert not is_run_complete(tmp_path)

    import torch

    torch.save({"weight": torch.ones(1)}, tmp_path / FINAL_MODEL_FILENAME)
    assert is_run_complete(tmp_path)

    (tmp_path / FINAL_MODEL_FILENAME).write_bytes(b"")
    assert not is_run_complete(tmp_path)

    group = get_sweep_group_dir("primary", "cifar10", "mobilenetv2", "benchmark_grid")
    run_dir = get_canonical_output_dir(
        phase="primary",
        dataset="cifar10",
        model="mobilenetv2",
        exp_group="benchmark_grid",
        algorithm="fedmaq",
        heterogeneity="dirichlet_alpha_0.1",
        seed=0,
    )
    assert group in run_dir.parents


def test_sweep_records_failed_indices_and_can_skip_completed_runs(tmp_path, monkeypatch):
    """A multi-day sweep must leave a machine-readable record of what failed.

    ``--start_at`` is positional, so without this the only account of *which*
    index failed is a log stream, and gap-filling means arithmetic against it.
    Drives the real runner with the subprocess call faked, so this exercises the
    dispatch loop rather than a reimplementation of it.
    """
    import subprocess
    import sys

    sys.path.insert(0, str(Path(CONF_DIR).parent))
    import scripts.run_matrix as run_matrix
    from fedmaq.core.checkpoint import FINAL_MODEL_FILENAME
    from fedmaq.core.manifest import MANIFEST_FILENAME
    from scripts.common import SWEEP_STATUS_FILENAME

    matrix_dir = tmp_path / "conf" / "matrix"
    matrix_dir.mkdir(parents=True)
    (matrix_dir / "probe.yaml").write_text(
        "phase: smoke\n"
        "experiment_group: status_probe\n"
        "dataset: cifar10\n"
        "model: mobilenetv2\n"
        "total_rounds: 1\n"
        "seeds: [0]\n"
        "heterogeneities: [dirichlet_alpha_0.1]\n"
        "runs:\n"
        "  - alg: fedavg\n"
        "  - alg: fedprox\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_matrix, "kill_ray_processes", lambda: None)
    monkeypatch.setattr(run_matrix.time, "sleep", lambda _seconds: None)

    dispatched: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        dispatched.append(cmd)
        # First arm fails, second succeeds.
        return subprocess.CompletedProcess(cmd, 1 if "algorithm=fedavg" in cmd else 0)

    monkeypatch.setattr(run_matrix.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_matrix.py", "--matrix", "probe"])
    run_matrix.main()

    group_dir = Path("outputs/smoke/cifar10_mobilenetv2/status_probe")
    status = json.loads((group_dir / SWEEP_STATUS_FILENAME).read_text(encoding="utf-8"))
    assert status["state"] == "finished"
    assert status["total_tasks"] == 2
    assert status["completed"] == 1
    assert status["failed_indices"] == [1]
    assert status["failures"][0]["label"] == "fedavg-dirichlet_alpha_0.1-seed0"
    assert status["failures"][0]["returncode"] == 1

    # Mark the arm that succeeded as complete; the gap-filling re-run must
    # dispatch only the one that failed.
    done = group_dir / "fedprox" / "dirichlet_alpha_0.1" / "seed_0"
    done.mkdir(parents=True, exist_ok=True)
    torch.save({"weight": torch.ones(1)}, done / FINAL_MODEL_FILENAME)
    (done / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "run": {
                    "dataset": "cifar10",
                    "algorithm": "fedprox",
                    "algorithm_config": "fedprox",
                    "alpha": 0.1,
                    "seed": 0,
                    "total_rounds": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    (done / "experiment_log.csv").write_text(
        "round,train/loss,communication/cumulative_mb\n1,0.5,1.0\n",
        encoding="utf-8",
    )
    (done / "experiment_log.jsonl").write_text(
        '{"round": 1, "communication/cumulative_mb": 1.0, "client/avg_train_loss": 0.5}\n',
        encoding="utf-8",
    )

    dispatched.clear()
    monkeypatch.setattr(sys, "argv", ["run_matrix.py", "--matrix", "probe", "--skip_completed"])
    run_matrix.main()

    assert len(dispatched) == 1, "the completed arm should not have been re-dispatched"
    assert "algorithm=fedavg" in dispatched[0]


def test_shard_dispatches_only_canonical_members_and_writes_host_status(tmp_path, monkeypatch):
    import socket
    import subprocess
    import sys

    sys.path.insert(0, str(Path(CONF_DIR).parent))
    import scripts.run_matrix as run_matrix
    from scripts.common import sharded_sweep_status_filename

    group_dir = _write_probe_matrix(tmp_path, ["fedavg", "fedprox", "fedpaq", "fedmaq", "qsgd"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_matrix, "kill_ray_processes", lambda: None)
    monkeypatch.setattr(run_matrix.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(socket, "gethostname", lambda: "host-b")

    dispatched = []

    def fake_run(cmd, *args, **kwargs):
        dispatched.append((cmd, kwargs["env"]))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(run_matrix.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_matrix.py", "--matrix", "probe", "--shard", "2/3"],
    )
    run_matrix.main()

    assert ["algorithm=fedprox", "algorithm=qsgd"] == [
        next(arg for arg in cmd if arg.startswith("algorithm=")) for cmd, _ in dispatched
    ]
    assert all(env["FEDMAQ_SWEEP_HOST"] == "host-b" for _, env in dispatched)
    assert all(env["FEDMAQ_SWEEP_SHARD_INDEX"] == "2" for _, env in dispatched)
    status = json.loads(
        (group_dir / sharded_sweep_status_filename(2, 3)).read_text(encoding="utf-8")
    )
    assert status["total_tasks"] == 5
    assert status["shard_tasks"] == 2
    assert status["shard"]["canonical_indices"] == [2, 5]
    assert {run["host"] for run in status["runs"]} == {"host-b"}
    assert {run["source_root"] for run in status["runs"]} == {
        run_dir.resolve().as_posix()
        for run_dir in [
            group_dir / "fedprox" / "dirichlet_alpha_0.1" / "seed_0",
            group_dir / "qsgd" / "dirichlet_alpha_0.1" / "seed_0",
        ]
    }


def test_sharding_rejects_only_filter(tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(Path(CONF_DIR).parent))
    import scripts.run_matrix as run_matrix

    _write_probe_matrix(tmp_path, ["fedavg"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_matrix.py", "--matrix", "probe", "--shard", "1/2", "--only", "fedavg"],
    )

    with pytest.raises(SystemExit):
        run_matrix.main()


def test_merge_shard_statuses_requires_full_disjoint_union(tmp_path):
    import sys

    sys.path.insert(0, str(Path(CONF_DIR).parent))
    from scripts.merge_sweep_status import merge_statuses

    def status(shard_index, indices):
        return {
            "matrix": "conf/matrix/probe.yaml",
            "experiment_group": "status_probe",
            "total_tasks": 4,
            "state": "finished",
            "host": f"host-{shard_index}",
            "started_at": "2026-08-27T00:00:00",
            "updated_at": "2026-08-27T00:01:00",
            "shard": {"index": shard_index, "count": 2, "canonical_indices": indices},
            "runs": [
                {
                    "index": index,
                    "label": f"run-{index}",
                    "state": "completed",
                    "host": f"host-{shard_index}",
                }
                for index in indices
            ],
            "failures": [],
        }

    merged = merge_statuses([status(1, [1, 3]), status(2, [2, 4])])
    assert merged["state"] == "finished"
    assert merged["completed"] == 4
    assert merged["hosts"] == ["host-1", "host-2"]

    with pytest.raises(ValueError, match="shard union is not the matrix"):
        merge_statuses([status(1, [1, 3])])


def test_run_manifest_records_host_and_shard_provenance(monkeypatch, tmp_path):
    import fedmaq.core.manifest as manifest

    monkeypatch.setattr(manifest.socket, "gethostname", lambda: "host-c")
    monkeypatch.setenv("FEDMAQ_SWEEP_SHARD_INDEX", "3")
    monkeypatch.setenv("FEDMAQ_SWEEP_SHARD_COUNT", "4")

    record = manifest.build_manifest({"algorithm": {}, "seed": 0}, repo_root=tmp_path)

    assert record["environment"]["host"] == "host-c"
    assert record["dispatch"]["shard"] == {"index": 3, "count": 4}


def test_run_manifest_records_source_root(tmp_path):
    from fedmaq.core.manifest import write_run_manifest

    path = write_run_manifest({"algorithm": {}, "seed": 0}, tmp_path)
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["source_root"] == tmp_path.resolve().as_posix()


def _write_probe_matrix(tmp_path, arms):
    """Write a minimal matrix file with one run per entry in ``arms``."""
    matrix_dir = tmp_path / "conf" / "matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    runs = "".join(f"  - alg: {alg}\n" for alg in arms)
    (matrix_dir / "probe.yaml").write_text(
        "phase: smoke\n"
        "experiment_group: status_probe\n"
        "dataset: cifar10\n"
        "model: mobilenetv2\n"
        "total_rounds: 1\n"
        "seeds: [0]\n"
        "heterogeneities: [dirichlet_alpha_0.1]\n"
        f"runs:\n{runs}",
        encoding="utf-8",
    )
    return Path("outputs/smoke/cifar10_mobilenetv2/status_probe")


def test_sweep_aborts_on_consecutive_failures_without_claiming_it_finished(tmp_path, monkeypatch):
    """A systemic failure must stop the queue, not be repeated 100 more times.

    On a contended GPU a co-tenant VRAM spike or a leaked Ray actor fails one run
    and then every run after it. Detached via ``setsid`` there is no terminal to
    notice from, so the sweep has to notice itself and say so in a file. The
    terminal state must not be ``finished``: runs were left undispatched, and
    ``sweep_status.json`` is the only account of that.
    """
    import subprocess
    import sys

    sys.path.insert(0, str(Path(CONF_DIR).parent))
    import scripts.run_matrix as run_matrix
    from scripts.common import SWEEP_STATUS_FILENAME

    group_dir = _write_probe_matrix(tmp_path, ["fedavg", "fedprox", "fedpaq", "fedmaq", "qsgd"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_matrix, "kill_ray_processes", lambda: None)
    monkeypatch.setattr(run_matrix.time, "sleep", lambda _seconds: None)

    dispatched: list[list[str]] = []
    monkeypatch.setattr(
        run_matrix.subprocess,
        "run",
        lambda cmd, *a, **k: (
            dispatched.append(cmd),
            subprocess.CompletedProcess(cmd, 1),
        )[1],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_matrix.py", "--matrix", "probe", "--max_consecutive_failures", "3"],
    )

    with pytest.raises(SystemExit) as exc:
        run_matrix.main()
    assert exc.value.code == 1, "an aborted sweep must exit non-zero"

    assert len(dispatched) == 3, "the queue must stop at the threshold, not run on"
    status = json.loads((group_dir / SWEEP_STATUS_FILENAME).read_text(encoding="utf-8"))
    assert status["state"] == "aborted"
    assert status["abort_reason"] and "3 consecutive failures" in status["abort_reason"]
    assert status["total_tasks"] == 5
    assert status["failed"] == 3


def test_sweep_threshold_counts_consecutive_failures_not_total(tmp_path, monkeypatch):
    """One success resets the counter: scattered failures are not a systemic one.

    Without the reset, a threshold of 3 would abort any long sweep that merely
    accumulated three unrelated bad runs across 100+ dispatches, which is exactly
    the case ``--skip_completed`` gap-filling already handles well.
    """
    import subprocess
    import sys

    sys.path.insert(0, str(Path(CONF_DIR).parent))
    import scripts.run_matrix as run_matrix
    from scripts.common import SWEEP_STATUS_FILENAME

    group_dir = _write_probe_matrix(tmp_path, ["fedavg", "fedprox", "fedpaq", "fedmaq", "qsgd"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_matrix, "kill_ray_processes", lambda: None)
    monkeypatch.setattr(run_matrix.time, "sleep", lambda _seconds: None)

    # Fail, fail, succeed, fail, fail: five tasks, four failures, never three in a row.
    codes = iter([1, 1, 0, 1, 1])
    monkeypatch.setattr(
        run_matrix.subprocess,
        "run",
        lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, next(codes)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_matrix.py", "--matrix", "probe", "--max_consecutive_failures", "3"],
    )
    run_matrix.main()

    status = json.loads((group_dir / SWEEP_STATUS_FILENAME).read_text(encoding="utf-8"))
    assert status["state"] == "finished"
    assert status["abort_reason"] is None
    assert status["failed"] == 4


def test_sweep_kills_a_hung_run_and_records_it_distinguishably(tmp_path, monkeypatch):
    """A Ray deadlock must not freeze the sweep at ``running`` forever.

    Ray can deadlock waiting on an actor that never starts, the documented
    low-system-RAM failure mode, and a bare ``subprocess.run`` waits on it
    indefinitely. The timeout also has to reap Ray itself: ``subprocess.run`` kills
    the direct child, while the raylet and actors are grandchildren that survive it
    still holding VRAM.
    """
    import subprocess
    import sys

    sys.path.insert(0, str(Path(CONF_DIR).parent))
    import scripts.run_matrix as run_matrix
    from scripts.common import SWEEP_STATUS_FILENAME

    group_dir = _write_probe_matrix(tmp_path, ["fedavg", "fedprox"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_matrix.time, "sleep", lambda _seconds: None)

    cleanups: list[int] = []
    monkeypatch.setattr(run_matrix, "kill_ray_processes", lambda: cleanups.append(1))

    seen_timeouts: list[object] = []

    def fake_run(cmd, *args, **kwargs):
        seen_timeouts.append(kwargs.get("timeout"))
        if "algorithm=fedavg" in cmd:
            raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout") or 0)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(run_matrix.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_matrix.py", "--matrix", "probe", "--run_timeout_seconds", "900"],
    )
    run_matrix.main()

    assert seen_timeouts == [900, 900], "the timeout must reach subprocess.run"
    status = json.loads((group_dir / SWEEP_STATUS_FILENAME).read_text(encoding="utf-8"))
    assert status["state"] == "finished", "one hang is not a systemic failure"
    assert status["failed"] == 1 and status["completed"] == 1
    hang = status["failures"][0]
    assert hang["timed_out"] is True
    assert hang["returncode"] == run_matrix.TIMEOUT_RETURNCODE
    # Two per-run cleanups plus the post-sweep one, plus the extra reap on timeout.
    assert len(cleanups) == 4, "a timeout must trigger its own Ray cleanup"


def test_no_timeout_by_default_so_a_slow_run_is_not_a_failed_one(tmp_path, monkeypatch):
    """The default must stay ``None``, not a guessed number.

    Total grid runtime is unmeasured, and the contended host makes the tail long. A
    default timeout would convert slowness into fabricated failures.
    """
    import subprocess
    import sys

    sys.path.insert(0, str(Path(CONF_DIR).parent))
    import scripts.run_matrix as run_matrix

    _write_probe_matrix(tmp_path, ["fedavg"])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_matrix, "kill_ray_processes", lambda: None)
    monkeypatch.setattr(run_matrix.time, "sleep", lambda _seconds: None)

    seen: list[object] = []

    def fake_run(cmd, *args, **kwargs):
        seen.append(kwargs.get("timeout"))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(run_matrix.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_matrix.py", "--matrix", "probe"])
    run_matrix.main()

    assert seen == [None]


def test_sweep_wide_overrides_reach_every_run_but_never_outrank_the_matrix(tmp_path, monkeypatch):
    """Host settings belong on the command line; regime settings belong to the matrix.

    ``ray.temp_dir`` and ``ray.object_store_gb`` are properties of the machine, and
    the alternative to a flag is hand-editing conf/config.yaml on the host and
    carrying that diff across a multi-day grid, where one git pull reverts it
    partway through. But Hydra resolves the last override, so the flag has to be
    emitted *before* the matrix's own: ``algorithm.post_process`` fixes which regime
    a run is comparable in, and a hand-typed flag must not be able to displace it.
    """
    import subprocess
    import sys

    sys.path.insert(0, str(Path(CONF_DIR).parent))
    import scripts.run_matrix as run_matrix

    matrix_dir = tmp_path / "conf" / "matrix"
    matrix_dir.mkdir(parents=True)
    (matrix_dir / "probe.yaml").write_text(
        "phase: smoke\n"
        "experiment_group: status_probe\n"
        "dataset: cifar10\n"
        "model: mobilenetv2\n"
        "total_rounds: 1\n"
        "seeds: [0]\n"
        "heterogeneities: [dirichlet_alpha_0.1]\n"
        "runs:\n"
        "  - alg: fedmaq\n"
        "    overrides: [algorithm.post_process=true]\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_matrix, "kill_ray_processes", lambda: None)
    monkeypatch.setattr(run_matrix.time, "sleep", lambda _seconds: None)

    dispatched: list[list[str]] = []
    monkeypatch.setattr(
        run_matrix.subprocess,
        "run",
        lambda cmd, *a, **k: (
            dispatched.append(cmd),
            subprocess.CompletedProcess(cmd, 0),
        )[1],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_matrix.py",
            "--matrix",
            "probe",
            "-o",
            "ray.temp_dir=/tmp/ray-cjb",
            "-o",
            "algorithm.post_process=false",
        ],
    )
    run_matrix.main()

    cmd = dispatched[0]
    assert "ray.temp_dir=/tmp/ray-cjb" in cmd
    assert cmd.index("algorithm.post_process=false") < cmd.index("algorithm.post_process=true"), (
        "the matrix file's regime setting must be the one Hydra resolves"
    )


def test_ray_init_args_default_to_flowers_stock_behaviour():
    """Absent config must produce no ``init_args`` at all, not an empty dict of them.

    The local development rig and the CI suite run without either setting, so the
    null path has to be byte-identical to what shipped before this existed.
    """
    from fedmaq.simulation import build_ray_init_args

    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config")
    assert cfg.ray.object_store_gb is None
    assert cfg.ray.temp_dir is None
    assert build_ray_init_args(cfg) == {}


def test_ray_init_args_translate_to_ray_kwargs():
    """The two host settings must arrive as the exact ``ray.init`` keyword names.

    Flower forwards every ``init_args`` key verbatim into ``ray.init()`` with no
    schema, so a misspelling is not caught anywhere: Ray absorbs unknown
    underscore-prefixed names into ``**kwargs``. Pin the spellings and the GB to
    bytes conversion here.
    """
    from fedmaq.simulation import build_ray_init_args

    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["ray.object_store_gb=4", "ray.temp_dir=/tmp/ray-cjb"],
        )
    args = build_ray_init_args(cfg)
    assert args["object_store_memory"] == 4 * 1024**3
    assert args["_temp_dir"] == "/tmp/ray-cjb"


def test_ray_temp_dir_must_be_absolute():
    """Ray requires an absolute temp dir and reports a relative one poorly.

    Rejecting it here keeps the failure attributable to the config that caused it.
    """
    from fedmaq.simulation import build_ray_init_args

    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=["ray.temp_dir=ray-tmp"])
    with pytest.raises(ValueError, match="absolute path"):
        build_ray_init_args(cfg)


@pytest.mark.parametrize("algorithm", ALGORITHM_CONFIGS)
def test_algorithm_config_composes(algorithm):
    """Every algorithm config must compose into a structurally valid experiment config.

    A malformed ``conf/algorithm/*.yaml`` (or a broken default/interpolation) is
    otherwise invisible to the suite, since the unit tests build cfg dicts inline.
    """
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"algorithm={algorithm}"])

    # Composition wiring: the four config groups + a resolvable algorithm name.
    assert cfg.algorithm.name, f"{algorithm} config is missing algorithm.name"
    assert cfg.dataset.name
    assert cfg.experiment.num_clients > 0
    assert cfg.experiment.total_rounds > 0
    # Manuscript Table 4.1 anchors that must survive composition.
    assert cfg.experiment.batch_size == 64
    assert cfg.experiment.num_public_samples == 3000


def test_run_cfg_smoke_fedavg(mock_dataset, tmp_path, monkeypatch):
    """The extracted run(cfg) entry point drives a real 1-round simulation in-process.

    Unlike a subprocess ``scripts/run.py`` smoke (which only checks an exit code),
    this asserts on returned telemetry state, making orchestration regressions and
    empty-payload bugs observable.
    """
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    from fedmaq.simulation import run

    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=fedavg",
                # Route to the lightweight SimpleCNN path so the 1x28x28 mock fits.
                "dataset.name=mnist",
                "dataset.num_classes=10",
                "experiment.num_clients=2",
                "experiment.total_rounds=1",
                "experiment.num_public_samples=10",
                "experiment.batch_size=2",
                "experiment.local_epochs=1",
                "experiment.client_fraction=1.0",
                "experiment.client_gpus=0.0",
            ],
        )

    telemetry = run(cfg)

    # The run completed and accounted for transmitted bytes over the wire.
    assert telemetry.cumulative_bytes > 0
    assert telemetry.jsonl_path.exists()
    assert np.isfinite(telemetry.cumulative_bytes)


def test_run_cfg_smoke_feddistill_two_rounds(mock_dataset, tmp_path, monkeypatch):
    """FedDistill+ over 2 rounds through the real Flower orchestration.

    Validates the bytes-over-Flower transport that the in-process unit tests bypass:
    clients emit per-class logits in FitRes.metrics, the server averages and
    rebroadcasts them via FitIns.config, and round 2 runs the logit-KD reg path.
    """
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    from fedmaq.simulation import run

    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=feddistill",
                "dataset.name=mnist",
                "dataset.num_classes=10",
                "experiment.num_clients=2",
                "experiment.total_rounds=2",
                "experiment.num_public_samples=10",
                "experiment.batch_size=2",
                "experiment.local_epochs=1",
                "experiment.client_fraction=1.0",
                "experiment.client_gpus=0.0",
            ],
        )

    telemetry = run(cfg)

    assert telemetry.cumulative_bytes > 0
    assert np.isfinite(telemetry.cumulative_bytes)


def test_run_cfg_smoke_cfd_two_rounds(mock_dataset, tmp_path, monkeypatch):
    """CFD over 2 rounds through the real Flower orchestration.

    Validates the soft-label transport the in-process unit tests bypass: clients
    return quantized codes as ``parameters`` (not weights), the server dequantizes
    + averages + dual-distills its persistent server_model, and round 2 broadcasts
    quantized server labels so the client-side digest (KL) branch engages.
    """
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    from fedmaq.simulation import run

    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[
                "algorithm=cfd",
                "dataset.name=mnist",
                "dataset.num_classes=10",
                "experiment.num_clients=2",
                "experiment.total_rounds=2",
                "experiment.num_public_samples=10",
                "experiment.batch_size=2",
                "experiment.local_epochs=1",
                "experiment.client_fraction=1.0",
                "experiment.client_gpus=0.0",
            ],
        )

    telemetry = run(cfg)

    # Round 1 has no downstream broadcast (untrained server model); round 2 does,
    # so cumulative bytes must still be positive and finite overall.
    assert telemetry.cumulative_bytes > 0
    assert np.isfinite(telemetry.cumulative_bytes)


def test_power_mean_omega_matrix_expands_to_twelve_cells():
    from scripts.common import expand_matrix

    matrix = OmegaConf.to_container(
        OmegaConf.load(Path(CONF_DIR) / "matrix" / "power_mean_omega.yaml"),
        resolve=True,
    )
    tasks = expand_matrix(matrix, "power_mean_omega")
    assert len(tasks) == 12
    assert {t["variant"] for t in tasks} == {"omega0.25", "omega0.75"}
    assert {t["seed"] for t in tasks} == {0, 42, 123}


def test_fedpaq_pipeline_matrices_expand_to_fifteen_cells():
    from scripts.common import expand_matrix

    c10 = expand_matrix(
        OmegaConf.to_container(
            OmegaConf.load(Path(CONF_DIR) / "matrix" / "fedpaq_pipeline.yaml"),
            resolve=True,
        ),
        "fedpaq_pipeline",
    )
    c100 = expand_matrix(
        OmegaConf.to_container(
            OmegaConf.load(Path(CONF_DIR) / "matrix" / "fedpaq_pipeline_cifar100.yaml"),
            resolve=True,
        ),
        "fedpaq_pipeline_cifar100",
    )
    femnist = expand_matrix(
        OmegaConf.to_container(
            OmegaConf.load(Path(CONF_DIR) / "matrix" / "fedpaq_pipeline_femnist.yaml"),
            resolve=True,
        ),
        "fedpaq_pipeline_femnist",
    )
    assert len(c10) == 6
    assert len(c100) == 6
    assert len(femnist) == 3
    assert len(c10 + c100 + femnist) == 15
    assert all(t["experiment_group"] == "fedpaq_pipeline" for t in c10 + c100 + femnist)


def test_memory_sensitivity_matrix_expands_to_twelve_net_new_cells():
    from scripts.common import expand_matrix

    matrix = OmegaConf.to_container(
        OmegaConf.load(Path(CONF_DIR) / "matrix" / "memory_sensitivity.yaml"),
        resolve=True,
    )
    tasks = expand_matrix(matrix, "memory_sensitivity")
    assert len(tasks) == 12
    assert {t["variant"] for t in tasks} == {"cunit512", "cunit2048"}
    assert {t["seed"] for t in tasks} == {0, 42, 123}


def _evaluate_arm_decisions(arm: str, formulation: int | str) -> list[tuple[float, int]]:
    """Evaluate decision-details over a 3D signal and capacity grid."""
    from fedmaq.core.quantization_planner import _QuantParams, compute_fedmaq_q_k_t_details

    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=[f"algorithm={arm}", f"algorithm.formulation={formulation}"],
        )
    alg_dict = OmegaConf.to_container(cfg.algorithm, resolve=True)
    qp = _QuantParams.from_cfg(alg_dict)

    g_signals = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_signals = [10, 30, 50, 70, 90]
    capacities = [2048.0, 16384.0]

    decisions = []
    for c_k in capacities:
        for g in g_signals:
            for n in n_signals:
                d = compute_fedmaq_q_k_t_details(
                    c_k=c_k,
                    c_unit=qp.c_unit,
                    g_k=g,
                    g_max=1.0,
                    n_k=n,
                    n_max=100,
                    formulation=qp.formulation,
                    q_min=qp.q_min,
                    q_max=qp.q_max,
                    gamma1=qp.gamma1,
                    gamma2=qp.gamma2,
                    kappa=qp.kappa,
                    tau_g=qp.tau_g,
                    tau_n=qp.tau_n,
                    p=qp.p,
                    omega=qp.omega,
                    bit_widths=qp.bit_widths,
                    resource_aware=qp.resource_aware,
                )
                decisions.append((d.q_hat, d.q))
    return decisions


@pytest.mark.parametrize("formulation", [2, "power_mean"])
@pytest.mark.parametrize("arm", ["fedmaq_no_data", "fedmaq_no_state", "fedmaq_no_resource"])
def test_ablation_arm_decision_vector_differs_from_full_fedmaq(arm, formulation):
    """Under both formulation=2 and formulation=power_mean, each ablation arm must
    differ from full FedMAQ.

    Guards against the latent collapse where power_mean ignores gamma1/gamma2,
    making fedmaq_no_data and fedmaq_no_state byte-identical to full FedMAQ.
    """
    full_vector = _evaluate_arm_decisions("fedmaq", formulation)
    arm_vector = _evaluate_arm_decisions(arm, formulation)
    assert arm_vector != full_vector, (
        f"Ablation arm {arm} produced a decision vector identical to full FedMAQ "
        f"under formulation={formulation!r}. Check that overrides are defined."
    )


@pytest.mark.parametrize("arm", ["fedmaq_no_kd", "fedmaq_no_refinements"])
def test_downstream_ablation_arms_preserve_identical_quantizer_decisions(arm):
    """Confirm §4.3.7 server-side KD and refinement arms do not alter quantizer decisions."""
    for formulation in (2, "power_mean"):
        full_vector = _evaluate_arm_decisions("fedmaq", formulation)
        arm_vector = _evaluate_arm_decisions(arm, formulation)
        assert arm_vector == full_vector, (
            f"Downstream ablation arm {arm} unexpectedly altered quantizer decisions "
            f"under formulation={formulation!r}."
        )


def test_power_mean_base_and_fedmaq_configs_agree_on_shared_keys():
    """Assert _power_mean_base.yaml and fedmaq.yaml agree on all shared keys.

    Key sets must differ only by formulation and the two exponents (gamma1, gamma2),
    and all shared values must be equal. Must fail if either file's memory unit is changed alone.
    """
    fedmaq_cfg = OmegaConf.to_container(
        OmegaConf.load(Path(CONF_DIR) / "algorithm" / "fedmaq.yaml"), resolve=True
    )
    power_mean_base_cfg = OmegaConf.to_container(
        OmegaConf.load(Path(CONF_DIR) / "algorithm" / "_power_mean_base.yaml"), resolve=True
    )

    fedmaq_keys = set(fedmaq_cfg.keys())
    base_keys = set(power_mean_base_cfg.keys())

    # Key sets must differ ONLY by formulation and the two exponents
    assert fedmaq_keys - base_keys == {"formulation", "gamma1", "gamma2"}, (
        f"fedmaq.yaml has unexpected keys not in _power_mean_base.yaml: "
        f"{fedmaq_keys - base_keys - {'formulation', 'gamma1', 'gamma2'}}"
    )
    assert base_keys - fedmaq_keys == set(), (
        f"_power_mean_base.yaml has keys not in fedmaq.yaml: {base_keys - fedmaq_keys}"
    )

    # All shared keys must have equal values
    for k in base_keys:
        assert fedmaq_cfg[k] == power_mean_base_cfg[k], (
            f"Config discrepancy on shared key {k!r}: "
            f"fedmaq.yaml={fedmaq_cfg[k]!r} vs _power_mean_base.yaml={power_mean_base_cfg[k]!r}"
        )


def test_stage_1b_p_is_explicit_and_fails_closed_until_selection():
    """Stage 1b must carry `p` as an override, and must refuse to dispatch unresolved.

    ADR-0021 D2 requires the omega follow-up to run at the `p` Stage 1a selected, and
    this matrix is the only thing that puts `p` on those runs. `_power_mean_base.yaml`
    defaults `p: 0`, which is itself a Stage-1a candidate, and neither `variant` nor
    `identity_key` encodes `p` -- so a run that silently took the default is
    indistinguishable on disk from one at the selected degree, and `--skip_completed`
    would treat it as already done. A comment naming the placeholder is not enough:
    the override must exist, and while unresolved it must be unreadable rather than
    plausible.

    Bidirectional, so it does not block the author's Stage-1a write-back. It holds
    while `p` is the `???` sentinel (dispatch raises before the first round) and after
    a concrete degree is written in (dispatch composes and plans normally). The only
    state it rejects is the lossy one: `p` absent, or resolvable to a default that
    no selection chose.
    """
    matrix = _matrix("power_mean_omega")
    for run in matrix["runs"]:
        overrides = run.get("overrides") or []
        values = {
            key.strip(): value.strip()
            for key, _, value in (override.partition("=") for override in overrides)
        }
        assert "algorithm.p" in values, (
            f"Stage-1b run {run['label']!r} sets no algorithm.p override, so Hydra "
            f"resolves it to the _power_mean_base.yaml default p=0 -- a Stage-1a "
            f"candidate that no selection chose."
        )

        with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
            composed = compose(
                config_name="config",
                overrides=[f"algorithm={run['alg']}", *overrides],
            )

        if values["algorithm.p"] == "???":
            with pytest.raises(MissingMandatoryValue):
                _ = composed.algorithm.p
        else:
            resolved = composed.algorithm.p
            assert resolved == "min" or math.isfinite(float(resolved)), (
                f"Stage-1b run {run['label']!r} resolves algorithm.p to {resolved!r}, "
                f"which is neither a finite degree nor 'min'."
            )


def _planner_probe_matrix(runs: list[dict]) -> dict:
    return {
        "phase": "smoke",
        "experiment_group": "planner_probe",
        "dataset": "cifar10",
        "model": "mobilenetv2",
        "total_rounds": 1,
        "client_gpus": 0.0,
        "ledger": "smoke",
        "seeds": [0],
        "heterogeneities": ["dirichlet_alpha_0.1"],
        "runs": runs,
    }


def test_dispatch_refuses_an_override_left_at_the_missing_sentinel(tmp_path):
    """Planning must reject `???` rather than relying on the consumer to raise --
    see `test_the_missing_sentinel_is_not_safe_by_itself` for why it cannot.
    """
    from scripts.matrix_planner import plan_matrix

    for override in ("algorithm.p=???", "algorithm.p='???'", "+algorithm.p=???"):
        matrix = _planner_probe_matrix(
            [{"alg": "power_mean", "label": "omega-probe", "overrides": [override]}]
        )
        # The guard is only possible because the token is an element of an `overrides`
        # list, not a config node: resolution passes it through instead of raising.
        resolved = OmegaConf.to_container(OmegaConf.create(matrix), resolve=True)
        assert resolved["runs"][0]["overrides"] == [override]

        with pytest.raises(ValueError, match=r"omega-probe.*algorithm\.p"):
            plan_matrix(tmp_path / "probe.yaml", matrix)


def test_the_missing_sentinel_is_not_safe_by_itself():
    """Why the guard sits in the planner and not at the consumers.

    `???` fails closed only under a subscript. `quantization_planner.py` subscripts
    `p` only when the formulation requires it, and `manifest.py` records the run with
    `.get` -- so an unresolved `p` can reach a run as a plausible default *and* leave
    no trace of the placeholder in that run's provenance.

    If OmegaConf ever makes `.get` raise, this test fails and the guard's rationale
    narrows; the guard itself still stands, because booleans remain fail-open.
    """
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(
            config_name="config",
            overrides=["algorithm=power_mean", "algorithm.p=???", "algorithm.omega=0.25"],
        )

    with pytest.raises(MissingMandatoryValue):
        _ = cfg.algorithm["p"]
    assert cfg.algorithm.get("p", 0.0) == 0.0
    assert cfg.algorithm.get("p") is None
    assert OmegaConf.to_container(OmegaConf.create({"k": "???"}), resolve=True).get("k", False)


def test_dispatch_refuses_a_run_declaring_pending_selection(tmp_path):
    """A placeholder whose value is *plausible* needs a declared marker, not a comment.

    `algorithm.soft_voting=true` under a `# PLACEHOLDER` comment is indistinguishable
    from a decided value: it is well-formed, it composes, and it trains. The run
    therefore declares its own unresolved keys in a field the planner reads, and
    clearing that field is the act that records the decision.
    """
    from scripts.matrix_planner import plan_matrix

    matrix = _planner_probe_matrix(
        [
            {
                "alg": "fedmaq",
                "label": "surviving-set-probe",
                "pending_selection": ["soft_voting"],
                "overrides": ["algorithm.soft_voting=true"],
            }
        ]
    )
    with pytest.raises(ValueError, match=r"surviving-set-probe.*soft_voting"):
        plan_matrix(tmp_path / "probe.yaml", matrix)


def test_a_misspelt_run_spec_key_is_rejected_rather_than_ignored(tmp_path):
    """`expand_matrix` reads run-spec keys through `.get`, so `pending_selections`
    would be silently ignored and the guard would never fire -- fail-open, relocated
    from the value to the key. The planner therefore rejects unknown keys outright.
    """
    from scripts.matrix_planner import plan_matrix

    matrix = _planner_probe_matrix(
        [
            {
                "alg": "fedmaq",
                "label": "typo-probe",
                "pending_selections": ["soft_voting"],
                "overrides": ["algorithm.soft_voting=true"],
            }
        ]
    )
    with pytest.raises(ValueError, match=r"typo-probe.*pending_selections"):
        plan_matrix(tmp_path / "probe.yaml", matrix)


def test_dispatch_plans_normally_once_the_selection_is_written_in(tmp_path):
    """Bidirectional, so the guard cannot become a permanent block on the campaign."""
    from scripts.matrix_planner import plan_matrix

    matrix = _planner_probe_matrix(
        [
            {"alg": "power_mean", "label": "omega-probe", "overrides": ["algorithm.p=0"]},
            {
                "alg": "fedmaq",
                "label": "surviving-set-probe",
                "variant": "surviving-set",
                "pending_selection": [],
                "overrides": ["algorithm.soft_voting=true"],
            },
        ]
    )
    plan = plan_matrix(tmp_path / "probe.yaml", matrix)
    assert [task.label for task in plan.tasks] == [
        "omega-probe-dirichlet_alpha_0.1-seed0",
        "surviving-set-probe-dirichlet_alpha_0.1-seed0",
    ]


def test_the_remaining_pre_selection_matrix_refuses_to_dispatch_today():
    """`power_mean_omega` left this guard when ADR-0024 D1 / ADR-0026 resolved its
    `algorithm.p=???` write-in. Expected to fail the moment the author resolves
    `pass3_freeze_confirm` too -- that is the signal to delete this test, not to
    weaken the guard.
    """
    from scripts.matrix_planner import plan_matrix

    for name in ("pass3_freeze_confirm",):
        path = Path(CONF_DIR) / "matrix" / f"{name}.yaml"
        with pytest.raises(ValueError, match="unresolved"):
            plan_matrix(path, OmegaConf.load(path))


def test_every_registered_matrix_either_matches_its_contract_or_is_pre_selection():
    """`matrix_contracts` is hand-maintained and had no test; editing a registered
    matrix silently breaks its own dispatch, which `just check` cannot see.

    A pre-selection matrix is exempt from the fingerprint because it is still being
    written -- but only while the guard refuses to dispatch it, so the exemption cannot
    be claimed by a matrix that would actually run. The exemption is derived from
    `unresolved_selection` rather than an allowlist, so it expires on write-in.
    """
    from fedmaq.core.protocol import REPLACEMENT_PROTOCOL, validate_matrix_against_protocol
    from scripts.common import expand_matrix
    from scripts.matrix_planner import plan_matrix, unresolved_selection

    protocol_path = Path(CONF_DIR) / "protocol" / f"{REPLACEMENT_PROTOCOL}.yaml"
    protocol = OmegaConf.to_container(OmegaConf.load(protocol_path), resolve=True)
    contracts = protocol["matrix_contracts"]
    assert contracts, "protocol registers no matrix contracts"

    for name in contracts:
        path = Path(CONF_DIR) / "matrix" / f"{name}.yaml"
        assert path.exists(), f"protocol registers {name!r} but conf/matrix has no such matrix"
        matrix = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        pre_selection = any(unresolved_selection(run) for run in matrix.get("runs") or [])

        if pre_selection:
            with pytest.raises(ValueError, match="unresolved"):
                plan_matrix(path, OmegaConf.load(path))
        else:
            validate_matrix_against_protocol(name, matrix, len(expand_matrix(matrix, name)))
