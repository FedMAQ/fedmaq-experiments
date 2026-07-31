"""Phase 0 safety-net tests: Hydra config composition + in-process run(cfg) smoke.

These guard the orchestration path that the hand-rolled unit tests in
``test_environment.py`` never exercise: that every ``conf/algorithm/*.yaml`` composes
into a valid config, and that the decorator-free :func:`fedmaq.simulation.run` entry
point drives the real ``client_fn``/``server_fn`` wiring end-to-end.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

CONF_DIR = str((Path(__file__).parent.parent / "conf").resolve())

# Every selectable algorithm config (including the FedDistill/CFD stubs, whose YAML
# must still compose even though their hooks are not yet implemented).
ALGORITHM_CONFIGS = [
    "fedavg",
    "fedprox",
    "fedpaq",
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


# Manuscript §4.3.7 defines the ablation as a leave-one-out design: each arm is
# full FedMAQ with exactly one thing removed. That property is what makes an
# arm's delta against Configuration 7 attributable to the component it drops.
# It lives entirely in YAML, so nothing else in the suite can catch a config edit
# that quietly reintroduces a second difference — which is the defect that made
# the pre-2026-07-25 arms unattributable. Each entry is the complete set of keys
# an arm may differ from ``fedmaq.yaml`` in.
#
# Two entries are computed from the freeze rather than written down. A literal
# list would have to be hand-edited in lockstep with fedmaq.yaml at exactly the
# moment the freeze lands, and a test that needs editing to keep passing after a
# config change is not a guard against that change. Deriving them means the
# freeze can write any surviving set -- including the empty one -- and these
# tests still describe the arms correctly.
def _ablation_arm_diffs():
    frozen = _frozen_refinements()
    return {
        # Configuration 2: Tier-1 memory ceiling lifted.
        "fedmaq_no_resource": {"resource_aware"},
        # Configuration 3: Formulation 3's data modulator removed at kappa = 0.
        # One key, because the arm stays on the winner's formulation.
        "fedmaq_no_data": {"lambda_val"},
        # Configuration 4: the pre-registered fallback arm. Formulation 3 carries
        # no weight on the gradient term and cannot express state-awareness
        # removal, so this arm alone changes formulation, and is anchored to the
        # formulation study's own Formulation 1 runs rather than to Config 7.
        "fedmaq_no_state": {"formulation", "gamma1", "gamma2"},
        # Configuration 5: KD removed. soft_voting goes with it as INAPPLICABLE
        # (it weights teacher logits, and this arm distills nothing) -- but it
        # only registers as a *difference* if the freeze turned it on.
        "fedmaq_no_kd": {"kd_epochs"} | ({"soft_voting"} & frozen),
        # Configuration 8: the frozen refinement layer removed. Removing the
        # layer is removing exactly what was frozen, no more and no less.
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
    # Arms that remove an awareness signal must carry the layer untouched.
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
    assert len(matrix["runs"]) == 7, (
        "§4.3.7 dispatches 7 net-new arms (42 runs); only Configuration 1, "
        "uncompressed FedAvg, is inherited from the primary grid."
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
        f"conf/matrix/baseline_tuning.yaml dispatches {total} runs; Decision 67 "
        "and docs/RUNBOOK.md Stage 1b both quote 55 (5 baselines x (5 + 3 + 3))."
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
    (tmp_path / MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    (tmp_path / "experiment_log.csv").write_text("round\n", encoding="utf-8")
    assert not is_run_complete(tmp_path)

    (tmp_path / FINAL_MODEL_FILENAME).write_bytes(b"")
    assert is_run_complete(tmp_path)

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
    (done / FINAL_MODEL_FILENAME).write_bytes(b"")

    dispatched.clear()
    monkeypatch.setattr(sys, "argv", ["run_matrix.py", "--matrix", "probe", "--skip_completed"])
    run_matrix.main()

    assert len(dispatched) == 1, "the completed arm should not have been re-dispatched"
    assert "algorithm=fedavg" in dispatched[0]


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
