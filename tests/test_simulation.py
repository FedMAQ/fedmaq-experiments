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


# Manuscript §4.3.7 defines the ablation as a leave-one-out design: each arm is
# full FedMAQ with exactly one thing removed. That property is what makes an
# arm's delta against Configuration 7 attributable to the component it drops.
# It lives entirely in YAML, so nothing else in the suite can catch a config edit
# that quietly reintroduces a second difference — which is the defect that made
# the pre-2026-07-25 arms unattributable. Each entry below is the complete set of
# keys an arm may differ from ``fedmaq.yaml`` in.
ABLATION_ARM_DIFFS = {
    # Configuration 2: Tier-1 memory ceiling lifted.
    "fedmaq_no_resource": {"resource_aware"},
    # Configuration 3: Formulation 3's data modulator removed at kappa = 0.
    # One key, because the arm stays on the winner's formulation.
    "fedmaq_no_data": {"lambda_val"},
    # Configuration 4: the pre-registered fallback arm. Formulation 3 carries no
    # weight on the gradient term and cannot express state-awareness removal, so
    # this arm alone changes formulation, and is anchored to the formulation
    # study's own Formulation 1 runs rather than to Configuration 7.
    "fedmaq_no_state": {"formulation", "gamma1", "gamma2"},
    # Configuration 5: KD removed. soft_voting goes with it as INAPPLICABLE
    # (it weights teacher logits, and this arm distills nothing).
    "fedmaq_no_kd": {"kd_epochs", "soft_voting"},
    # Configuration 8: the frozen refinement layer removed.
    "fedmaq_no_refinements": {"soft_voting", "ema_student", "grad_norm_ema"},
}


def _algorithm_cfg(algorithm):
    with initialize_config_dir(config_dir=CONF_DIR, version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"algorithm={algorithm}"])
    return OmegaConf.to_container(cfg.algorithm, resolve=True)


@pytest.mark.parametrize("arm,expected_diff", sorted(ABLATION_ARM_DIFFS.items()))
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
    refinements = ("soft_voting", "ema_student", "grad_norm_ema")
    full = _algorithm_cfg("fedmaq")
    # Arms that remove an awareness signal must carry the layer untouched.
    for arm in ("fedmaq_no_resource", "fedmaq_no_data", "fedmaq_no_state"):
        arm_cfg = _algorithm_cfg(arm)
        for flag in refinements:
            assert arm_cfg[flag] == full[flag], (
                f"{arm} deviates from the shared refinement layer on {flag}; "
                f"§4.3.7 requires the difference between arms to be the "
                f"awareness signal alone."
            )

    # ema_student is quantization-independent, so the no-quantization arm carries
    # it too. The other two act on a quantization signal it never produces.
    fedavg_kd = _algorithm_cfg("fedavg_kd")
    assert fedavg_kd["ema_student"] is True
    assert fedavg_kd["soft_voting"] is False
    assert fedavg_kd["grad_norm_ema"] is False


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


@pytest.mark.parametrize(
    "matrix_name",
    ["benchmark_grid", "ablation", "uniform_memory_control", "formulation_study"],
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
