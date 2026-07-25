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


def test_primary_grid_turns_the_post_process_pipeline_on():
    """§4.3 applies difference coding, error compensation, and lossless encoding
    to FedMAQ for primary benchmarking.

    ``post_process`` ships false in every conf/algorithm/*.yaml, deliberately, so
    the only thing standing between the manuscript's promise and a grid that
    silently reports uncompressed payload numbers is this override. A comment is
    not a guard; a plausible-looking communication table is exactly the kind of
    defect nobody questions after the fact.
    """
    enabled = _post_process_overrides(_matrix("benchmark_grid"))
    assert enabled.get("fedmaq") is True, (
        "conf/matrix/benchmark_grid.yaml must run FedMAQ with "
        "algorithm.post_process=true. Without it the primary grid measures "
        "communication without the §4.3 pipeline and reports it as if it had."
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
