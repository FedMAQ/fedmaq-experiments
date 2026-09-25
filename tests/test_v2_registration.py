"""ADR-0016's 2026-09-25 V2 registration invariants the contract hashes cannot state."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

MATRIX_DIR = Path(__file__).resolve().parents[1] / "conf" / "matrix"
V2_MATRICES = ("v2_confirm_cifar10", "v2_confirm_cifar100", "v2_confirm_femnist")
V2_SEEDS = {19, 37, 73, 101, 131}


def _load(path: Path) -> dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _seeds(matrix: dict) -> set[int]:
    seeds = {int(seed) for seed in matrix.get("seeds") or []}
    for run in matrix.get("runs") or []:
        seeds.update(int(seed) for seed in run.get("seeds") or [])
    return seeds


def test_v2_seeds_have_never_run_in_any_other_matrix() -> None:
    """A seed any earlier matrix used may have shaped a selection or a reading."""
    others = [path for path in MATRIX_DIR.glob("*.yaml") if path.stem not in V2_MATRICES]
    assert others
    for name in V2_MATRICES:
        assert _seeds(_load(MATRIX_DIR / f"{name}.yaml")) == V2_SEEDS
    for path in others:
        overlap = _seeds(_load(path)) & V2_SEEDS
        assert not overlap, f"{path.name} runs V2 confirmation seed(s) {sorted(overlap)}"


def test_v2_output_namespace_is_disjoint_from_every_other_matrix() -> None:
    for name in V2_MATRICES:
        matrix = _load(MATRIX_DIR / f"{name}.yaml")
        assert (matrix["phase"], matrix["experiment_group"]) == ("v2", "v2_confirm")
        assert (matrix["stage"], matrix["protocol_stage"], matrix["split"]) == (
            "v2_confirm",
            "v2_confirm",
            "test",
        )
    for path in MATRIX_DIR.glob("*.yaml"):
        if path.stem in V2_MATRICES:
            continue
        matrix = _load(path)
        assert matrix.get("phase") != "v2", path.name
        assert matrix.get("experiment_group") != "v2_confirm", path.name
        assert matrix.get("protocol_stage") != "v2_confirm", path.name


def test_fedkd_preflight_is_the_three_failed_cells_with_only_the_detach_changed() -> None:
    """ADR-0028 item 6, gate 2: the first-study FedKD arm plus the detach, nothing else."""
    from scripts.common import expand_matrix

    matrix = _load(MATRIX_DIR / "v2_fedkd_preflight.yaml")
    assert (matrix["stage"], matrix["protocol_stage"], matrix["split"]) == (
        "v2_fedkd_preflight",
        "v2_fedkd_preflight",
        "test",
    )
    assert (matrix["dataset"], matrix["total_rounds"]) == ("cifar10", 100)
    specs = expand_matrix(matrix, "v2_fedkd_preflight")
    assert sorted(spec["seed"] for spec in specs) == [0, 42, 123]
    for spec in specs:
        assert (spec["algorithm_config"], spec["heterogeneity"]) == ("fedkd", "dirichlet_alpha_0.1")
        assert spec["overrides"] == ["+algorithm.detach_kl_targets=true"]


def test_fedkd_preflight_output_namespace_is_its_own() -> None:
    for path in MATRIX_DIR.glob("*.yaml"):
        matrix = _load(path)
        owns = path.stem == "v2_fedkd_preflight"
        assert (matrix.get("experiment_group") == "v2_fedkd_preflight") is owns, path.name
        assert (matrix.get("protocol_stage") == "v2_fedkd_preflight") is owns, path.name


SCREEN_MATRICES = ("v2_kd_screen_cifar10", "v2_kd_screen_cifar100", "v2_kd_screen_femnist")


def test_kd_screen_runs_one_repair_family_per_arm_on_validation() -> None:
    """ADR-0028 items 2-3: 18 single-family arms plus both references, all piped."""
    from hydra import compose, initialize_config_dir

    from fedmaq.core.kd_repair import resolve_kd_repair
    from scripts.common import expand_matrix

    families: dict[str, set[str]] = {}
    with initialize_config_dir(config_dir=str(MATRIX_DIR.parent), version_base="1.3"):
        for name in SCREEN_MATRICES:
            matrix = _load(MATRIX_DIR / f"{name}.yaml")
            assert (matrix["phase"], matrix["experiment_group"]) == ("v2_kd", "v2_kd_screen")
            assert (matrix["stage"], matrix["protocol_stage"], matrix["split"]) == (
                "v2_kd_screen",
                "v2_kd_screen",
                "val",
            )
            assert _seeds(matrix) == {0, 42, 123}
            arms = {run.get("variant") or run["alg"]: run for run in matrix["runs"]}
            assert len(arms) == 20 == len(matrix["runs"])
            for label, run in arms.items():
                selections = [
                    f"dataset={matrix['dataset']}",
                    f"heterogeneity={matrix['heterogeneities'][0]}",
                    f"algorithm={run['alg']}",
                ]
                if matrix.get("experiment"):
                    selections.append(f"experiment={matrix['experiment']}")
                cfg = compose(config_name="config", overrides=selections + run["overrides"])
                algorithm = OmegaConf.to_container(cfg.algorithm, resolve=True)
                assert algorithm["post_process"] is True, (name, label)
                family = resolve_kd_repair(algorithm).family
                if label in ("fedmaq_no_kd", "fedmaq"):
                    assert family is None, (name, label)
                else:
                    assert run["alg"] == "fedmaq", (name, label)
                    families.setdefault(family, set()).add(label)
            specs = expand_matrix(matrix, name)
            assert len(specs) == 20 * 3 * len(matrix["heterogeneities"])
    assert {family: len(labels) for family, labels in families.items()} == {
        "schedule": 3,
        "temperature": 3,
        "teacher_selection": 3,
        "teacher_weighting": 3,
        "guard": 3,
        "class_balanced": 3,
    }


def test_kd_screen_output_namespace_is_its_own() -> None:
    for path in MATRIX_DIR.glob("*.yaml"):
        matrix = _load(path)
        owns = path.stem in SCREEN_MATRICES
        assert (matrix.get("experiment_group") == "v2_kd_screen") is owns, path.name
        assert (matrix.get("protocol_stage") == "v2_kd_screen") is owns, path.name
