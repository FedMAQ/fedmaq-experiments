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


# ADR-0028 item 5: the KD-repair confirmation reuses the V2 seeds; no other stage may.
V2_SEED_STAGES = ("v2_confirm", "v2_kd_confirm", "v2_extension")


def _v2_seed_violation(matrix: dict) -> str | None:
    """Why ``matrix`` breaks V2 seed exclusivity, or ``None`` when it keeps it."""
    seeds = _seeds(matrix)
    if matrix.get("protocol_stage") in V2_SEED_STAGES:
        return None if seeds == V2_SEEDS else f"runs seeds {sorted(seeds)}, not the V2 seeds"
    overlap = seeds & V2_SEEDS
    return f"runs V2 confirmation seed(s) {sorted(overlap)}" if overlap else None


def test_v2_seeds_have_never_run_in_any_other_matrix() -> None:
    """A seed any earlier matrix used may have shaped a selection or a reading."""
    others = [path for path in MATRIX_DIR.glob("*.yaml") if path.stem not in V2_MATRICES]
    assert others
    for name in V2_MATRICES:
        assert _load(MATRIX_DIR / f"{name}.yaml")["protocol_stage"] == "v2_confirm"
    for path in MATRIX_DIR.glob("*.yaml"):
        violation = _v2_seed_violation(_load(path))
        assert violation is None, f"{path.name} {violation}"


def test_v2_seed_guard_admits_the_kd_confirmation_and_no_other_stage() -> None:
    assert (
        _v2_seed_violation({"protocol_stage": "v2_kd_confirm", "seeds": sorted(V2_SEEDS)}) is None
    )
    assert _v2_seed_violation({"protocol_stage": "v2_kd_confirm", "seeds": [19, 37]})
    for stage in ("v2_kd_screen", "v2_fedkd_preflight", "v2_fedkd_first_study", "benchmark", None):
        assert _v2_seed_violation({"protocol_stage": stage, "seeds": [0, 19]}), stage
        assert _v2_seed_violation({"protocol_stage": stage, "runs": [{"seeds": [131]}]}), stage


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


FIRST_STUDY_MATRICES = (
    "v2_fedkd_first_study_cifar10",
    "v2_fedkd_first_study_cifar100",
    "v2_fedkd_first_study_femnist",
)


def test_fedkd_first_study_is_the_other_twelve_cells_with_only_the_detach_changed() -> None:
    """ADR-0028 item 6: the first-study FedKD arm on its grid, less the preflight cells."""
    from scripts.common import expand_matrix

    first_study = {
        (spec["dataset"], spec["heterogeneity"], spec["seed"]): spec
        for name in ("benchmark_grid", "benchmark_grid_cifar100", "benchmark_grid_femnist")
        for spec in expand_matrix(_load(MATRIX_DIR / f"{name}.yaml"), name)
        if spec["algorithm_config"] == "fedkd"
    }
    preflight = {
        (spec["dataset"], spec["heterogeneity"], spec["seed"])
        for spec in expand_matrix(
            _load(MATRIX_DIR / "v2_fedkd_preflight.yaml"), "v2_fedkd_preflight"
        )
    }
    cells = set()
    for name in FIRST_STUDY_MATRICES:
        matrix = _load(MATRIX_DIR / f"{name}.yaml")
        assert (matrix["phase"], matrix["experiment_group"]) == ("v2_kd", "v2_fedkd_first_study")
        assert (matrix["stage"], matrix["protocol_stage"], matrix["split"]) == (
            "v2_fedkd_first_study",
            "v2_fedkd_first_study",
            "test",
        )
        for spec in expand_matrix(matrix, name):
            key = (spec["dataset"], spec["heterogeneity"], spec["seed"])
            source = first_study[key]
            assert (spec["model"], spec["total_rounds"]) == (source["model"], 100), key
            assert spec["algorithm_config"] == "fedkd", key
            assert spec["overrides"] == [
                *source["overrides"],
                "+algorithm.detach_kl_targets=true",
            ], key
            cells.add(key)
    assert len(first_study) == 15
    assert cells == set(first_study) - preflight
    assert len(cells) == 12


def test_fedkd_first_study_output_namespace_is_its_own() -> None:
    for path in MATRIX_DIR.glob("*.yaml"):
        matrix = _load(path)
        owns = path.stem in FIRST_STUDY_MATRICES
        assert (matrix.get("experiment_group") == "v2_fedkd_first_study") is owns, path.name
        assert (matrix.get("protocol_stage") == "v2_fedkd_first_study") is owns, path.name


SCREEN_MATRICES = ("v2_kd_screen_cifar10", "v2_kd_screen_cifar100", "v2_kd_screen_femnist")
KD_CONFIRM_MATRICES = {
    "v2_kd_confirm_cifar10": ("cifar10", ["dirichlet_alpha_0.1", "dirichlet_alpha_1.0"], 10),
    "v2_kd_confirm_cifar100": (
        "cifar100",
        ["dirichlet_alpha_0.1", "dirichlet_alpha_1.0"],
        10,
    ),
    "v2_kd_confirm_femnist": ("femnist", ["femnist"], 5),
}

V2_EXTENSION_MATRICES = {
    "v2_extension_cifar10": (
        "cifar10",
        ["dirichlet_alpha_0.1", "dirichlet_alpha_1.0"],
        140,
        None,
    ),
    "v2_extension_cifar100": (
        "cifar100",
        ["dirichlet_alpha_0.1", "dirichlet_alpha_1.0"],
        80,
        None,
    ),
    "v2_extension_femnist": ("femnist", ["femnist"], 40, "femnist"),
}
N2_SCREEN_MATRIX = "v2_n2_screen_femnist"
REGISTERED_SEEDS = {0, 7, 21, 42, 123, 19, 37, 73, 101, 131}

# These are the pre-registration contracts present before v2_kd_confirm was added.
# Keep them pinned so adding a new stage cannot silently bless edits to prior cells.
PREEXISTING_MATRIX_HASHES = {
    "baseline_tuning_wide": "41a4456817dd410f6685feabcce0e8600e36d1489d3bdf7c80e6a208b26603c0",
    "power_mean_design": "ca023c6e6e43b9ba3b643e27dd49000be2f1de4a03e8d76e5ebd8af877496f37",
    "power_mean_omega": "ccafcbbead38430a8697a5a64256430e250fe4bcb7c8cf7a4a48a32ea94bbac3",
    "benchmark_grid": "0ff37127405a3b11d7c00c537782d258fff9c7213acb0f1eecc61262a3f13617",
    "benchmark_grid_cifar100": "af27fe122b5b84f82601a283185c9641e74e2773907a080f54df9503cac58d5f",
    "benchmark_grid_femnist": "6c5ae4801c28cae3055ef8f743da894e297b30d08680d1a657748cc6e9ebe85b",
    "ablation": "290123f30afa9bcc82e18ef08570fa8895d5f881e9a312396bf92fe70a0a6601",
    "fedpaq_pipeline": "cf994fb4ab93c37a8d6da926ad4270987e8c5d01b480d00b3055f0770906bdbd",
    "fedpaq_pipeline_cifar100": "9fd47583db13646ae1a7bf583b4cc563f1e7b4a04315610cb99e6316f6facb99",
    "fedpaq_pipeline_femnist": "3581a7ed8c3fec8a598c7f73ec7684c007c4ad8753256970f5af1799369e4248",
    "memory_sensitivity": "dafaab00c23d29b6dc8d5b5bbc7cd1061c2fd99d2f322e2d81bd248b09cad78e",
    "uniform_memory_control": "0e52e8dec7421985314e249889a4067c27613ff2daf419175bb3c38be3f19a20",
    "v2_confirm_cifar10": "cb41148cc29bd7251a8ca4617c79a199a0e3a7e13fa88e7a572fc6796529245c",
    "v2_confirm_cifar100": "94dc39cf9fdfbe616c4e8890a6cbcbaf9c2d429fdf45576d55073aa2e539251f",
    "v2_confirm_femnist": "c7481c7e4bd01ea6f4c1c41174eb7ec6745c3b813df4f2bb4790555a3cdd9829",
    "v2_fedkd_preflight": "a2d459120076963eb2e7d080d23b409fcb9e5d744ae43d3e3772f0402f12d405",
    "v2_fedkd_first_study_cifar10": (
        "749808293d1b9bbc7b83ed1af808f49e555b7a87fbf8a7a25e2d1fa970a5f0be"
    ),
    "v2_fedkd_first_study_cifar100": (
        "31bdc688e7dcb399c2c302d91c1f2816e0ab65d23cc2fc46c68ac1944b51be08"
    ),
    "v2_fedkd_first_study_femnist": (
        "3cba3b3fa7fb9510a4dfd8990e4acd7ba7f0f2ed74b3f13f41508ee4ebf58675"
    ),
    "v2_kd_screen_cifar10": "2c74bafdcb68893793435e31eaafb6cf1ae2c594d59423295542618b862c2c13",
    "v2_kd_screen_cifar100": "c37944068140ff679c02543f58918f0fcfa2ec667a0a811ffb232223f30bbd17",
    "v2_kd_screen_femnist": "535150abe679d51b1d5b56679403eba43a28321aa1522620c80663cb5dca6e17",
}


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


def test_kd_confirmation_registers_only_detached_fedkd_for_25_cells() -> None:
    from hydra import compose, initialize_config_dir

    from fedmaq.core.protocol import validate_matrix_against_protocol
    from scripts.common import expand_matrix

    protocol = _load(MATRIX_DIR.parent / "protocol" / "replacement-v1.yaml")
    all_tasks = []
    with initialize_config_dir(config_dir=str(MATRIX_DIR.parent), version_base="1.3"):
        for name, (dataset, heterogeneities, expected_count) in KD_CONFIRM_MATRICES.items():
            matrix = _load(MATRIX_DIR / f"{name}.yaml")
            assert (matrix["phase"], matrix["experiment_group"]) == ("v2_kd", "v2_kd_confirm")
            assert (matrix["stage"], matrix["protocol_stage"], matrix["split"]) == (
                "v2_kd_confirm",
                "v2_kd_confirm",
                "test",
            )
            assert matrix["ledger"] == "scientific"
            assert (matrix["dataset"], matrix["heterogeneities"]) == (dataset, heterogeneities)
            assert matrix["total_rounds"] == 100
            assert matrix["seeds"] == [19, 37, 73, 101, 131]
            assert matrix["runs"] == [
                {
                    "alg": "fedkd",
                    "label": "fedkd_detached",
                    "overrides": ["+algorithm.detach_kl_targets=true"],
                }
            ]

            tasks = expand_matrix(matrix, name)
            assert len(tasks) == expected_count
            validate_matrix_against_protocol(name, matrix, len(tasks))
            assert {task["seed"] for task in tasks} == V2_SEEDS
            assert all(
                task["phase"] == "v2_kd"
                and task["stage"] == "v2_kd_confirm"
                and task["protocol_stage"] == "v2_kd_confirm"
                and task["split"] == "test"
                and task["ledger"] == "scientific"
                and task["algorithm_config"] == "fedkd"
                and task["variant"] == ""
                and task["overrides"] == ["+algorithm.detach_kl_targets=true"]
                for task in tasks
            )
            all_tasks.extend(tasks)

            selections = [
                f"dataset={dataset}",
                f"heterogeneity={heterogeneities[0]}",
                "algorithm=fedkd",
            ]
            if matrix.get("experiment"):
                selections.append(f"experiment={matrix['experiment']}")
            cfg = compose(
                config_name="config",
                overrides=selections + ["+algorithm.detach_kl_targets=true"],
            )
            algorithm = OmegaConf.to_container(cfg.algorithm, resolve=True)
            assert algorithm["detach_kl_targets"] is True
            assert algorithm["tmax"] == 0.95
            assert algorithm["post_process"] is False

    assert len(all_tasks) == 25
    assert set(protocol["matrix_contracts"]) == (
        set(PREEXISTING_MATRIX_HASHES)
        | set(KD_CONFIRM_MATRICES)
        | set(V2_EXTENSION_MATRICES)
        | {N2_SCREEN_MATRIX}
    )


def test_kd_confirmation_preserves_preexisting_matrix_contract_hashes() -> None:
    protocol = _load(MATRIX_DIR.parent / "protocol" / "replacement-v1.yaml")
    contracts = protocol["matrix_contracts"]
    actual = {name: contracts[name]["sha256"] for name in PREEXISTING_MATRIX_HASHES}
    assert actual == PREEXISTING_MATRIX_HASHES


def test_kd_confirmation_output_namespace_is_its_own() -> None:
    for path in MATRIX_DIR.glob("*.yaml"):
        matrix = _load(path)
        owns = path.stem in KD_CONFIRM_MATRICES
        assert (matrix.get("experiment_group") == "v2_kd_confirm") is owns, path.name
        assert (matrix.get("protocol_stage") == "v2_kd_confirm") is owns, path.name


def test_v2_extension_registers_all_260_cells_and_fixed_conditions() -> None:
    from hydra import compose, initialize_config_dir

    from fedmaq.core.protocol import PROTOCOL_STAGES, validate_matrix_against_protocol
    from scripts.common import expand_matrix

    protocol = _load(MATRIX_DIR.parent / "protocol" / "replacement-v1.yaml")
    assert {"v2_extension", "v2_n2_screen"} <= PROTOCOL_STAGES
    assert protocol["stages"]["v2_extension"] == {
        "selection_data": "frozen_validation_verdicts",
        "reserved_test": True,
        "split": "test",
    }

    extension_cells = 0
    for name, (
        dataset,
        heterogeneities,
        expected_count,
        experiment,
    ) in V2_EXTENSION_MATRICES.items():
        matrix = _load(MATRIX_DIR / f"{name}.yaml")
        assert (matrix["phase"], matrix["experiment_group"]) == (
            "v2_extension",
            "v2_extension",
        )
        assert (matrix["stage"], matrix["protocol_stage"], matrix["split"]) == (
            "v2_extension",
            "v2_extension",
            "test",
        )
        assert matrix["ledger"] == "scientific"
        assert (matrix["dataset"], matrix["heterogeneities"]) == (dataset, heterogeneities)
        assert matrix.get("experiment") == experiment
        assert matrix["total_rounds"] == 100
        assert matrix["seeds"] == [19, 37, 73, 101, 131]

        expected_labels = {"e1-fedprox", "e1-feddistill"}
        expected_labels |= (
            {f"e2-config{number}" for number in range(2, 8)}
            if dataset == "cifar10"
            else {f"e3-config{number}" for number in range(2, 8)}
        )
        if dataset == "cifar10":
            expected_labels |= {
                "e4-cunit512-fedmaq",
                "e4-cunit512-no-kd",
                "e4-cunit2048-fedmaq",
                "e4-cunit2048-no-kd",
                "e5-fedmaq",
                "e5-no-kd",
            }
        assert {run["label"] for run in matrix["runs"]} == expected_labels

        tasks = expand_matrix(matrix, name)
        assert len(tasks) == expected_count
        validate_matrix_against_protocol(name, matrix, len(tasks))
        if dataset == "cifar10":
            e5_tasks = [task for task in tasks if task["label"].startswith("e5-")]
            assert len(e5_tasks) == 20
            assert {task["heterogeneity"] for task in e5_tasks} == {
                "uniform_memory_alpha_0.1",
                "uniform_memory_alpha_1.0",
            }
        assert {task["seed"] for task in tasks} == V2_SEEDS
        assert all(
            task["phase"] == "v2_extension"
            and task["stage"] == "v2_extension"
            and task["protocol_stage"] == "v2_extension"
            and task["split"] == "test"
            and task["ledger"] == "scientific"
            for task in tasks
        )
        extension_cells += len(tasks)

        with initialize_config_dir(config_dir=str(MATRIX_DIR.parent), version_base="1.3"):
            for run in matrix["runs"]:
                selections = [
                    f"dataset={dataset}",
                    f"heterogeneity={heterogeneities[0]}",
                    f"algorithm={run['alg']}",
                ]
                if experiment:
                    selections.append(f"experiment={experiment}")
                cfg = compose(config_name="config", overrides=selections + run.get("overrides", []))
                algorithm = OmegaConf.to_container(cfg.algorithm, resolve=True)
                category = run["label"].split("-", maxsplit=1)[0]
                assert algorithm["post_process"] is (category in {"e4", "e5"}), run["label"]
                if category == "e4":
                    assert algorithm["c_unit"] in {512.0, 2048.0}, run["label"]
                if category == "e5":
                    assert run["heterogeneities"] == [
                        "uniform_memory_alpha_0.1",
                        "uniform_memory_alpha_1.0",
                    ]
                    for heterogeneity in run["heterogeneities"]:
                        uniform_cfg = compose(
                            config_name="config",
                            overrides=[
                                f"dataset={dataset}",
                                f"heterogeneity={heterogeneity}",
                                f"algorithm={run['alg']}",
                                *run.get("overrides", []),
                            ],
                        )
                        assert uniform_cfg.heterogeneity.uniform_memory_mb == 16384

    assert extension_cells == 260


def test_n2_screen_registers_three_p_values_and_reusable_reference() -> None:
    from hydra import compose, initialize_config_dir

    from fedmaq.core.protocol import validate_matrix_against_protocol
    from scripts.common import expand_matrix

    protocol = _load(MATRIX_DIR.parent / "protocol" / "replacement-v1.yaml")
    assert protocol["stages"]["v2_n2_screen"] == {
        "selection_data": "validation_only",
        "reserved_test": True,
        "split": "val",
    }
    matrix = _load(MATRIX_DIR / f"{N2_SCREEN_MATRIX}.yaml")
    assert (matrix["phase"], matrix["experiment_group"]) == ("v2_n2_screen", "v2_n2_screen")
    assert (matrix["stage"], matrix["protocol_stage"], matrix["split"]) == (
        "v2_n2_screen",
        "v2_n2_screen",
        "val",
    )
    assert (matrix["dataset"], matrix["model"], matrix["experiment"]) == (
        "femnist",
        "simplecnn",
        "femnist",
    )
    assert matrix["seeds"] == [0, 42, 123]
    assert matrix["heterogeneities"] == ["femnist"]
    assert matrix["total_rounds"] == 100
    assert [run["label"] for run in matrix["runs"]] == [
        "n2-p0p5",
        "n2-p0",
        "n2-pminus0p5",
    ]

    tasks = expand_matrix(matrix, N2_SCREEN_MATRIX)
    assert len(tasks) == 9
    validate_matrix_against_protocol(N2_SCREEN_MATRIX, matrix, len(tasks))
    assert {task["seed"] for task in tasks} == {0, 42, 123}
    assert all(
        task["protocol_stage"] == "v2_n2_screen" and task["split"] == "val" for task in tasks
    )

    screen = _load(MATRIX_DIR / "v2_kd_screen_femnist.yaml")
    screen_reference = {
        task["seed"]: task
        for task in expand_matrix(screen, "v2_kd_screen_femnist")
        if task["label"] == "fedmaq_no_kd"
    }
    n2_reference = {task["seed"]: task for task in tasks if task["label"] == "n2-p0p5"}
    assert set(screen_reference) == set(n2_reference) == {0, 42, 123}
    assert all(
        (screen_reference[seed]["dataset"], screen_reference[seed]["heterogeneity"])
        == (n2_reference[seed]["dataset"], n2_reference[seed]["heterogeneity"])
        for seed in screen_reference
    )

    with initialize_config_dir(config_dir=str(MATRIX_DIR.parent), version_base="1.3"):
        for run, expected_p in zip(matrix["runs"], [0.5, 0.0, -0.5], strict=True):
            cfg = compose(
                config_name="config",
                overrides=[
                    "dataset=femnist",
                    "heterogeneity=femnist",
                    "algorithm=fedmaq_no_kd",
                    "experiment=femnist",
                    *run["overrides"],
                ],
            )
            algorithm = OmegaConf.to_container(cfg.algorithm, resolve=True)
            assert algorithm["p"] == expected_p
            assert algorithm["kd_epochs"] == 0
            assert algorithm["post_process"] is True

        reference_run = next(run for run in screen["runs"] if run["label"] == "fedmaq_no_kd")
        screen_cfg = compose(
            config_name="config",
            overrides=[
                "dataset=femnist",
                "heterogeneity=femnist",
                "algorithm=fedmaq_no_kd",
                "experiment=femnist",
                *reference_run.get("overrides", []),
            ],
        )
        n2_cfg = compose(
            config_name="config",
            overrides=[
                "dataset=femnist",
                "heterogeneity=femnist",
                "algorithm=fedmaq_no_kd",
                "experiment=femnist",
                *matrix["runs"][0]["overrides"],
            ],
        )
        assert OmegaConf.to_container(screen_cfg.algorithm, resolve=True) == OmegaConf.to_container(
            n2_cfg.algorithm, resolve=True
        )


def test_fresh_seed_exclusion_is_derived_from_all_matrix_seeds() -> None:
    registered: set[int] = set()
    for path in MATRIX_DIR.glob("*.yaml"):
        registered.update(_seeds(_load(path)))

    assert registered == REGISTERED_SEEDS
    assert ({0, 7, 19, 131, 2026} - registered) == {2026}
