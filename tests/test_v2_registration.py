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
