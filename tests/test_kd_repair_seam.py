"""The KD-repair seam: default-off parity, one-family refusal, and repair telemetry (#126)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from torch.utils.data import DataLoader, TensorDataset

from fedmaq.core import kd_utils
from fedmaq.core.kd_repair import (
    KD_REPAIR_FAMILIES,
    participant_class_counts,
    resolve_kd_repair,
    validate_kd_repair_config,
)
from fedmaq.core.kd_utils import distill_ensemble_into_global, run_server_side_kd
from fedmaq.core.models import get_model_parameters, set_model_parameters

NUM_CLASSES = 4
NUM_PUBLIC = 24


def _factory(_dataset: str, num_classes: int) -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(6, num_classes))


def _weights(seed: int) -> list[np.ndarray]:
    torch.manual_seed(seed)
    return get_model_parameters(_factory("toy", NUM_CLASSES))


def _results(n: int) -> list[tuple[object, FitRes]]:
    status = Status(code=Code.OK, message="")
    return [
        (object(), FitRes(status, ndarrays_to_parameters(_weights(10 + i)), 5, {}))
        for i in range(n)
    ]


@pytest.fixture
def public_loader(monkeypatch):
    generator = torch.Generator().manual_seed(7)
    images = torch.randn(NUM_PUBLIC, 6, generator=generator)
    dataset = TensorDataset(images, torch.zeros(NUM_PUBLIC, dtype=torch.long))

    def loaders(*_args, **_kwargs):
        return DataLoader(dataset, batch_size=8, shuffle=False), None, None

    monkeypatch.setattr(kd_utils, "get_server_loaders", loaders)
    return loaders


def _reference(results, alg_cfg, teacher_bit_widths=None) -> list[np.ndarray]:
    """The unrepaired pass, rebuilt from the KD primitives the seam wraps."""
    student = _factory("toy", NUM_CLASSES)
    set_model_parameters(student, _weights(1))
    teachers = []
    for _, fit_res in results:
        teacher = _factory("toy", NUM_CLASSES)
        set_model_parameters(teacher, parameters_to_ndarrays(fit_res.parameters))
        teacher.eval()
        teachers.append(teacher)
    loader, _, _ = kd_utils.get_server_loaders("toy", list(range(NUM_PUBLIC)), batch_size=8)
    run_server_side_kd(
        student_model=student,
        teachers=teachers,
        public_loader=loader,
        temperature=float(alg_cfg.get("temperature", 1.0)),
        learning_rate=float(alg_cfg.get("server_kd_lr", 0.01)),
        momentum=float(alg_cfg.get("server_kd_momentum", 0.9)),
        epochs=int(alg_cfg.get("kd_epochs", 1)),
        device=torch.device("cpu"),
        teacher_bit_widths=teacher_bit_widths,
        entropy_weight_scale=float(alg_cfg.get("entropy_weight", 1.0)),
        precision_weight_scale=float(alg_cfg.get("precision_weight", 1.0)),
    )
    return get_model_parameters(student)


def _seam(results, alg_cfg, teacher_bit_widths=None, **kwargs):
    return distill_ensemble_into_global(
        model_factory=_factory,
        aggregated_parameters=ndarrays_to_parameters(_weights(1)),
        results=results,
        public_indices=list(range(NUM_PUBLIC)),
        dataset_name="toy",
        num_classes=NUM_CLASSES,
        batch_size=8,
        alg_cfg=alg_cfg,
        device=torch.device("cpu"),
        teacher_bit_widths=teacher_bit_widths,
        **kwargs,
    )


ALG_CFG = {"name": "fedmaq", "temperature": 3.0, "kd_epochs": 2, "server_kd_lr": 0.05}
ALL_OFF = {family: {"enabled": False} for family in KD_REPAIR_FAMILIES}


@pytest.mark.parametrize("bit_widths", [None, [8, 4, 2]])
@pytest.mark.parametrize("repair_group", [None, {}, ALL_OFF])
def test_default_repair_config_is_bit_identical(public_loader, bit_widths, repair_group):
    alg_cfg = dict(ALG_CFG) if repair_group is None else {**ALG_CFG, "kd_repair": repair_group}
    results = _results(3)
    updated, _ = _seam(
        results, alg_cfg, bit_widths, server_round=4, class_counts=[[1, 2, 3, 4]] * 3
    )
    expected = _reference(results, ALG_CFG, bit_widths)
    for got, want in zip(parameters_to_ndarrays(updated), expected, strict=True):
        assert np.array_equal(got, want)


def test_unrepaired_telemetry(public_loader):
    _, metrics = _seam(
        _results(3), ALG_CFG, num_public_samples=NUM_PUBLIC, server_compute_speed=12.0
    )
    assert metrics["kd_skipped"] == 0.0
    assert metrics["kd_applied_weight"] == 1.0
    assert metrics["kd_effective_teachers"] == 3.0
    assert metrics["kd_teacher_weight_min"] == metrics["kd_teacher_weight_max"] == 1.0 / 3
    assert metrics["kd_skipped_batches"] == 0.0
    assert metrics["kd_guard_accept"] == 1.0
    assert metrics["kd_server_sim_time"] == NUM_PUBLIC * 2 * 3 / 12.0


def test_skipped_pass_reports_zero_weight():
    _, metrics = distill_ensemble_into_global(
        model_factory=_factory,
        aggregated_parameters=ndarrays_to_parameters(_weights(1)),
        results=_results(2),
        public_indices=None,
        dataset_name="toy",
        num_classes=NUM_CLASSES,
        batch_size=8,
        alg_cfg=ALG_CFG,
        device=torch.device("cpu"),
    )
    assert metrics["kd_skipped"] == 1.0
    assert metrics["kd_applied_weight"] == 0.0


def test_two_families_fail_at_config_resolution():
    config = {
        "algorithm": {
            "name": "fedmaq",
            "kd_repair": {"schedule": {"enabled": True}, "guard": {"enabled": True}},
        }
    }
    with pytest.raises(ValueError, match="one repair family per arm"):
        validate_kd_repair_config(config)


def test_unknown_family_is_refused():
    with pytest.raises(ValueError, match="Unknown KD-repair families"):
        resolve_kd_repair({"kd_repair": {"mixup": {"enabled": True}}})


def test_single_family_resolves_with_its_settings():
    repair = resolve_kd_repair(
        {"kd_repair": {"class_balanced": {"enabled": True, "power": 0.5}, "guard": {}}}
    )
    assert repair.family == "class_balanced"
    assert dict(repair.settings) == {"power": 0.5}
    assert repair.needs_class_counts


def test_enabled_family_is_refused_until_implemented(public_loader):
    alg_cfg = {**ALG_CFG, "kd_repair": {"temperature": {"enabled": True}}}
    with pytest.raises(NotImplementedError, match="temperature"):
        _seam(_results(2), alg_cfg)


def test_participant_class_counts_follow_partition_order():
    labels = np.array([0, 1, 1, 2, 3, 3, 3])
    client_indices = {"0": [0, 1, 2], "1": [3, 4, 5, 6]}
    assert participant_class_counts([1, 0], client_indices, labels, NUM_CLASSES) == [
        [0, 0, 1, 3],
        [1, 2, 0, 0],
    ]
