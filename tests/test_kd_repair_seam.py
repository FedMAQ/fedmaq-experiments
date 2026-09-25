"""The KD-repair seam (#126) and its schedule and temperature families (#127)."""

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
    schedule_weight,
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


def test_kd_repair_telemetry(public_loader):
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
    assert metrics["kd_temperature"] == 3.0


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


def test_unimplemented_family_is_refused(public_loader):
    alg_cfg = {**ALG_CFG, "kd_repair": {"guard": {"enabled": True}}}
    with pytest.raises(NotImplementedError, match="guard"):
        _seam(_results(2), alg_cfg)


def test_participant_class_counts_follow_partition_order():
    labels = np.array([0, 1, 1, 2, 3, 3, 3])
    client_indices = {"0": [0, 1, 2], "1": [3, 4, 5, 6]}
    assert participant_class_counts([1, 0], client_indices, labels, NUM_CLASSES) == [
        [0, 0, 1, 3],
        [1, 2, 0, 0],
    ]


def _repair(family: str, **settings) -> dict:
    return {**ALG_CFG, "kd_repair": {family: {"enabled": True, **settings}}}


def _distance(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    return float(sum(np.abs(x - y).sum() for x, y in zip(a, b, strict=True)))


SCHEDULES = {
    "rounds 1-20": {"start": 1, "stop": 20},
    "rounds 1-40": {"start": 1, "stop": 40},
    "start 25, 10-round ramp": {"start": 25, "ramp_rounds": 10},
}


@pytest.mark.parametrize("server_round", [1, 17, 60])
def test_schedule_always_on_is_bit_identical(public_loader, server_round):
    results = _results(2)
    updated, metrics = _seam(results, _repair("schedule", start=1), server_round=server_round)
    expected = _reference(results, ALG_CFG)
    for got, want in zip(parameters_to_ndarrays(updated), expected, strict=True):
        assert np.array_equal(got, want)
    assert metrics["kd_applied_weight"] == 1.0


@pytest.mark.parametrize(
    ("settings", "server_round", "weight"),
    [
        (SCHEDULES["rounds 1-20"], 1, 1.0),
        (SCHEDULES["rounds 1-20"], 20, 1.0),
        (SCHEDULES["rounds 1-20"], 21, 0.0),
        (SCHEDULES["rounds 1-40"], 40, 1.0),
        (SCHEDULES["rounds 1-40"], 41, 0.0),
        (SCHEDULES["start 25, 10-round ramp"], 24, 0.0),
        (SCHEDULES["start 25, 10-round ramp"], 25, 0.1),
        (SCHEDULES["start 25, 10-round ramp"], 34, 1.0),
        (SCHEDULES["start 25, 10-round ramp"], 100, 1.0),
    ],
)
def test_schedule_window_and_ramp_edges(settings, server_round, weight):
    assert schedule_weight(settings, server_round) == pytest.approx(weight)


@pytest.mark.parametrize("name", list(SCHEDULES))
def test_schedule_moves_student_only_inside_its_window(public_loader, name):
    settings = SCHEDULES[name]
    ramped = "ramp_rounds" in settings
    results = _results(2)
    full = _reference(results, ALG_CFG)
    inside = 30 if ramped else 5
    outside = 24 if ramped else settings["stop"] + 1

    off, off_metrics = _seam(results, _repair("schedule", **settings), server_round=outside)
    for got, want in zip(parameters_to_ndarrays(off), _weights(1), strict=True):
        assert np.array_equal(got, want)
    assert off_metrics["kd_applied_weight"] == 0.0
    assert off_metrics["kd_server_sim_time"] == 0.0

    on, on_metrics = _seam(results, _repair("schedule", **settings), server_round=inside)
    weight = on_metrics["kd_applied_weight"]
    assert weight == pytest.approx(0.6 if ramped else 1.0)
    assert _distance(parameters_to_ndarrays(on), _weights(1)) == pytest.approx(
        weight * _distance(full, _weights(1)), rel=1e-5
    )


def test_schedule_without_round_fails_loud(public_loader):
    with pytest.raises(ValueError, match="server round"):
        _seam(_results(2), _repair("schedule", start=1, stop=20))


@pytest.mark.parametrize(
    "settings",
    [{"start": 0}, {"start": 5, "stop": 4}, {"start": 1, "ramp_rounds": 0}, {"begin": 1}],
)
def test_bad_schedule_settings_are_refused(settings):
    with pytest.raises(ValueError, match="schedule"):
        resolve_kd_repair({"kd_repair": {"schedule": {"enabled": True, **settings}}})


FROZEN = {**ALG_CFG, "temperature": 1.0}


def test_temperature_at_frozen_value_is_bit_identical(public_loader):
    results = _results(2)
    alg_cfg = {**FROZEN, "kd_repair": {"temperature": {"enabled": True, "value": 1.0}}}
    updated, metrics = _seam(results, alg_cfg)
    expected = _reference(results, FROZEN)
    for got, want in zip(parameters_to_ndarrays(updated), expected, strict=True):
        assert np.array_equal(got, want)
    assert metrics["kd_temperature"] == 1.0


def _target_entropy(results, temperature: float) -> float:
    images = torch.randn(NUM_PUBLIC, 6, generator=torch.Generator().manual_seed(7))
    probs = []
    for _, fit_res in results:
        teacher = _factory("toy", NUM_CLASSES)
        set_model_parameters(teacher, parameters_to_ndarrays(fit_res.parameters))
        with torch.no_grad():
            probs.append(torch.softmax(teacher(images) / temperature, dim=1))
    target = torch.stack(probs).mean(dim=0)
    return float(-(target * torch.log(target + 1e-12)).sum(dim=1).mean())


def test_temperature_softens_targets_and_reaches_the_pass(public_loader):
    results = _results(2)
    baseline = _reference(results, FROZEN)
    entropies = []
    for value in (0.5, 2.0, 4.0):
        alg_cfg = {**FROZEN, "kd_repair": {"temperature": {"enabled": True, "value": value}}}
        updated, metrics = _seam(results, alg_cfg)
        got = parameters_to_ndarrays(updated)
        expected = _reference(results, {**FROZEN, "temperature": value})
        assert all(np.array_equal(g, w) for g, w in zip(got, expected, strict=True))
        assert _distance(got, baseline) > 0.0
        assert metrics["kd_temperature"] == value
        entropies.append(_target_entropy(results, value))
    assert entropies[0] < _target_entropy(results, 1.0) < entropies[1] < entropies[2]


@pytest.mark.parametrize("value", [0.5, 4.0])
def test_temperature_extremes_stay_finite(public_loader, value):
    status = Status(code=Code.OK, message="")
    sharp = [w * 200.0 for w in _weights(10)]  # near-one-hot teacher outputs
    results = [
        (object(), FitRes(status, ndarrays_to_parameters(sharp), 5, {})),
        (object(), FitRes(status, ndarrays_to_parameters(_weights(11)), 5, {})),
    ]
    updated, metrics = _seam(results, _repair("temperature", value=value))
    assert np.isfinite(metrics["server_kd_loss"])
    assert all(np.isfinite(a).all() for a in parameters_to_ndarrays(updated))


@pytest.mark.parametrize("settings", [{}, {"value": 0.0}, {"value": 2.0, "rescale": False}])
def test_bad_temperature_settings_are_refused(settings):
    with pytest.raises(ValueError, match="temperature"):
        resolve_kd_repair({"kd_repair": {"temperature": {"enabled": True, **settings}}})


def test_schedule_and_temperature_together_are_refused():
    config = {
        "algorithm": {
            "name": "fedmaq",
            "kd_repair": {
                "schedule": {"enabled": True, "start": 1, "stop": 20},
                "temperature": {"enabled": True, "value": 2.0},
            },
        }
    }
    with pytest.raises(ValueError, match="one repair family per arm"):
        validate_kd_repair_config(config)
