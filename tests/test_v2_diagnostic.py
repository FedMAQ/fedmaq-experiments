"""Read-only V2 measurements must preserve the training state."""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from fedmaq.core.protocol import preregistration_contract
from fedmaq.core.quantization_planner import QuantPlan
from fedmaq.core.strategy import TelemetryFedAvg
from fedmaq.core.strategy_hooks.fedmaq import FedMAQHook
from fedmaq.core.telemetry import TelemetryManager
from fedmaq.core.v2_diagnostic import V2DiagnosticObserver
from scripts.matrix_planner import plan_matrix


def _parameters(weight: list[list[float]]) -> object:
    return ndarrays_to_parameters(
        [np.asarray(weight, dtype=np.float32), np.zeros(2, dtype=np.float32)]
    )


def test_observer_records_paired_models_and_preserves_rng_and_parameters(tmp_path) -> None:
    inputs = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2, shuffle=False)
    observer = V2DiagnosticObserver(
        log_dir=tmp_path,
        dataset_name="synthetic",
        num_classes=2,
        device=torch.device("cpu"),
        validation_loader=loader,
        proxy_loader=loader,
        observed_rounds=[1],
        temperature=1.0,
    )
    observer.model_factory = lambda _dataset, _classes: torch.nn.Linear(2, 2)
    average = _parameters([[2.0, 0.0], [0.0, 2.0]])
    post = _parameters([[0.0, 2.0], [2.0, 0.0]])
    teacher = _parameters([[2.0, 0.0], [0.0, 2.0]])
    before_average = [value.copy() for value in parameters_to_ndarrays(average)]
    before_post = [value.copy() for value in parameters_to_ndarrays(post)]
    result = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=teacher,
        num_examples=12,
        metrics={
            "partition_id": 3,
            "v2_raw_update_norm": 2.0,
            "v2_raw_to_reconstructed_norm": 0.2,
        },
    )
    proxy = SimpleNamespace(cid="runtime-node-9")
    plan = QuantPlan(
        client_q={proxy.cid: 4},
        grad_norms=[0.7],
        client_q_max={proxy.cid: 5.0},
        client_q_hat={proxy.cid: 6.0},
        client_grad_norms={proxy.cid: 0.7},
    )

    random.seed(23)
    np.random.seed(23)
    torch.manual_seed(23)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    elapsed = observer.observe(
        server_round=1,
        parameter_average=average,
        post_kd=post,
        results=[(proxy, result)],
        plan=plan,
        cumulative_bytes=800,
        round_bytes=300,
        post_process=True,
    )

    record = json.loads((tmp_path / "v2_diagnostic.jsonl").read_text(encoding="utf-8"))
    assert elapsed > 0
    assert record["parameter_average"]["accuracy"] == 1.0
    assert record["post_kd_student"]["accuracy"] == 0.0
    assert record["ensemble"]["accuracy"] == 1.0
    assert record["post_minus_average_accuracy"] == -1.0
    assert record["clients"][0]["partition_id"] == 3
    assert record["clients"][0]["q"] == 4
    assert record["clients"][0]["teacher_validation_accuracy"] == 1.0
    assert record["cumulative_bidirectional_bytes"] == 800
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)
    for before, after in zip(before_average, parameters_to_ndarrays(average), strict=True):
        np.testing.assert_array_equal(before, after)
    for before, after in zip(before_post, parameters_to_ndarrays(post), strict=True):
        np.testing.assert_array_equal(before, after)

    assert (
        observer.observe(
            server_round=2,
            parameter_average=average,
            post_kd=post,
            results=[(proxy, result)],
            plan=plan,
            cumulative_bytes=900,
            round_bytes=100,
            post_process=True,
        )
        == 0.0
    )
    assert len((tmp_path / "v2_diagnostic.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_strategy_observer_keeps_aggregate_and_method_bytes(monkeypatch, tmp_path) -> None:
    from flwr.server.strategy import FedAvg

    average = _parameters([[2.0, 0.0], [0.0, 2.0]])
    post = _parameters([[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(FedAvg, "aggregate_fit", lambda *_args: (average, {}))
    monkeypatch.setattr(FedMAQHook, "aggregate_fit", lambda *_args: (post, {}))
    observed: list[tuple[int, int]] = []

    def capture_observation(_observer, **kwargs):
        np.testing.assert_array_equal(
            parameters_to_ndarrays(kwargs["parameter_average"])[0],
            parameters_to_ndarrays(average)[0],
        )
        np.testing.assert_array_equal(
            parameters_to_ndarrays(kwargs["post_kd"])[0],
            parameters_to_ndarrays(post)[0],
        )
        observed.append((kwargs["round_bytes"], kwargs["cumulative_bytes"]))
        return 0.0

    monkeypatch.setattr(V2DiagnosticObserver, "observe", capture_observation)
    loader = DataLoader(TensorDataset(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long)))
    client_result = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=average,
        num_examples=1,
        metrics={"partition_id": 0},
    )

    def aggregate(enabled: bool):
        config = {
            "algorithm": {"name": "fedmaq", "temperature": 1.0, "post_process": True},
            "dataset": {"name": "synthetic", "num_classes": 2},
            "experiment": {"num_clients": 1, "batch_size": 1, "num_public_samples": 1},
            "seed": 42,
            "split": "val",
            "v2_diagnostic": {"enabled": enabled, "rounds": [1]},
        }
        telemetry = TelemetryManager(config)
        telemetry.log_dir = tmp_path
        monkeypatch.setattr(telemetry, "record_fit_round", lambda *_args: (1.5, 123))
        strategy = TelemetryFedAvg(
            telemetry_manager=telemetry,
            config=config,
            diagnostic_validation_loader=loader,
            diagnostic_proxy_loader=loader,
            fraction_fit=1.0,
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )
        return strategy.aggregate_fit(1, [(SimpleNamespace(cid="node"), client_result)], [])

    without, without_metrics = aggregate(False)
    with_observer, with_metrics = aggregate(True)
    assert without is post and with_observer is post
    assert without_metrics == with_metrics == {"round_time": 1.5, "round_bytes": 123}
    assert observed == [(123, 123)]


def test_pilot_matrices_are_matched_nonpromotable_pairs() -> None:
    root = Path(__file__).resolve().parents[1]
    conf_dir = str(root / "conf")
    with initialize_config_dir(config_dir=conf_dir, version_base="1.3"):
        full = OmegaConf.to_container(
            compose(config_name="config", overrides=["algorithm=fedmaq"]).algorithm,
            resolve=True,
        )
        no_kd = OmegaConf.to_container(
            compose(config_name="config", overrides=["algorithm=fedmaq_no_kd"]).algorithm,
            resolve=True,
        )
    assert isinstance(full, dict) and isinstance(no_kd, dict)
    assert {key for key in full if full[key] != no_kd[key]} == {"kd_epochs"}

    for name, alpha, seed in (
        ("v2_diagnostic_severe", "dirichlet_alpha_0.1", 123),
        ("v2_diagnostic_mild", "dirichlet_alpha_1.0", 42),
    ):
        path = root / "conf" / "matrix" / f"{name}.yaml"
        matrix = OmegaConf.load(path)
        plan = plan_matrix(path, matrix)
        assert len(plan.tasks) == 2
        assert plan.ledger == "diagnostic" and plan.split == "val"
        assert preregistration_contract("downstream")["split"] == "test"
        assert {task.algorithm for task in plan.tasks} == {"fedmaq", "fedmaq_no_kd"}
        assert {task.heterogeneity for task in plan.tasks} == {alpha}
        assert {task.seed for task in plan.tasks} == {seed}
        assert [str(item) for item in matrix.runs[0].overrides] == [
            str(item) for item in matrix.runs[1].overrides
        ]
