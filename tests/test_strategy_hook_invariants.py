from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from torch.utils.data import DataLoader, TensorDataset

from fedmaq.core.strategy_hooks.dadaquant import DAdaQuantHook
from fedmaq.core.strategy_hooks.fedavg_kd import FedAvgKDHook
from fedmaq.core.strategy_hooks.fedkd import FedKDHook


def test_dadaquant_keeps_quantization_bounds_resolved_at_construction() -> None:
    config = {"algorithm": {"q_min": 3, "q_max": 7}}
    hook = DAdaQuantHook(config)
    config["algorithm"]["q_min"] = 1
    config["algorithm"]["q_max"] = 1

    client_instructions = [
        (SimpleNamespace(cid="client-0"), FitIns(ndarrays_to_parameters([]), {}))
    ]
    strategy = SimpleNamespace(client_indices_dict=None)

    updated = hook.configure_fit(strategy, 1, ndarrays_to_parameters([]), None, client_instructions)

    assert updated[0][1].config["q"] == 3


def test_round_phase_metrics_are_empty_before_their_first_round() -> None:
    fedavg_kd = FedAvgKDHook({})
    fedkd = FedKDHook({})

    assert fedavg_kd.get_eval_metrics(None, 0) == {}
    assert "algorithm/fedkd/mean_rank_retained" not in fedkd.get_eval_metrics(None, 0)


def test_fedkd_pre_evaluate_returns_the_aggregate_not_the_fit_broadcast() -> None:
    """The model evaluated after a round must be the model produced by aggregation."""
    hook = FedKDHook(
        {
            "seed": 7,
            "algorithm": {"tmin": 0.1, "tmax": 0.9},
            "experiment": {"total_rounds": 2},
            "dataset": {"name": "mnist", "num_classes": 10},
        }
    )
    initial = ndarrays_to_parameters([np.zeros((4, 2), dtype=np.float32)])
    broadcast = hook.pre_configure_fit(None, 1, initial)
    aggregate = ndarrays_to_parameters([np.full((4, 2), 3.0, dtype=np.float32)])

    np.testing.assert_allclose(
        parameters_to_ndarrays(hook.pre_evaluate(None, 1, aggregate))[0],
        parameters_to_ndarrays(aggregate)[0],
    )
    assert hook.download_size_bytes(None, [np.full((4, 2), 3.0, dtype=np.float32)]).payloads
    assert broadcast is not aggregate


def test_fedkd_two_round_lifecycle_evaluates_checkpoints_and_broadcasts_aggregate(
    tmp_path,
) -> None:
    """Round evaluation/checkpoint state must be the prior round's aggregate."""
    from fedmaq.core.checkpoint import write_final_global_model

    hook = FedKDHook(
        {
            "algorithm": {"tmin": 0.1, "tmax": 0.9},
            "experiment": {"total_rounds": 2},
            "dataset": {"name": "mnist", "num_classes": 10},
        }
    )
    current = ndarrays_to_parameters([np.zeros((1, 2), dtype=np.float32)])

    for server_round in (1, 2):
        broadcast = hook.pre_configure_fit(None, server_round, current)
        aggregate = ndarrays_to_parameters([np.full((1, 2), server_round, dtype=np.float32)])
        evaluated = hook.pre_evaluate(None, server_round, aggregate)
        np.testing.assert_allclose(
            parameters_to_ndarrays(evaluated)[0], parameters_to_ndarrays(aggregate)[0]
        )

        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.from_numpy(parameters_to_ndarrays(aggregate)[0]))
        checkpoint = write_final_global_model(model, tmp_path, server_round, 2)
        if server_round == 1:
            assert checkpoint is None
        else:
            assert checkpoint is not None
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            np.testing.assert_allclose(
                saved["weight"].numpy(), parameters_to_ndarrays(evaluated)[0]
            )

        current = aggregate
        assert broadcast is not aggregate


def test_fedkd_telemetry_lifecycle_passes_aggregate_to_eval_and_next_broadcast(
    tmp_path, monkeypatch
) -> None:
    """The public strategy lifecycle keeps aggregate, evaluation, and checkpoint aligned."""
    from flwr.common import FitIns
    from flwr.server.strategy import FedAvg

    from fedmaq.core.checkpoint import write_final_global_model
    from fedmaq.core.telemetry import TelemetryManager

    config = {
        "algorithm": {"name": "fedkd", "tmin": 0.1, "tmax": 0.9},
        "experiment": {"total_rounds": 2},
        "dataset": {"name": "mnist", "num_classes": 10},
    }
    telemetry = TelemetryManager(config)
    telemetry.log_dir = tmp_path
    telemetry.log = lambda **_kwargs: None
    aggregates = {
        1: ndarrays_to_parameters([np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)]),
        2: ndarrays_to_parameters([np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32)]),
    }
    evaluated: list[list[np.ndarray]] = []

    def fake_configure_fit(_strategy, _server_round, parameters, _client_manager):
        return [(SimpleNamespace(), FitIns(parameters, {}))]

    def fake_aggregate_fit(_strategy, server_round, _results, _failures):
        return aggregates[server_round], {}

    monkeypatch.setattr(FedAvg, "configure_fit", fake_configure_fit)
    monkeypatch.setattr(FedAvg, "aggregate_fit", fake_aggregate_fit)
    monkeypatch.setattr(telemetry, "record_fit_round", lambda *_args: (0.0, 0))

    def evaluate_fn(server_round, parameters, _config):
        evaluated.append([array.copy() for array in parameters])
        model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.from_numpy(parameters[0]))
        write_final_global_model(model, tmp_path, server_round, 2)
        return 0.0, {"accuracy": 0.0}

    from fedmaq.core.strategy import TelemetryFedAvg

    strategy = TelemetryFedAvg(
        telemetry_manager=telemetry,
        config=config,
        fraction_fit=1.0,
        min_fit_clients=1,
        min_available_clients=1,
        evaluate_fn=evaluate_fn,
        initial_parameters=ndarrays_to_parameters([np.zeros((2, 2), dtype=np.float32)]),
    )

    current = strategy.initial_parameters
    broadcasts = []
    for server_round in (1, 2):
        instructions = strategy.configure_fit(server_round, current, None)
        broadcasts.append(parameters_to_ndarrays(instructions[0][1].parameters)[0])
        current, _ = strategy.aggregate_fit(server_round, [], [])
        strategy.evaluate(server_round, current)

    np.testing.assert_allclose(evaluated[0][0], parameters_to_ndarrays(aggregates[1])[0])
    np.testing.assert_allclose(evaluated[1][0], parameters_to_ndarrays(aggregates[2])[0])
    assert broadcasts[1].shape == parameters_to_ndarrays(aggregates[1])[0].shape
    assert not np.shares_memory(broadcasts[1], evaluated[1][0])
    saved = torch.load(tmp_path / "final_global_model.pt", map_location="cpu", weights_only=True)
    np.testing.assert_allclose(saved["weight"].numpy(), evaluated[1][0])


def test_fedkd_download_report_is_the_report_for_the_broadcast_state() -> None:
    hook = FedKDHook(
        {
            "seed": 7,
            "algorithm": {"tmin": 0.1, "tmax": 0.9},
            "experiment": {"total_rounds": 2},
            "dataset": {"name": "mnist", "num_classes": 10},
        }
    )
    hook.pre_configure_fit(None, 1, ndarrays_to_parameters([np.zeros((4, 2), dtype=np.float32)]))
    report = hook.download_size_bytes(None, [np.full((4, 2), 3.0, dtype=np.float32)])

    assert (
        report.payload_bytes
        == hook.get_eval_metrics(None, 1)["algorithm/fedkd/download_payload_bytes"]
    )


def test_round_seed_is_stable_and_separates_training_and_compression_streams() -> None:
    from fedmaq.core.randomness import derive_seed

    first = derive_seed("training", 42, 3, 7)
    replay = derive_seed("training", 42, 3, 7)

    assert first == replay
    assert first != derive_seed("training", 42, 3, 8)
    assert first != derive_seed("compression", 42, 3, 7)


def test_compression_stream_varies_by_round_and_replays_by_identity() -> None:
    from fedmaq.baselines.quantization import FedPAQCompressionHook
    from fedmaq.core.randomness import derive_numpy_rng

    delta = np.linspace(-1.0, 1.0, 2048, dtype=np.float32)
    round_one, _ = FedPAQCompressionHook(
        q=4, rng=derive_numpy_rng("compression", 42, 3, 1)
    ).compress([delta])
    round_two, _ = FedPAQCompressionHook(
        q=4, rng=derive_numpy_rng("compression", 42, 3, 2)
    ).compress([delta])
    replay, _ = FedPAQCompressionHook(
        q=4, rng=derive_numpy_rng("compression", 42, 3, 1)
    ).compress([delta])

    assert not np.array_equal(round_one[0], round_two[0])
    np.testing.assert_array_equal(round_one[0], replay[0])


def test_generic_client_rebuilds_batch_order_per_round_and_replays_it() -> None:
    from fedmaq.core.client import CompressionHook, GenericClient, LossHook
    from fedmaq.core.models import TinyCNN
    from fedmaq.core.randomness import derive_seed

    dataset = TensorDataset(torch.arange(12).float().reshape(12, 1, 1, 1), torch.zeros(12))

    def make_client(observed: list[list[int]]) -> GenericClient:
        def loader_for_round(server_round: int) -> DataLoader:
            generator = torch.Generator().manual_seed(derive_seed("training", 42, 3, server_round))
            return DataLoader(dataset, batch_size=3, shuffle=True, generator=generator)

        client = GenericClient(
            cid="3",
            trainloader=loader_for_round(1),
            testloader=loader_for_round(1),
            model=TinyCNN(in_channels=1, num_classes=10),
            loss_hook=LossHook(),
            compressor_hook=CompressionHook(),
            config={"algorithm": {"name": "fedavg"}, "dataset": {"name": "mnist"}},
            trainloader_factory=loader_for_round,
        )

        def observe(_client, _parameters, config):
            observed.append([int(images.flatten()[0]) for images, _labels in _client.trainloader])
            return [], 0, {}

        client.fit_strategy = SimpleNamespace(fit=observe)
        return client

    first_observation: list[list[int]] = []
    first_client = make_client(first_observation)
    first_client.fit([], {"server_round": 1})
    first_client.fit([], {"server_round": 2})

    replay_observation: list[list[int]] = []
    replay_client = make_client(replay_observation)
    replay_client.fit([], {"server_round": 1})
    replay_client.fit([], {"server_round": 2})

    assert first_observation[0] != first_observation[1]
    assert replay_observation == first_observation
