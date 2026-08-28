from __future__ import annotations

from types import SimpleNamespace

from flwr.common import FitIns, ndarrays_to_parameters

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
