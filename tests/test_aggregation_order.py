"""Aggregation order must key on partition ID, not ``ClientProxy.cid``.

``cid`` is ``str(node_id)`` and Flower mints node IDs from ``urandom`` on every
run, so a cid-keyed sort is canonical *within* a run but an arbitrary permutation
of partitions *across* runs. Since ``aggregate()`` reduces float32 arrays
sequentially, that permutation silently changes the aggregated global model —
the model stops being reproducible even at ``client_gpus=1.0``.

These tests pin the invariant that caught it: the same partitions must reduce in
the same order no matter what node IDs the runtime happened to assign.
"""

import random

import flwr as fl
import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters

from fedmaq.core.strategy import TelemetryFedAvg
from fedmaq.core.telemetry import TelemetryManager

NUM_CLIENTS = 4


class _NodeIdProxy(fl.server.client_proxy.ClientProxy):
    """Proxy whose ``cid`` is a random int64 node ID, as in real simulation."""

    def __init__(self, node_id: int) -> None:
        super().__init__(str(node_id))

    def get_properties(self, ins, timeout=None, group_id=None):
        raise AssertionError(
            "partition_id is present in fit metrics; the sort key must not need "
            "a get_properties round-trip on the common path."
        )

    def get_parameters(self, ins, timeout=None, group_id=None):
        return None

    def fit(self, ins, timeout=None, group_id=None):
        return None

    def evaluate(self, ins, timeout=None, group_id=None):
        return None

    def reconnect(self, ins, timeout=None, group_id=None):
        return None


def _make_strategy() -> TelemetryFedAvg:
    cfg_dict = {
        "num_clients": NUM_CLIENTS,
        "batch_size": 2,
        "seed": 42,
        "experiment": {
            "num_clients": NUM_CLIENTS,
            "client_fraction": 1.0,
            "total_rounds": 1,
        },
        "algorithm": {"name": "fedavg"},
    }
    return TelemetryFedAvg(
        telemetry_manager=TelemetryManager(cfg_dict),
        config=cfg_dict,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )


def _results_for_run(rng: random.Random):
    """One run's worth of fit results: fixed partitions, fresh random node IDs.

    Returned in node-ID order, which is what Flower's completion-ordered ``set``
    would hand to a cid-keyed sort.
    """
    results = []
    for pid in range(NUM_CLIENTS):
        node_id = rng.getrandbits(63)
        fit_res = FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=ndarrays_to_parameters([np.array([float(pid)], dtype=np.float32)]),
            num_examples=10,
            metrics={"partition_id": pid},
        )
        results.append((_NodeIdProxy(node_id), fit_res))
    return results


def test_sort_key_recovers_partition_order_regardless_of_node_ids():
    """Two runs assign different node IDs; both must reduce in partition order."""
    strategy = _make_strategy()

    for seed in (1, 2, 3):
        results = _results_for_run(random.Random(seed))
        random.Random(seed + 100).shuffle(results)  # completion order is arbitrary

        ordered = sorted(results, key=lambda r: strategy._partition_sort_key(*r))
        pids = [int(fit_res.metrics["partition_id"]) for _, fit_res in ordered]

        assert pids == list(range(NUM_CLIENTS)), f"expected canonical partition order, got {pids}"


def test_cid_sort_would_not_be_stable_across_runs():
    """Guard the premise: sorting on cid genuinely permutes partitions per run.

    If this ever stops holding, Flower changed how it mints node IDs and the
    comment in ``aggregate_fit`` needs revisiting.
    """
    orders = set()
    for seed in range(20):
        results = _results_for_run(random.Random(seed))
        by_cid = sorted(results, key=lambda r: r[0].cid)
        orders.add(tuple(int(fit_res.metrics["partition_id"]) for _, fit_res in by_cid))

    assert len(orders) > 1, (
        "cid sort produced one stable partition order across runs; the "
        "random-node-id premise no longer holds"
    )


def test_missing_partition_id_raises_rather_than_using_an_unstable_key():
    """A missing metric must fail loudly, not degrade to ``hash(cid)``.

    Python randomises string hashing per process, so the hash fallback would
    silently reintroduce the run-to-run permutation this sort prevents.
    """
    strategy = _make_strategy()
    proxy = _NodeIdProxy(random.Random(0).getrandbits(63))
    fit_res = FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters([np.array([0.0], dtype=np.float32)]),
        num_examples=10,
        metrics={},  # no partition_id
    )

    with pytest.raises(ValueError, match="partition_id"):
        strategy._partition_sort_key(proxy, fit_res)
