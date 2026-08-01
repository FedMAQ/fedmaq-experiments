"""Regression tests for SeededPartitionClientManager (the sampling half of the
reproducibility oracle).

Flower's default SimpleClientManager draws with process-global ``random`` over
node IDs whose dict order varies run-to-run, so *which* clients train each round
is not reproducible. These tests pin the deterministic, partition-keyed draw:
same seed -> identical per-round selection regardless of registration order, and
distinct (but reproducible) subsets across rounds.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fedmaq.core import client_manager as cm_module
from fedmaq.core.client_manager import (
    _PARTITION_QUERY_MAX_ATTEMPTS,
    PartitionResolutionError,
    SeededPartitionClientManager,
)


class _FakeProxy:
    """Minimal ClientProxy stand-in: a random node id mapping to a partition id."""

    def __init__(self, node_id: str, partition_id: int) -> None:
        self.cid = node_id
        self._pid = partition_id

    def get_properties(self, ins, timeout=None, group_id=None):  # noqa: ANN001
        return SimpleNamespace(properties={"cid": str(self._pid)})


def _make_manager(num_clients: int, node_order, seed: int = 42):
    """Build a manager with proxies registered in the given node-id order."""
    mgr = SeededPartitionClientManager(seed=seed, num_clients=num_clients)
    # node_order maps registration position -> partition id; the node id (dict key)
    # is an arbitrary string, mimicking Ray's random node ids.
    for pos, pid in enumerate(node_order):
        node_id = f"node-{pos}-{pid}"
        mgr.clients[node_id] = _FakeProxy(node_id, pid)
    return mgr


def _sample_rounds(mgr, num_sample: int, rounds: int):
    out = []
    for r in range(1, rounds + 1):
        mgr.set_round_seed(r)
        selected = mgr.sample(num_sample)
        out.append(sorted(mgr._partition_id(p) for p in selected))
    return out


def test_sampling_reproducible_across_managers() -> None:
    # Two managers, same seed, DIFFERENT registration order -> identical draws.
    order_a = list(range(10))
    order_b = list(reversed(range(10)))
    mgr_a = _make_manager(10, order_a)
    mgr_b = _make_manager(10, order_b)

    rounds_a = _sample_rounds(mgr_a, num_sample=3, rounds=4)
    rounds_b = _sample_rounds(mgr_b, num_sample=3, rounds=4)

    assert rounds_a == rounds_b


def test_sampling_varies_across_rounds() -> None:
    # Per-round seeding must give round-varying subsets (not the same 3 every round).
    mgr = _make_manager(10, list(range(10)))
    rounds = _sample_rounds(mgr, num_sample=3, rounds=4)
    assert len({tuple(r) for r in rounds}) > 1


def test_sample_seed_changes_selection() -> None:
    # A different base seed yields a different selection for the same round.
    mgr1 = _make_manager(10, list(range(10)), seed=1)
    mgr2 = _make_manager(10, list(range(10)), seed=2)
    mgr1.set_round_seed(1)
    mgr2.set_round_seed(1)
    s1 = sorted(mgr1._partition_id(p) for p in mgr1.sample(3))
    s2 = sorted(mgr2._partition_id(p) for p in mgr2.sample(3))
    assert s1 != s2


def test_oversized_request_returns_empty() -> None:
    mgr = _make_manager(3, list(range(3)))
    mgr.set_round_seed(1)
    assert mgr.sample(5) == []


class _FlakyProxy:
    """A proxy whose ``get_properties`` fails ``fail_times`` times, then succeeds."""

    def __init__(self, node_id: str, partition_id: int, fail_times: int) -> None:
        self.cid = node_id
        self._pid = partition_id
        self._remaining_failures = fail_times
        self.calls = 0

    def get_properties(self, ins, timeout=None, group_id=None):  # noqa: ANN001
        self.calls += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise ValueError(
                "Message contains an Error (reason: Error: Message Unavailable - "
                "the destination SuperNode was removed from the federation.)"
            )
        return SimpleNamespace(properties={"cid": str(self._pid)})


@pytest.fixture
def _no_backoff(monkeypatch):
    """Collapse the retry backoff so these tests do not actually sleep ~20s."""
    monkeypatch.setattr(cm_module.time, "sleep", lambda _seconds: None)


def test_partition_id_survives_transient_query_failures(_no_backoff) -> None:
    # A SuperNode that blips but recovers must not kill the run: the round trip is
    # retried and the resolved ID is correct. Regression for the 2026-08-01 sweep,
    # where a single unguarded attempt lost three runs mid-sweep (Decision 77).
    proxy = _FlakyProxy("node-abc", partition_id=7, fail_times=_PARTITION_QUERY_MAX_ATTEMPTS - 1)
    mgr = SeededPartitionClientManager(seed=42, num_clients=10)

    assert mgr._partition_id(proxy) == 7
    assert proxy.calls == _PARTITION_QUERY_MAX_ATTEMPTS


def test_partition_id_raises_rather_than_guessing_when_node_is_gone(_no_backoff) -> None:
    # Permanent loss must abort. The alternatives -- a hash-based fallback ID, or
    # dropping the proxy and sampling a short round -- would both silently change
    # WHICH clients train, which is precisely the property this class provides.
    proxy = _FlakyProxy("node-gone", partition_id=3, fail_times=_PARTITION_QUERY_MAX_ATTEMPTS)
    mgr = SeededPartitionClientManager(seed=42, num_clients=10)

    with pytest.raises(PartitionResolutionError, match="skip_completed"):
        mgr._partition_id(proxy)

    assert proxy.calls == _PARTITION_QUERY_MAX_ATTEMPTS
    # Nothing may be cached from a failed resolution -- a guessed entry would be
    # served silently to every later round.
    assert mgr._partition_cache == {}


def test_partition_id_cached_after_first_query() -> None:
    # get_properties should be hit once per node id, then served from cache.
    mgr = _make_manager(4, list(range(4)))
    proxy = next(iter(mgr.clients.values()))
    calls = {"n": 0}
    orig = proxy.get_properties

    def counting(ins, timeout=None, group_id=None):  # noqa: ANN001
        calls["n"] += 1
        return orig(ins, timeout=timeout, group_id=group_id)

    proxy.get_properties = counting
    mgr._partition_id(proxy)
    mgr._partition_id(proxy)
    assert calls["n"] == 1
