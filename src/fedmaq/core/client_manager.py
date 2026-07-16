"""Deterministic, partition-keyed client manager for reproducible sampling.

Flower's default :class:`SimpleClientManager.sample` draws with the process-global
``random`` module over ``list(self.clients)`` — a dict keyed by Ray-assigned node
IDs whose values and insertion order both vary run-to-run (random node IDs,
timing-dependent registration). Even with a fixed global seed the *set* of clients
selected each round is therefore not reproducible; only per-worker *training* is.

:class:`SeededPartitionClientManager` fixes this by sampling over **partition IDs**
(the only cross-run-stable client identity) with a dedicated per-round RNG:

* Each ``ClientProxy`` is resolved to its partition ID via ``get_properties``
  (``GenericClient.get_properties`` returns ``{"cid": str(partition_id)}``), cached
  by node ID since node IDs are stable *within* a run.
* Before each round, :class:`~fedmaq.core.strategy.TelemetryFedAvg.configure_fit`
  calls :meth:`set_round_seed`, so the draw is seeded by ``(base_seed, round)`` and
  is robust to how many times ``sample`` happens to be called.
* ``sample`` waits for the **full** population before drawing, so a partial,
  timing-dependent set can never be sorted into a false-deterministic order.

Together these make *which* clients train each round bit-identical across runs given
a fixed seed — the sampling half of the end-to-end reproducibility oracle.
"""

from __future__ import annotations

import logging
import random

from flwr.common.typing import GetPropertiesIns
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

logger = logging.getLogger(__name__)


class SeededPartitionClientManager(SimpleClientManager):
    """A :class:`SimpleClientManager` that samples reproducibly by partition ID."""

    def __init__(self, seed: int, num_clients: int) -> None:
        super().__init__()
        self._base_seed = int(seed)
        self._num_clients = int(num_clients)
        # node-id (proxy.cid) -> partition id; node ids are stable within a run.
        self._partition_cache: dict[str, int] = {}
        # Seed of the current round's draw; set by the strategy before sampling.
        self._round_seed: int = 0

    def set_round_seed(self, server_round: int) -> None:
        """Set the seed for the next :meth:`sample` call (call once per round)."""
        self._round_seed = int(server_round)

    def _partition_id(self, proxy: ClientProxy) -> int:
        """Resolve (and cache) the partition ID a client proxy owns."""
        node_id = str(proxy.cid)
        cached = self._partition_cache.get(node_id)
        if cached is not None:
            return cached
        try:
            res = proxy.get_properties(GetPropertiesIns(config={}), timeout=30.0, group_id=0)
        except TypeError:  # older Flower signature without group_id
            res = proxy.get_properties(GetPropertiesIns(config={}), timeout=30.0)
        pid = int(res.properties["cid"])
        self._partition_cache[node_id] = pid
        return pid

    def sample(
        self,
        num_clients: int,
        min_num_clients: int | None = None,
        criterion: Criterion | None = None,
    ) -> list[ClientProxy]:
        """Sample ``num_clients`` proxies deterministically by partition ID.

        Overrides the global-``random`` draw of :class:`SimpleClientManager` with a
        partition-keyed draw from a per-round-seeded :class:`random.Random`.
        """
        # Wait for the FULL population, not just ``min_num_clients``: sampling a
        # partial, still-registering set would reintroduce timing nondeterminism.
        self.wait_for(self._num_clients)

        proxies = list(self.clients.values())
        if criterion is not None:
            proxies = [p for p in proxies if criterion.select(p)]

        # Deterministic candidate order: sort by partition ID (independent of the
        # node-id dict order Ray happens to produce this run).
        pid_to_proxy: dict[int, ClientProxy] = {self._partition_id(p): p for p in proxies}
        available_pids = sorted(pid_to_proxy)

        if num_clients > len(available_pids):
            logger.info(
                "Sampling failed: available clients (%d) < requested (%d).",
                len(available_pids),
                num_clients,
            )
            return []

        # Dedicated RNG seeded per round -> reproducible AND round-varying, with no
        # dependence on process-global random state or the number of sample() calls.
        rng = random.Random(self._base_seed * 1_000_003 + self._round_seed)
        chosen = rng.sample(available_pids, num_clients)
        return [pid_to_proxy[pid] for pid in chosen]
