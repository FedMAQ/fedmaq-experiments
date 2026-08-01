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

WHEN A PARTITION ID CANNOT BE RESOLVED, THE RUN ABORTS
------------------------------------------------------
``get_properties`` is a live round trip to a SuperNode, and on a shared host it can
fail two ways: transiently (a message expires under contention) or permanently (the
SuperNode left the federation). Resolution therefore retries generously — see
``_PARTITION_QUERY_MAX_ATTEMPTS`` — so that only *permanent* loss exhausts the budget.

If it does exhaust, this module raises rather than substituting a guessed partition
ID. That is deliberate and is the opposite of what
``fedmaq.core.strategy_hooks._partition.resolve_partition_id`` does, which falls back
to ``hash(cid) % num_clients``. A guessed ID here would not merely mislabel one
client: it would change *which* partitions are drawn, silently, in a way that varies
run to run — destroying the exact property this class exists to provide. Dropping the
proxy instead is no better, because a 99-client round is not the ``K = 100, C = 0.1``
protocol of Table 4.1 and nothing downstream would record the deviation.

Aborting costs one run, recoverable with ``run_matrix.py --skip_completed``. A
silently non-reproducible run costs the reproducibility claim and is not detectable
after the fact.
"""

from __future__ import annotations

import logging
import random
import time

from flwr.common.typing import GetPropertiesIns
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

logger = logging.getLogger(__name__)

# Resilience budget for the partition-ID round trip. Sized so a transient failure is
# absorbed and only a SuperNode that is genuinely gone exhausts it: five attempts with
# linear backoff span ~20s of waiting on top of the per-attempt timeouts. Raised from
# a single unguarded attempt after a 26-run sweep lost three runs to mid-run
# federation membership changes on 2026-08-01 (Decision 77).
_PARTITION_QUERY_TIMEOUT = 30.0
_PARTITION_QUERY_MAX_ATTEMPTS = 5
_PARTITION_QUERY_BACKOFF_SECONDS = 2.0


class PartitionResolutionError(RuntimeError):
    """A client proxy's partition ID could not be resolved; the run cannot continue.

    Fatal by design — see this module's docstring for why no fallback mapping is
    substituted and no proxy is silently dropped.
    """


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
        """Resolve (and cache) the partition ID a client proxy owns.

        Retries a failed round trip up to ``_PARTITION_QUERY_MAX_ATTEMPTS`` times with
        linear backoff, then raises :class:`PartitionResolutionError`. Never guesses.
        """
        node_id = str(proxy.cid)
        cached = self._partition_cache.get(node_id)
        if cached is not None:
            return cached

        last_exc: Exception | None = None
        for attempt in range(1, _PARTITION_QUERY_MAX_ATTEMPTS + 1):
            try:
                try:
                    res = proxy.get_properties(
                        GetPropertiesIns(config={}),
                        timeout=_PARTITION_QUERY_TIMEOUT,
                        group_id=0,
                    )
                except TypeError:  # older Flower signature without group_id
                    res = proxy.get_properties(
                        GetPropertiesIns(config={}), timeout=_PARTITION_QUERY_TIMEOUT
                    )
                pid = int(res.properties["cid"])
            except Exception as exc:  # noqa: BLE001 — retried, then re-raised as fatal
                last_exc = exc
                if attempt < _PARTITION_QUERY_MAX_ATTEMPTS:
                    delay = _PARTITION_QUERY_BACKOFF_SECONDS * attempt
                    logger.warning(
                        "Attempt %d/%d failed to resolve the partition ID for node %s: "
                        "%s. Retrying in %.1fs.",
                        attempt,
                        _PARTITION_QUERY_MAX_ATTEMPTS,
                        node_id,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                continue
            self._partition_cache[node_id] = pid
            return pid

        raise PartitionResolutionError(
            f"Could not resolve the partition ID for node {node_id} after "
            f"{_PARTITION_QUERY_MAX_ATTEMPTS} attempts; last error: {last_exc}. "
            "This usually means the SuperNode left the federation mid-run. Aborting "
            "rather than guessing a partition ID or sampling a short round, either of "
            "which would silently change which clients train and break run-to-run "
            "reproducibility. Re-dispatch with `run_matrix.py --skip_completed` to "
            "resume; completed runs are not repeated."
        ) from last_exc

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
