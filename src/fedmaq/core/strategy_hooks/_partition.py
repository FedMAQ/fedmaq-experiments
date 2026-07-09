"""Shared partition-resolution helpers for client-adaptive strategy hooks.

FedMAQ and DAdaQuant both need to map a Flower ``ClientProxy`` to its data
partition and look up that partition's dataset size; these helpers keep that
logic in one place.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from flwr.common.typing import GetPropertiesIns
from flwr.server.client_proxy import ClientProxy

if TYPE_CHECKING:
    from fedmaq.core.strategy import TelemetryFedAvg

logger = logging.getLogger(__name__)


def resolve_partition_id(
    client: ClientProxy,
    strategy: TelemetryFedAvg,
) -> int:
    """Resolve a client's partition ID, caching the result on the strategy.

    First checks the cache; then treats a digit ``cid`` as the partition ID
    (the Flower simulation default); then queries ``get_properties`` with a
    ``group_id`` fallback for older Flower versions; finally falls back to a
    hash-based mapping and logs a WARNING so the failure is visible.
    """
    cid_str = str(client.cid)
    if cid_str in strategy.proxy_cid_to_partition_id:
        return strategy.proxy_cid_to_partition_id[cid_str]

    # Flower simulation shortcut: client.cid is typically the stringified partition ID
    if cid_str.isdigit():
        pid = int(cid_str)
        if pid < strategy.num_clients:
            strategy.proxy_cid_to_partition_id[cid_str] = pid
            return pid

    try:
        try:
            res = client.get_properties(
                GetPropertiesIns(config={}), timeout=5.0, group_id=0
            )
        except TypeError:
            res = client.get_properties(GetPropertiesIns(config={}), timeout=5.0)
        pid = int(res.properties["cid"])
        strategy.proxy_cid_to_partition_id[cid_str] = pid
        logger.info(f"Queried partition ID {pid} for Client Proxy {cid_str}")
        return pid
    except Exception as exc:
        pid = hash(client.cid) % strategy.num_clients
        strategy.proxy_cid_to_partition_id[cid_str] = pid
        logger.warning(
            f"Could not resolve partition ID for client {cid_str} ({exc}). "
            f"Falling back to hash-based mapping -> partition {pid}. "
            "Verify that GenericClient.get_properties exposes 'cid'."
        )
        return pid


def partition_dataset_size(
    client_indices_dict: dict[str, list[int]] | dict[int, list[int]] | None,
    pid: int,
    default: int = 1,
) -> int:
    """Return the number of samples in partition ``pid``.

    Handles both string- and int-keyed partition dictionaries and falls back to
    ``default`` (with a WARNING) when the partition is absent or no dictionary is
    supplied.
    """
    if client_indices_dict is None:
        return default
    key_str, key_int = str(pid), int(pid)
    if key_str in client_indices_dict:
        return len(client_indices_dict[key_str])
    if key_int in client_indices_dict:
        return len(client_indices_dict[key_int])
    logger.warning(
        f"Partition ID {pid} not found in client_indices_dict. "
        f"Defaulting size to {default}."
    )
    return default
