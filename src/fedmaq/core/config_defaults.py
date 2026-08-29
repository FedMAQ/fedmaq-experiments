"""Single source of truth for cross-hook *fallback* defaults.

These constants back the ``cfg.get(key, <literal>)`` fallbacks that several
strategy hooks previously copy-pasted (F-series audit noted the drift risk).
Centralizing them keeps the fallbacks from silently diverging between hooks.

IMPORTANT — these are **defensive fallbacks**, not the canonical experiment
values. In every real/simulated run the resolved Hydra config supplies these
keys, so the fallback branch is dead; it only fires when a hook is constructed
from a minimal/empty config (e.g. unit tests). The authoritative values live in
``conf/`` (``conf/experiment/*.yaml``, ``conf/algorithm/*.yaml``).

One fallback deliberately differs from its canonical ``conf/`` value and is
therefore *not* centralized here — folding it in would enshrine a value that
contradicts the real config:
  * ``weight_decay``: hooks fall back to ``0.0``; canonical is ``1e-4``.

``num_public_samples`` (|D_pub|, canonical ``3000``) has **no fallback**: a
silent ``200`` fallback corrupts the public-proxy pool size (F12). Resolve it
via :func:`require_num_public_samples`, which fails loud on a missing key.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

# Server-side KD delay model (§3.3): simulated samples/sec. This is a bare
# fallback, NOT the grid's value: every conf/algorithm/*.yaml that performs
# server-side KD sets 5000.0 (MobileNetV2GN on the L40S), and
# conf/experiment/femnist.yaml overrides to 10000.0 for SimpleCNN. Nothing in
# the 183-run grid reaches this constant, because the only server-side
# distiller is FedMAQ and every FedMAQ variant declares the key. It is kept at
# 2000.0 because tests/test_timing_golden.py pins the pre-refactor number; do
# not "align" it to 5000.0 without re-baselining those golden tests.
SERVER_COMPUTE_SPEED: float = 2000.0


def resolve_server_compute_speed(config: Mapping[str, Any]) -> float:
    """Resolve server compute speed, preferring experiment-level override.

    The experiment config can carry a per-dataset ``server_compute_speed``
    (e.g. FEMNIST's SimpleCNN is ~2× faster than MobileNetV2GN on the L40S).
    When present it takes priority over the algorithm-level default so that
    a single ``experiment=femnist`` override is sufficient — no manual
    ``algorithm.server_compute_speed=…`` CLI override needed.
    """
    exp_cfg = resolve_experiment_config(config)
    alg_cfg = resolve_algorithm_config(config)
    return float(
        exp_cfg.get(
            "server_compute_speed",
            alg_cfg.get("server_compute_speed", SERVER_COMPUTE_SPEED),
        )
    )


# Mini-batch size B. Matches conf/experiment/default.yaml.
BATCH_SIZE: int = 64

# Defensive dataset fallbacks used when cfg["dataset"] is absent. num_classes is
# dataset-dependent at runtime; these only apply to a config-less construction.
DATASET_NAME: str = "mnist"
NUM_CLASSES: int = 10

# Run-level fallbacks are only used by lightweight hook/unit-test construction.
# The composed experiment config remains authoritative for real runs.
SEED: int = 42


def resolve_dataset_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the dataset section from either a nested or flat config."""
    dataset = config.get("dataset")
    if isinstance(dataset, Mapping):
        flat: dict[str, Any] = {}
        if "dataset_name" in config:
            flat["name"] = config["dataset_name"]
        if "num_classes" in config:
            flat["num_classes"] = config["num_classes"]
        flat.update(dataset)
        return flat
    flat = dict(config)
    if dataset is not None:
        flat.setdefault("name", dataset)
    return flat


def resolve_experiment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the experiment section, treating a section-less config as flat."""
    experiment = config.get("experiment")
    if isinstance(experiment, Mapping):
        flat = {
            key: config[key]
            for key in (
                "batch_size",
                "num_clients",
                "num_public_samples",
                "total_rounds",
                "server_compute_speed",
            )
            if key in config
        }
        flat.update(experiment)
        return flat
    return dict(config)


def resolve_algorithm_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return algorithm configuration without placing it in :class:`RunContext`.

    A nested root config uses ``config["algorithm"]``. A flat algorithm config is
    accepted as-is; a flat run config can provide ``algorithm_name`` without
    making unrelated run fields algorithm-specific.
    """
    algorithm = config.get("algorithm")
    if isinstance(algorithm, Mapping):
        return dict(algorithm)
    if algorithm is not None:
        return {"name": algorithm}
    if "algorithm_name" in config:
        algorithm_cfg = {
            key: value
            for key, value in config.items()
            if key not in {
                "algorithm_name",
                "dataset",
                "dataset_name",
                "num_classes",
                "batch_size",
                "device",
                "seed",
                "num_clients",
                "num_public_samples",
                "total_rounds",
            }
        }
        algorithm_cfg["name"] = config["algorithm_name"]
        return algorithm_cfg

    run_keys = {
        "dataset",
        "dataset_name",
        "num_classes",
        "batch_size",
        "device",
        "seed",
        "num_clients",
        "num_public_samples",
        "total_rounds",
    }
    return {} if run_keys.intersection(config) else dict(config)


def require_num_public_samples(config) -> int:
    """Resolve |D_pub| from a resolved config, failing loud if absent.

    Unlike the other cross-hook values there is no safe fallback: the canonical
    size is 3000 (``conf/experiment/default.yaml``) and a silent 200 would
    quietly corrupt the public-proxy pool (F12). Every real/simulated run
    supplies ``experiment.num_public_samples``, so a missing key signals a
    misconfigured run that must abort rather than proceed on a wrong size.
    """
    # Mirror the experiment-else-whole-config resolution used elsewhere in the
    # strategy (strategy.py) so flat and experiment-wrapped configs both work.
    experiment = resolve_experiment_config(config) if isinstance(config, Mapping) else {}
    if "num_public_samples" not in experiment:
        raise KeyError(
            "experiment.num_public_samples is required (canonical 3000 per "
            "conf/experiment/default.yaml); there is no fallback. Add it to the "
            "resolved config before constructing KD hooks."
        )
    return int(experiment["num_public_samples"])


@dataclass(frozen=True)
class RunContext:
    """The run-level values shared by strategy hooks and quantization planning."""

    dataset_name: str
    num_classes: int
    batch_size: int
    device: torch.device
    seed: int
    num_public_samples: int | None
    server_compute_speed: float
    algorithm_name: str


def resolve_run_context(config: Mapping[str, Any]) -> RunContext:
    """Resolve shared run values from nested, flat, or partial configuration.

    Nested sections take precedence over flat keys. Missing values use explicit
    defensive fallbacks, except ``num_public_samples`` which stays ``None`` so
    callers that require the canonical proxy-pool size can fail loudly.
    """
    from fedmaq.core.models import DEVICE

    dataset_cfg = resolve_dataset_config(config)
    experiment_cfg = resolve_experiment_config(config)

    def value_or(mapping: Mapping[str, Any], key: str, fallback: Any) -> Any:
        value = mapping.get(key)
        return fallback if value is None else value

    public_samples = experiment_cfg.get("num_public_samples")
    algorithm_cfg = resolve_algorithm_config(config)
    return RunContext(
        dataset_name=str(
            value_or(
                dataset_cfg,
                "name",
                config.get("dataset_name", DATASET_NAME),
            )
        ),
        num_classes=int(
            value_or(dataset_cfg, "num_classes", config.get("num_classes", NUM_CLASSES))
        ),
        batch_size=int(value_or(experiment_cfg, "batch_size", BATCH_SIZE)),
        device=torch.device(value_or(config, "device", DEVICE)),
        seed=int(value_or(config, "seed", SEED)),
        num_public_samples=int(public_samples) if public_samples is not None else None,
        server_compute_speed=resolve_server_compute_speed(config),
        algorithm_name=str(value_or(algorithm_cfg, "name", "")),
    )
