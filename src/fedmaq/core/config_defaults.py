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

# Server-side KD delay model (§3.3): simulated samples/sec. Matches the value in
# every conf/algorithm/*.yaml that defines it, so this fallback is safe.
SERVER_COMPUTE_SPEED: float = 2000.0


def resolve_server_compute_speed(config: dict) -> float:
    """Resolve server compute speed, preferring experiment-level override.

    The experiment config can carry a per-dataset ``server_compute_speed``
    (e.g. FEMNIST's SimpleCNN is ~2× faster than MobileNetV2GN on the L40S).
    When present it takes priority over the algorithm-level default so that
    a single ``experiment=femnist`` override is sufficient — no manual
    ``algorithm.server_compute_speed=…`` CLI override needed.
    """
    exp_cfg = config.get("experiment", config) if isinstance(config, dict) else {}
    alg_cfg = config.get("algorithm", {}) if isinstance(config, dict) else {}
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
    experiment = config.get("experiment", config) if hasattr(config, "get") else {}
    if "num_public_samples" not in experiment:
        raise KeyError(
            "experiment.num_public_samples is required (canonical 3000 per "
            "conf/experiment/default.yaml); there is no fallback. Add it to the "
            "resolved config before constructing KD hooks."
        )
    return int(experiment["num_public_samples"])
