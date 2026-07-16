"""Single source of truth for cross-hook *fallback* defaults.

These constants back the ``cfg.get(key, <literal>)`` fallbacks that several
strategy hooks previously copy-pasted (F-series audit noted the drift risk).
Centralizing them keeps the fallbacks from silently diverging between hooks.

IMPORTANT — these are **defensive fallbacks**, not the canonical experiment
values. In every real/simulated run the resolved Hydra config supplies these
keys, so the fallback branch is dead; it only fires when a hook is constructed
from a minimal/empty config (e.g. unit tests). The authoritative values live in
``conf/`` (``conf/experiment/*.yaml``, ``conf/algorithm/*.yaml``).

Two fallbacks deliberately differ from their canonical ``conf/`` values and are
therefore *not* centralized here — folding them in would enshrine a value that
contradicts the real config:
  * ``num_public_samples``: hooks fall back to ``200``; canonical is ``3000``
    (``conf/experiment/default.yaml``). Left inline to avoid presenting ``200``
    as "the default".
  * ``weight_decay``: hooks fall back to ``0.0``; canonical is ``1e-4``.
"""

# Server-side KD delay model (§3.3): simulated samples/sec. Matches the value in
# every conf/algorithm/*.yaml that defines it, so this fallback is safe.
SERVER_COMPUTE_SPEED: float = 2000.0

# Mini-batch size B. Matches conf/experiment/default.yaml.
BATCH_SIZE: int = 64

# Defensive dataset fallbacks used when cfg["dataset"] is absent. num_classes is
# dataset-dependent at runtime; these only apply to a config-less construction.
DATASET_NAME: str = "mnist"
NUM_CLASSES: int = 10
