"""Abstract base class for per-algorithm strategy hooks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from flwr.common import Parameters, Scalar
from flwr.common.typing import FitIns, FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

if TYPE_CHECKING:
    from fedmaq.core.strategy import TelemetryFedAvg


class StrategyHook(ABC):
    """Per-algorithm extension points for :class:`~fedmaq.core.strategy.TelemetryFedAvg`.

    Each concrete hook handles exactly one algorithm's logic, keeping the strategy
    class free of ``if alg_name == ...`` dispatch chains.

    Hook call sequence per round
    -----------------------------
    1. ``pre_configure_fit``  — optionally compress parameters before client sampling
    2. ``configure_fit``      — post-process client instructions (e.g. assign q)
    3. ``pre_aggregate_fit``  — optionally bypass FedAvg aggregation entirely
    4. ``aggregate_fit``      — post-process aggregated parameters (e.g. server KD)
    5. ``pre_evaluate``       — optionally decompress parameters before evaluation
    6. ``get_eval_metrics``   — supply algorithm-specific metrics for the round log

    All methods receive the live ``strategy`` object, giving full read/write access
    to shared state (e.g. ``strategy.proxy_cid_to_partition_id``).
    New baselines (FedDistill, CFD) add a single hook file and register in
    ``__init__.py`` — the strategy itself stays untouched.
    """

    def pre_configure_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        parameters: Parameters,
    ) -> Parameters:
        """Optionally transform server parameters before FedAvg client sampling.

        Used by FedKD to apply SVD compression to the download path.
        Default: identity (no modification).
        """
        return parameters

    @abstractmethod
    def configure_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        client_instructions: list[tuple[ClientProxy, FitIns]],
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Post-process client fit instructions after FedAvg client sampling."""
        ...

    def pre_aggregate_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]] | None:
        """Optionally replace FedAvg weighted aggregation entirely.

        Return a ``(parameters, metrics)`` pair to skip FedAvg aggregation,
        or ``None`` to let FedAvg aggregate normally.
        Used by FedMD (logit aggregation) and future FedDistill / CFD.
        """
        return None

    @abstractmethod
    def aggregate_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
        aggregated_parameters: Parameters | None,
        metrics: dict[str, Scalar],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Post-process aggregated parameters (e.g. server-side KD, loss tracking)."""
        ...

    def pre_evaluate(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        parameters: Parameters,
    ) -> Parameters:
        """Optionally transform parameters before server-side evaluation.

        Used by FedKD to decompress SVD-compressed parameters.
        Default: identity (no modification).
        """
        return parameters

    def get_eval_metrics(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
    ) -> dict[str, Any]:
        """Return algorithm-specific metrics to merge into the evaluate() round log.

        Default: empty dict (no extra metrics).
        """
        return {}

    def metric_keys(self) -> list[str]:
        """Declare every key this hook's :meth:`get_eval_metrics` may emit.

        ``TelemetryManager`` composes the stable CSV header from these plus its
        own algorithm-independent columns, so a key that only appears starting
        round 1 (e.g. FedMAQ's grad-norm stats) doesn't get silently dropped
        because it was absent from round 0. Default: none (no extra metrics).
        """
        return []

    # ------------------------------------------------------------------ #
    # Simulated time / communication model contributions.                #
    # These keep PhysicalCostModel and the telemetry path free of        #
    # ``if alg_name == ...`` dispatch; each hook supplies its own model.  #
    # ------------------------------------------------------------------ #

    def download_size_bytes(
        self,
        strategy: TelemetryFedAvg,
        ndarrays: list[Any],
    ) -> int:
        """Transmitted size of the server->client model broadcast, in bytes.

        Default: the raw float32 size. FedKD overrides this with its SVD-compressed
        download size.
        """
        return sum(int(arr.nbytes) for arr in ndarrays)

    def compute_speed_scale(self) -> float:
        """Multiplicative factor on client compute speed for local-training time.

        Default: 1.0 (no penalty). FedKD returns ``1 / compute_penalty`` to model
        the slowdown from jointly training the student and teacher.
        """
        return 1.0

    def local_train_sample_count(
        self,
        num_samples: int,
        epochs: int,
        num_public: int,
        public_epochs: int,
        server_round: int,
    ) -> float:
        """Effective number of sample-epochs processed during local training.

        Divided by (scaled) client compute speed to get the training delay.
        Default: ``num_samples * epochs``. FedMD overrides this to fold in its
        round-1 public/private pre-training.
        """
        return num_samples * epochs

    def server_sim_time(
        self,
        strategy: TelemetryFedAvg,
        results: list[tuple[ClientProxy, FitRes]],
        aggregated_parameters: Parameters | None,
    ) -> float:
        """Simulated server-side compute time added to the round, in seconds.

        Default: 0.0 (no server compute). FedMAQ and FedAvgKD override this with
        their server-side distillation delay.
        """
        return 0.0
