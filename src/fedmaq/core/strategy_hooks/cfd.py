"""Strategy hook implementing CFD (Compressed Federated Distillation, Sattler et al. 2022).

CFD exchanges quantized soft-labels on a shared unlabeled public proxy set instead
of model weights or gradients. Three mechanisms beyond the existing distillation
baselines (FedMD, FedDistill): constrained uniform soft-label quantization
(``softlabel_codec.constrained_quantize``, b=1 -> one-hot/max-vote), delta +
lossless coding across rounds (``softlabel_codec.encode_bytes``), and server-side
dual distillation -- a persistent server model trained on the aggregated client
soft-labels each round, which generates the next round's downstream targets.

Data flow (per round t)
------------------------
::

    server_model.soft_labels(X_pub)  --Q_{b_down}+delta+zlib-->  clients  [configure_fit]
    client: fresh init -> distill to server labels (KL) -> train private (CE)
            -> soft_labels(X_pub)  --Q_{b_up}+delta+zlib-->  server       [CFDFit.fit]
    server: dequantize + average client soft-labels                       [pre_aggregate_fit]
            -> train server_model on averaged soft-labels (dual distillation)
            -> return server_model params as aggregated_parameters        [aggregate_fit]

Round 1 has no downstream server labels yet (server_model is freshly initialized)
and no delta reference, so ``configure_fit`` skips the broadcast; clients train
private-only and still upload their soft-labels, which seed dual distillation
from round 1 onward.

Fidelity caveats
-----------------
- zlib substitutes for CABAC/arithmetic coding: byte magnitudes are approximate,
  the compressibility trend is faithful.
- The public proxy is FedMAQ's 1600-sample pool (manuscript ``D_proxy=1600``),
  not the paper's ~80k-sample STL-10 -- matches this thesis's simulation scale.
- Clients fresh-init each round with only the tiny upstream delta-reference
  codes persisted (via ``client.state``), faithful to the paper's design and
  coherent with dual distillation carrying knowledge server-side.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import (
    FitIns,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.typing import FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from fedmaq.core.kd_utils import kd_server_sim_time
from fedmaq.core.models import DEVICE, get_client_model, get_model_parameters
from fedmaq.core.partitioning import get_server_loaders
from fedmaq.core.softlabel_codec import (
    codes_to_bytes,
    constrained_quantize,
    dequantize,
    encode_bytes,
)
from fedmaq.core.strategy_hooks.base import StrategyHook

if TYPE_CHECKING:
    from fedmaq.core.strategy import TelemetryFedAvg

logger = logging.getLogger(__name__)


class CFDHook(StrategyHook):
    """Server-side dual distillation + soft-label broadcast/aggregation for CFD.

    Holds a persistent ``server_model`` (the hook is a single long-lived object,
    unlike ephemeral simulated clients) that is refined every round via
    distillation on the aggregated client soft-labels, and whose predictions
    become next round's downstream distillation target.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        dataset_cfg = config.get("dataset", {})
        self.dataset_name = dataset_cfg.get("name", "mnist")
        self.num_classes = int(dataset_cfg.get("num_classes", 10))

        alg_cfg = config.get("algorithm", {})
        self.b_up = int(alg_cfg.get("b_up", 1))
        self.b_down = int(alg_cfg.get("b_down", 1))
        self.distill_epochs = int(alg_cfg.get("distill_epochs", 1))
        self.server_distill_epochs = int(alg_cfg.get("server_distill_epochs", 1))
        self.temperature = float(alg_cfg.get("temperature", 1.0))
        self.delta_coding = bool(alg_cfg.get("delta_coding", True))

        self.device = torch.device(config.get("device") or DEVICE)
        self.server_model = get_client_model("cfd", self.dataset_name, self.num_classes)
        self.server_model.to(self.device)

        self._prev_down_codes: np.ndarray | None = None
        self._last_downstream_bytes = 0
        self._pending_targets: np.ndarray | None = None
        self._public_loader: Any = None

    def _get_public_loader(self, strategy: TelemetryFedAvg) -> Any:
        if self._public_loader is None and strategy.public_indices is not None:
            exp_cfg = self._config.get("experiment", {})
            batch_size = int(exp_cfg.get("batch_size", 64))
            self._public_loader, _ = get_server_loaders(
                self.dataset_name, strategy.public_indices, batch_size=batch_size
            )
        return self._public_loader

    def configure_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        client_instructions: list[tuple[ClientProxy, FitIns]],
    ) -> list[tuple[ClientProxy, FitIns]]:
        if server_round == 1 or not client_instructions:
            return client_instructions

        public_loader = self._get_public_loader(strategy)
        if public_loader is None:
            return client_instructions

        self.server_model.eval()
        probs_list = []
        with torch.no_grad():
            for images, _ in public_loader:
                images = images.to(self.device)
                logits = self.server_model(images)
                probs_list.append(F.softmax(logits / self.temperature, dim=1).cpu().numpy())
        probs = np.concatenate(probs_list, axis=0).astype(np.float32)

        down_codes = constrained_quantize(probs, self.b_down)
        nbytes, codes_for_next = encode_bytes(
            down_codes, self._prev_down_codes, delta=self.delta_coding
        )
        self._prev_down_codes = codes_for_next
        self._last_downstream_bytes = nbytes

        labels_bytes = codes_to_bytes(down_codes)
        for _, fit_ins in client_instructions:
            fit_ins.config["cfd_server_labels"] = labels_bytes

        return client_instructions

    def pre_aggregate_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]] | None:
        self._pending_targets = None
        if not results:
            return None, {}

        probs_list = []
        for _, fit_res in results:
            codes = parameters_to_ndarrays(fit_res.parameters)[0]
            probs_list.append(dequantize(codes, self.b_up))
        self._pending_targets = np.mean(probs_list, axis=0).astype(np.float32)

        # Bypass FedAvg weight-averaging: our client "parameters" are quantized
        # soft-label codes, not model weights.
        return None, {}

    def aggregate_fit(
        self,
        strategy: TelemetryFedAvg,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
        aggregated_parameters: Parameters | None,
        metrics: dict[str, Scalar],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if self._pending_targets is not None:
            public_loader = self._get_public_loader(strategy)
            if public_loader is not None:
                self._train_server_model(public_loader, self._pending_targets)

        aggregated_parameters = ndarrays_to_parameters(
            get_model_parameters(self.server_model)
        )
        return aggregated_parameters, metrics

    def _train_server_model(self, public_loader: Any, targets: np.ndarray) -> None:
        alg_cfg = self._config.get("algorithm", {})
        optimizer = torch.optim.SGD(
            self.server_model.parameters(),
            lr=float(alg_cfg.get("server_kd_lr", 0.01)),
            momentum=float(alg_cfg.get("server_kd_momentum", 0.9)),
        )
        kl_criterion = nn.KLDivLoss(reduction="batchmean")

        self.server_model.train()
        for _ in range(self.server_distill_epochs):
            start = 0
            for images, _ in public_loader:
                images = images.to(self.device)
                batch_len = len(images)
                batch_targets = torch.tensor(
                    targets[start : start + batch_len],
                    dtype=torch.float32,
                    device=self.device,
                )
                start += batch_len

                optimizer.zero_grad()
                logits = self.server_model(images)
                log_probs = F.log_softmax(logits / self.temperature, dim=1)
                loss = kl_criterion(log_probs, batch_targets) * (self.temperature**2)
                loss.backward()
                optimizer.step()

    def local_train_sample_count(
        self,
        num_samples: int,
        epochs: int,
        num_public: int,
        public_epochs: int,
        server_round: int,
    ) -> float:
        base = num_samples * epochs
        if server_round > 1:
            base += num_public * self.distill_epochs
        return base

    def download_size_bytes(
        self,
        strategy: TelemetryFedAvg,
        ndarrays: list[Any],
    ) -> int:
        # CFD sends soft-label codes downstream, not model weights (0 in round 1).
        return self._last_downstream_bytes

    def server_sim_time(
        self,
        strategy: TelemetryFedAvg,
        results: list[tuple[ClientProxy, FitRes]],
        aggregated_parameters: Parameters | None,
    ) -> float:
        if self._pending_targets is None:
            return 0.0
        alg_cfg = self._config.get("algorithm", {})
        num_public = int(
            self._config.get("experiment", {}).get("num_public_samples", 200)
        )
        return kd_server_sim_time(
            num_public=num_public,
            kd_epochs=self.server_distill_epochs,
            num_teachers=1,
            server_compute_speed=float(alg_cfg.get("server_compute_speed", 2000.0)),
        )

    def get_eval_metrics(
        self, strategy: TelemetryFedAvg, server_round: int
    ) -> dict[str, Any]:
        return {}
