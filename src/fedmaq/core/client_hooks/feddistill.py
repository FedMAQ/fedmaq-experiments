"""FedDistill+ client-side fit strategy (FedAvg weights + label-wise logit distillation).

Reference: Zhu et al. 2021, "Data-Free Knowledge Distillation for Heterogeneous
Federated Learning" (FedGen codebase, ``userFedDistill.py``). FEDDISTILL+ (the
``'FL' in algorithm`` variant) shares both model parameters (via FedAvg) and a
per-class mean-logit matrix used as a distillation target.

Deviation from the reference: Flower's simulation recreates clients each round, so
the LogitTracker is per-round (accumulated across the E local epochs of this round)
rather than the reference's cross-round cumulative sum. This is a natural fit for
ephemeral clients and avoids dragging stale early-round logits into the target; the
logit matrix is a function of this round's global weights and local data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fedmaq.core.client_hooks.base import ClientFitStrategy
from fedmaq.core.models import get_model_parameters, set_model_parameters

if TYPE_CHECKING:
    from fedmaq.core.client import GenericClient


class LogitTracker:
    """Accumulates per-class summed logits to produce a mean-logit matrix.

    ``label_counts`` is initialized to ones (not zeros) so classes absent from a
    client's shard yield a finite all-zero mean row instead of a 0/0 NaN. This
    matters under strong non-IID (Dirichlet alpha=0.1), where most clients are
    missing most classes; ``softmax(zeros)`` is a benign uniform target.
    """

    def __init__(self, num_labels: int) -> None:
        self.num_labels = num_labels
        self.label_counts = torch.ones(num_labels)
        self.logit_sums = torch.zeros((num_labels, num_labels))

    def update(self, logits: torch.Tensor, y: torch.Tensor) -> None:
        """Add a batch of raw logits (``[batch, num_labels]``) keyed by label ``y``."""
        logits = logits.detach().cpu()
        y = y.detach().cpu()
        batch_labels, batch_counts = y.unique(dim=0, return_counts=True)
        self.label_counts[batch_labels] += batch_counts.float()
        # Expand label to logit width, scatter-add each sample's logits into its row.
        labels = y.view(y.size(0), 1).expand(-1, logits.size(1))
        batch_sums = torch.zeros((self.num_labels, self.num_labels))
        batch_sums.scatter_add_(0, labels, logits)
        self.logit_sums += batch_sums

    def avg(self) -> np.ndarray:
        """Return the per-class mean-logit matrix ``[num_labels, num_labels]``."""
        res = self.logit_sums / self.label_counts.unsqueeze(1)
        return res.numpy().astype(np.float32)


def logits_to_bytes(matrix: np.ndarray) -> bytes:
    """Serialize a float32 logit matrix for the Flower metrics/config channel."""
    return matrix.astype(np.float32).tobytes()


def bytes_to_logits(buf: bytes, num_labels: int) -> np.ndarray:
    """Deserialize a ``[num_labels, num_labels]`` logit matrix, failing loud on drift."""
    arr = np.frombuffer(buf, dtype=np.float32)
    expected = num_labels * num_labels
    if arr.size != expected:
        raise ValueError(
            f"FedDistill logit buffer has {arr.size} floats but expected "
            f"{expected} (= num_labels^2 for num_labels={num_labels})."
        )
    return arr.reshape(num_labels, num_labels).copy()


class FedDistillFit(ClientFitStrategy):
    """FEDDISTILL+: local training regularized toward the broadcast per-class logits.

    Loss is ``CE + reg_alpha * KLDiv(log_softmax(z), softmax(global_logits[y]))`` once
    global logits have been received (round >= 2); plain cross-entropy otherwise.
    Returns FedAvg model weights plus this round's per-class mean-logit matrix
    (as bytes in the fit metrics).
    """

    def fit(
        self,
        client: GenericClient,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        set_model_parameters(client.model, parameters)

        num_classes = int(client.config.get("dataset", {}).get("num_classes", 10))
        alg_cfg = client.config.get("algorithm", {})
        reg_alpha = float(alg_cfg.get("reg_alpha", 1.0))

        # Global per-class logits broadcast by the server (absent in round 1).
        global_logits: torch.Tensor | None = None
        gl_bytes = config.get("global_logits")
        if isinstance(gl_bytes, bytes) and len(gl_bytes) > 0:
            global_logits = torch.tensor(
                bytes_to_logits(gl_bytes, num_classes), device=client.device
            )

        exp_config = client.config.get("experiment", client.config)
        lr = client._get_decayed_lr(config)
        epochs = int(config.get("epochs", exp_config.get("local_epochs", 5)))
        weight_decay = float(exp_config.get("weight_decay", 0.0))
        momentum = float(exp_config.get("momentum", alg_cfg.get("momentum", 0.9)))

        optimizer = torch.optim.SGD(
            client.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )
        ce_criterion = nn.CrossEntropyLoss()
        kl_criterion = nn.KLDivLoss(reduction="batchmean")
        tracker = LogitTracker(num_classes)

        client.model.train()
        for _ in range(epochs):
            for images, labels in client.trainloader:
                images, labels = images.to(client.device), labels.to(client.device)
                optimizer.zero_grad()
                logits = client.model(images)
                tracker.update(logits, labels)

                loss = ce_criterion(logits, labels)
                if global_logits is not None:
                    target_p = F.softmax(global_logits[labels], dim=1)
                    reg_loss = kl_criterion(F.log_softmax(logits, dim=1), target_p)
                    loss = loss + reg_alpha * reg_loss
                loss.backward()
                optimizer.step()

        # FEDDISTILL+ shares full (unquantized) weights via FedAvg.
        updated_params = get_model_parameters(client.model)
        logit_bytes = logits_to_bytes(tracker.avg())
        params_bytes = sum(int(p.nbytes) for p in updated_params)

        return (
            updated_params,
            len(client.trainloader.dataset),
            {
                "bytes_uploaded": params_bytes + len(logit_bytes),
                "partition_id": int(client.cid),
                "local_loss": 0.0,
                "client_logits": logit_bytes,
            },
        )
