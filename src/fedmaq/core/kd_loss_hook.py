"""Client-side KD regularization loss hook (FedGKD-style).

Constrains local model logits to remain close to the incoming global model's
predictions during client training, reducing client drift under severe non-IID
skew.  Integrates with :class:`~fedmaq.core.client.LossHook` so it is
transparent to the existing :class:`StandardFit` training loop.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClientKDLossHook:
    """Loss hook adding client-side KD regularization against the global model.

    On each training step the loss becomes::

        L = (1 - alpha) * CE(y_pred, y_true)
            + alpha * KL(softmax(z_global / T) || softmax(z_local / T)) * T^2
            + (mu / 2) * sum ||w - w_global||^2

    where ``z_global`` are the frozen global-model logits, ``z_local`` are
    the current local-model logits, ``w`` is the current local model parameter,
    and ``w_global`` is the global model parameter. The ``T^2`` factor preserves
    gradient magnitudes when ``T > 1`` (Hinton et al. 2015).

    Parameters
    ----------
    alpha : float
        Blending weight in ``[0, 1]``.  ``0`` = pure CE, ``1`` = pure KD.
    temperature : float
        Softmax temperature for the KD term.  ``T > 1`` softens distributions
        to reveal inter-class structure; ``T = 1`` uses raw softmax outputs.
    mu : float
        FedProx-style proximal weight for L2 parameter regularization.
    """

    def __init__(
        self, alpha: float = 0.5, temperature: float = 2.0, mu: float = 0.0
    ) -> None:
        self.alpha = alpha
        self.temperature = temperature
        self.mu = mu
        self._global_model: nn.Module | None = None
        self.global_params: list[torch.Tensor] = []

    # -- LossHook interface --------------------------------------------------

    def on_train_begin(self, model: nn.Module) -> None:
        """Snapshot the incoming global model for reference during training."""
        self._global_model = copy.deepcopy(model)
        self._global_model.eval()
        for p in self._global_model.parameters():
            p.requires_grad = False

        if self.mu > 0.0:
            self.global_params = [
                p.clone().detach() for p in model.parameters() if p.requires_grad
            ]

    def compute_loss(
        self,
        model: nn.Module,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
        inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute blended CE + KD regularization loss + optional proximal term.

        Parameters
        ----------
        inputs : torch.Tensor | None
            Raw input batch (images).  Required for the KD term so the frozen
            global model can produce reference logits.  When ``None`` or when
            the hook is disabled (``alpha <= 0``), falls back to pure CE.
        """
        ce_loss = criterion(outputs, targets)

        if self._global_model is not None and self.alpha > 0.0 and inputs is not None:
            with torch.no_grad():
                global_logits = self._global_model(inputs)
                teacher_soft = F.softmax(global_logits / self.temperature, dim=1)

            student_log_soft = F.log_softmax(outputs / self.temperature, dim=1)
            kd_loss = (
                F.kl_div(student_log_soft, teacher_soft, reduction="batchmean")
                * self.temperature**2
            )
            loss = (1.0 - self.alpha) * ce_loss + self.alpha * kd_loss
        else:
            loss = ce_loss

        if self.mu > 0.0 and self.global_params:
            proximal_term: torch.Tensor | float = 0.0
            params = [p for p in model.parameters() if p.requires_grad]
            for p, gp in zip(params, self.global_params, strict=True):
                proximal_term += torch.sum((p - gp) ** 2)
            loss = loss + (self.mu / 2.0) * proximal_term

        return loss
