"""CFD client-side fit strategy (fresh init + optional server-label distillation
+ private CE + upstream soft-label quantization).

See ``core/strategy_hooks/cfd.py`` for the full round data-flow and fidelity
caveats. Clients hold no persistent model state (fresh init each round, per the
paper's design); only the tiny upstream delta-reference codes persist, via
``client.state`` (``Context.state``, the same mechanism ``postprocess.py`` uses).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.app import ArrayRecord

from fedmaq.core.client_hooks.base import ClientFitStrategy, standard_evaluate
from fedmaq.core.client_hooks.training_skeleton import StepResult, run_epochs
from fedmaq.core.models import set_model_parameters
from fedmaq.core.softlabel_codec import (
    codes_from_bytes,
    constrained_quantize,
    dequantize,
    encode_bytes,
)

if TYPE_CHECKING:
    from fedmaq.core.client import GenericClient

_PREV_UP_CODES_KEY = "cfd_prev_up_codes"


class CFDFit(ClientFitStrategy):
    """CFD: fresh-init local model each round; digest server soft-labels (KL,
    round >= 2) then train on private data (CE). Returns quantized soft-label
    codes on the shared public proxy set as ``parameters`` (not model weights);
    ``bytes_uploaded`` is the real zlib-measured delta-coded size.
    """

    def fit(
        self,
        client: GenericClient,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        alg_cfg = client.config.get("algorithm", {})
        b_up = int(alg_cfg.get("b_up", 1))
        b_down = int(alg_cfg.get("b_down", 1))
        distill_epochs = int(alg_cfg.get("distill_epochs", 1))
        temperature = float(alg_cfg.get("temperature", 1.0))
        delta_coding = bool(alg_cfg.get("delta_coding", True))
        num_classes = int(client.config.get("dataset", {}).get("num_classes", 10))

        exp_config = client.config.get("experiment", client.config)
        lr = client._get_decayed_lr(config)
        weight_decay = float(exp_config.get("weight_decay", 0.0))
        momentum = float(exp_config.get("momentum", alg_cfg.get("momentum", 0.9)))
        epochs = int(config.get("epochs", exp_config.get("local_epochs", 5)))

        # CFD's protocol (Sattler et al. 2022, Sec. II-B step 2) requires clients
        # to converge to a shared distilled theta each round before private
        # training, achieved in the paper via seed-synchronized distillation
        # (since real FD can't transmit weights). This simulation has the
        # persistent server_model's weights available in-process via Flower's
        # existing parameters channel (populated by CFDHook.aggregate_fit) --
        # loading them here reproduces "same starting theta for all clients"
        # directly instead of reconstructing it via many distillation epochs.
        # Communication accounting is unaffected: CFD's uploaded/downloaded
        # bytes are still soft-label codes only (see download_size_bytes).
        if parameters:
            set_model_parameters(client.model, parameters)

        optimizer = torch.optim.SGD(
            client.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )

        server_round = int(config.get("server_round", 1))
        server_labels_bytes = config.get("cfd_server_labels")

        distill_loss_sum = 0.0
        distill_batches = 0

        if (
            server_round > 1
            and isinstance(server_labels_bytes, bytes)
            and len(server_labels_bytes) > 0
            and client.public_loader is not None
        ):
            server_codes = codes_from_bytes(server_labels_bytes, num_classes)
            server_probs = dequantize(server_codes, b_down)

            kl_criterion = nn.KLDivLoss(reduction="batchmean")

            # Batch offset into server_probs resets each epoch (paper's protocol);
            # distill_loss_sum/distill_batches accumulate across all epochs into a
            # single average, so they're tracked manually in the exact same
            # left-to-right order as the original nested loop -- run_epochs' own
            # per-call averaging would reassociate the float sum across epochs and
            # risk a non-bit-exact result under Decision 40's golden-diff gate.
            def make_distill_step_fn(start_box: list[int]):
                def step_fn(images: torch.Tensor, labels: torch.Tensor) -> StepResult:
                    nonlocal distill_loss_sum, distill_batches
                    batch_len = len(images)
                    batch_targets = torch.tensor(
                        server_probs[start_box[0] : start_box[0] + batch_len],
                        dtype=torch.float32,
                        device=client.device,
                    )
                    start_box[0] += batch_len
                    logits = client.model(images)
                    log_probs = F.log_softmax(logits / temperature, dim=1)
                    loss = kl_criterion(log_probs, batch_targets) * (temperature**2)
                    distill_loss_sum += loss.item()
                    distill_batches += 1
                    return StepResult(loss=loss)

                return step_fn

            for _ in range(distill_epochs):
                run_epochs(
                    model=client.model,
                    loader=client.public_loader,
                    optimizer=optimizer,
                    epochs=1,
                    step_fn=make_distill_step_fn([0]),
                    device=client.device,
                )

        ce_criterion = nn.CrossEntropyLoss()

        def ce_step_fn(images: torch.Tensor, labels: torch.Tensor) -> StepResult:
            outputs = client.model(images)
            loss = ce_criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            return StepResult(
                loss=loss,
                correct=(predicted == labels).sum().item(),
                total=labels.size(0),
            )

        ce_result = run_epochs(
            model=client.model,
            loader=client.trainloader,
            optimizer=optimizer,
            epochs=epochs,
            step_fn=ce_step_fn,
            device=client.device,
        )

        # Soft-label predictions on the public proxy set, sent instead of weights.
        if client.public_loader is not None:
            client.model.eval()
            probs_list = []
            with torch.no_grad():
                for images, _ in client.public_loader:
                    images = images.to(client.device)
                    logits = client.model(images)
                    probs_list.append(F.softmax(logits / temperature, dim=1).cpu().numpy())
            predictions = np.concatenate(probs_list, axis=0).astype(np.float32)
        else:
            predictions = np.zeros((1, num_classes), dtype=np.float32)

        codes = constrained_quantize(predictions, b_up)

        prev_codes = None
        state = getattr(client, "state", None)
        if state is not None:
            prev_record = state.get(_PREV_UP_CODES_KEY)
            if prev_record is not None:
                prev_codes = prev_record.to_numpy_ndarrays()[0]

        nbytes, codes_for_next = encode_bytes(codes, prev_codes, delta=delta_coding)

        if state is not None:
            state[_PREV_UP_CODES_KEY] = ArrayRecord(numpy_ndarrays=[codes_for_next])

        avg_ce_loss = ce_result.avg_loss
        avg_distill_loss = distill_loss_sum / distill_batches if distill_batches > 0 else 0.0
        avg_train_acc = ce_result.accuracy if ce_result.accuracy is not None else 0.0

        return (
            [codes.astype(np.int64)],
            len(client.trainloader.dataset),
            {
                "bytes_uploaded": nbytes,
                "partition_id": int(client.cid),
                "local_loss": avg_ce_loss,
                "train_loss": avg_ce_loss,
                "train_acc": avg_train_acc,
                "epochs_trained": epochs,
                "distill_loss": avg_distill_loss,
            },
        )

    def evaluate(
        self,
        client: GenericClient,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, Any]]:
        return standard_evaluate(client, parameters, config, load_parameters=False)
