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

        optimizer = torch.optim.SGD(
            client.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
        )

        server_round = int(config.get("server_round", 1))
        server_labels_bytes = config.get("cfd_server_labels")

        if (
            server_round > 1
            and isinstance(server_labels_bytes, bytes)
            and len(server_labels_bytes) > 0
            and client.public_loader is not None
        ):
            server_codes = codes_from_bytes(server_labels_bytes, num_classes)
            server_probs = dequantize(server_codes, b_down)

            kl_criterion = nn.KLDivLoss(reduction="batchmean")
            client.model.train()
            for _ in range(distill_epochs):
                start = 0
                for images, _ in client.public_loader:
                    images = images.to(client.device)
                    batch_len = len(images)
                    batch_targets = torch.tensor(
                        server_probs[start : start + batch_len],
                        dtype=torch.float32,
                        device=client.device,
                    )
                    start += batch_len

                    optimizer.zero_grad()
                    logits = client.model(images)
                    log_probs = F.log_softmax(logits / temperature, dim=1)
                    loss = kl_criterion(log_probs, batch_targets) * (temperature**2)
                    loss.backward()
                    optimizer.step()

        ce_criterion = nn.CrossEntropyLoss()
        client.model.train()
        for _ in range(epochs):
            for images, labels in client.trainloader:
                images, labels = images.to(client.device), labels.to(client.device)
                optimizer.zero_grad()
                outputs = client.model(images)
                loss = ce_criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Soft-label predictions on the public proxy set, sent instead of weights.
        if client.public_loader is not None:
            client.model.eval()
            probs_list = []
            with torch.no_grad():
                for images, _ in client.public_loader:
                    images = images.to(client.device)
                    logits = client.model(images)
                    probs_list.append(
                        F.softmax(logits / temperature, dim=1).cpu().numpy()
                    )
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

        return (
            [codes.astype(np.int64)],
            len(client.trainloader.dataset),
            {
                "bytes_uploaded": nbytes,
                "partition_id": int(client.cid),
                "local_loss": 0.0,
            },
        )

    def evaluate(
        self,
        client: GenericClient,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, Any]]:
        return standard_evaluate(client, parameters, config, load_parameters=False)
