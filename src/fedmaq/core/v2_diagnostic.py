"""Opt-in, read-only V2 diagnosis of the executed FedMAQ round."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import Parameters, parameters_to_ndarrays
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from torch.utils.data import DataLoader

from fedmaq.core.models import get_server_model_factory, set_model_parameters
from fedmaq.core.quantization_planner import QuantPlan


@contextmanager
def _preserve_rng() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _new_scores(num_classes: int) -> dict[str, Any]:
    return {
        "samples": 0,
        "correct": 0,
        "loss_sum": 0.0,
        "class_total": [0] * num_classes,
        "class_correct": [0] * num_classes,
    }


def _update_scores(
    scores: dict[str, Any], probabilities: torch.Tensor, labels: torch.Tensor
) -> None:
    predictions = probabilities.argmax(dim=1)
    scores["samples"] += int(labels.numel())
    scores["correct"] += int((predictions == labels).sum().item())
    scores["loss_sum"] += float(
        F.nll_loss(probabilities.clamp_min(1e-12).log(), labels, reduction="sum").item()
    )
    num_classes = len(scores["class_total"])
    totals = torch.bincount(labels, minlength=num_classes).tolist()
    correct = torch.bincount(labels[predictions == labels], minlength=num_classes).tolist()
    for label in range(num_classes):
        scores["class_total"][label] += int(totals[label])
        scores["class_correct"][label] += int(correct[label])


def _finish_scores(scores: dict[str, Any]) -> dict[str, Any]:
    count = scores.pop("samples")
    correct = scores.pop("correct")
    loss_sum = scores.pop("loss_sum")
    return {
        "samples": count,
        "accuracy": correct / count if count else None,
        "cross_entropy": loss_sum / count if count else None,
        **scores,
    }


class V2DiagnosticObserver:
    """Measure model, teacher, and proxy quality without touching the training models."""

    def __init__(
        self,
        *,
        log_dir: Path,
        dataset_name: str,
        num_classes: int,
        device: torch.device,
        validation_loader: DataLoader,
        proxy_loader: DataLoader,
        observed_rounds: list[int],
        temperature: float,
    ) -> None:
        if not observed_rounds or any(round_num < 1 for round_num in observed_rounds):
            raise ValueError("v2_diagnostic.rounds must contain positive round numbers")
        if len(validation_loader.dataset) == 0:  # type: ignore[arg-type]
            raise ValueError("V2 diagnostic requires a nonempty validation loader")
        self.path = log_dir / "v2_diagnostic.jsonl"
        self.model_factory = get_server_model_factory("fedmaq")
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.device = device
        self.validation_loader = validation_loader
        self.proxy_loader = proxy_loader
        self.observed_rounds = frozenset(observed_rounds)
        self.temperature = temperature

    def _model(self, parameters: Parameters) -> nn.Module:
        model = self.model_factory(self.dataset_name, self.num_classes)
        set_model_parameters(model, parameters_to_ndarrays(parameters))
        return model.to(self.device).eval()

    def observe(
        self,
        *,
        server_round: int,
        parameter_average: Parameters,
        post_kd: Parameters,
        results: list[tuple[ClientProxy, FitRes]],
        plan: QuantPlan,
        cumulative_bytes: int,
        round_bytes: int,
        post_process: bool,
    ) -> float:
        if server_round not in self.observed_rounds:
            return 0.0
        started = time.perf_counter()
        with _preserve_rng(), torch.inference_mode():
            average_model = self._model(parameter_average)
            post_model = self._model(post_kd)
            teachers = [self._model(fit_res.parameters) for _, fit_res in results]
            average_scores = _new_scores(self.num_classes)
            post_scores = _new_scores(self.num_classes)
            ensemble_scores = _new_scores(self.num_classes)
            teacher_scores = [_new_scores(self.num_classes) for _ in teachers]
            calibration_bins = [
                {"count": 0, "confidence_sum": 0.0, "correct_sum": 0} for _ in range(10)
            ]
            disagreement_sum = 0.0
            for images, labels in self.validation_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                average_prob = F.softmax(average_model(images), dim=1)
                post_prob = F.softmax(post_model(images), dim=1)
                _update_scores(average_scores, average_prob, labels)
                _update_scores(post_scores, post_prob, labels)
                if not teachers:
                    continue
                teacher_probs = torch.stack(
                    [F.softmax(teacher(images) / self.temperature, dim=1) for teacher in teachers]
                )
                pooled = teacher_probs.mean(dim=0)
                _update_scores(ensemble_scores, pooled, labels)
                for scores, probabilities in zip(teacher_scores, teacher_probs, strict=True):
                    _update_scores(scores, probabilities, labels)
                predictions = pooled.argmax(dim=1)
                disagreement_sum += float(
                    (teacher_probs.argmax(dim=2) != predictions.unsqueeze(0)).sum().item()
                )
                confidence = pooled.max(dim=1).values
                bins = torch.clamp((confidence * 10).long(), max=9)
                for index, calibration_bin in enumerate(calibration_bins):
                    mask = bins == index
                    calibration_bin["count"] += int(mask.sum().item())
                    calibration_bin["confidence_sum"] += float(confidence[mask].sum().item())
                    calibration_bin["correct_sum"] += int(
                        ((predictions == labels) & mask).sum().item()
                    )

            proxy_count = 0
            proxy_entropy_sum = 0.0
            proxy_disagreement_sum = 0.0
            proxy_label_counts = [0] * self.num_classes
            proxy_predicted_counts = [0] * self.num_classes
            if teachers:
                for images, labels in self.proxy_loader:
                    images = images.to(self.device)
                    teacher_probs = torch.stack(
                        [
                            F.softmax(teacher(images) / self.temperature, dim=1)
                            for teacher in teachers
                        ]
                    )
                    pooled = teacher_probs.mean(dim=0)
                    prediction = pooled.argmax(dim=1)
                    proxy_count += int(prediction.numel())
                    proxy_entropy_sum += float(
                        (-(pooled * pooled.clamp_min(1e-12).log()).sum(dim=1)).sum().item()
                    )
                    proxy_disagreement_sum += float(
                        (teacher_probs.argmax(dim=2) != prediction.unsqueeze(0)).sum().item()
                    )
                    for label in range(self.num_classes):
                        proxy_label_counts[label] += int((labels == label).sum().item())
                        proxy_predicted_counts[label] += int((prediction == label).sum().item())

            average = _finish_scores(average_scores)
            post = _finish_scores(post_scores)
            ensemble = _finish_scores(ensemble_scores) if teachers else None
            teacher_summaries = [_finish_scores(scores) for scores in teacher_scores]
            validation_count = average["samples"]
            record = {
                "schema": "v2-diagnostic-1",
                "round": server_round,
                "split": "val",
                "post_process": post_process,
                "round_bytes": round_bytes,
                "cumulative_bidirectional_bytes": cumulative_bytes,
                "parameter_average": average,
                "post_kd_student": post,
                "ensemble": ensemble,
                "post_minus_average_accuracy": post["accuracy"] - average["accuracy"],
                "ensemble_minus_average_accuracy": (
                    ensemble["accuracy"] - average["accuracy"] if ensemble else None
                ),
                "post_minus_ensemble_accuracy": (
                    post["accuracy"] - ensemble["accuracy"] if ensemble else None
                ),
                "ensemble_calibration_bins": calibration_bins if teachers else None,
                "validation_teacher_disagreement_fraction": (
                    disagreement_sum / (validation_count * len(teachers))
                    if teachers and validation_count
                    else None
                ),
                "proxy": {
                    "samples": proxy_count,
                    "mean_ensemble_entropy": (
                        proxy_entropy_sum / proxy_count if proxy_count else None
                    ),
                    "teacher_disagreement_fraction": (
                        proxy_disagreement_sum / (proxy_count * len(teachers))
                        if proxy_count and teachers
                        else None
                    ),
                    "label_counts": proxy_label_counts,
                    "ensemble_predicted_counts": proxy_predicted_counts,
                },
                "clients": [
                    {
                        "partition_id": int(fit_res.metrics["partition_id"]),
                        "num_examples": fit_res.num_examples,
                        "q": plan.client_q.get(client_proxy.cid),
                        "q_hat": plan.client_q_hat.get(client_proxy.cid),
                        "q_max": plan.client_q_max.get(client_proxy.cid),
                        "probe_grad_norm": plan.client_grad_norms.get(client_proxy.cid),
                        "raw_update_norm": fit_res.metrics.get("v2_raw_update_norm"),
                        "raw_to_reconstructed_norm": fit_res.metrics.get(
                            "v2_raw_to_reconstructed_norm"
                        ),
                        "corrected_quantization_residual_norm": fit_res.metrics.get(
                            "v2_corrected_quantization_residual_norm"
                        ),
                        "teacher_validation_accuracy": teacher_summary["accuracy"],
                    }
                    for (client_proxy, fit_res), teacher_summary in zip(
                        results, teacher_summaries, strict=True
                    )
                ],
            }
        elapsed = time.perf_counter() - started
        record["observer_wall_time_sec"] = elapsed
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        return time.perf_counter() - started
