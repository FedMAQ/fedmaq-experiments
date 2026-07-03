"""Evaluation metrics and routines for federated learning models."""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from fedmaq.core.models import get_model

logger = logging.getLogger(__name__)


def compute_precision_recall_f1(
    all_preds: np.ndarray, all_labels: np.ndarray, num_classes: int = 10
) -> tuple[float, float, float]:
    """Calculate macro-averaged Precision, Recall, and F1-score using sklearn."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
        labels=list(range(num_classes)),
    )
    return float(precision), float(recall), float(f1)


def evaluate_global_model(
    model: nn.Module,
    test_loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Evaluate a single global model on the test dataset."""
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    loss_sum = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_sum += criterion(outputs, labels).item() * len(labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    loss = loss_sum / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    if all_preds:
        preds_concat = np.concatenate(all_preds)
        labels_concat = np.concatenate(all_labels)
        precision, recall, f1 = compute_precision_recall_f1(
            preds_concat, labels_concat, num_classes=num_classes
        )
    else:
        precision, recall, f1 = 0.0, 0.0, 0.0

    return loss, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_fedmd_ensemble(
    client_paths: list[Path],
    dataset_name: str,
    num_classes: int,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Evaluate each client model in FedMD and average their individual metrics."""
    total_loss = 0.0
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    num_eval_clients = 0

    # Instantiate model architecture once to avoid allocation overhead in loop
    if client_paths:
        try:
            client_model = get_model(dataset_name, num_classes)
            client_model.to(device)
        except Exception as e:
            logger.error(f"Failed to initialize evaluation model architecture: {e}")
            return 0.0, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    else:
        return 0.0, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for path in client_paths:
        try:
            client_model.load_state_dict(torch.load(path, map_location=device))
            client_loss, client_metrics = evaluate_global_model(
                client_model, test_loader, num_classes, device
            )
            total_loss += client_loss
            total_accuracy += client_metrics["accuracy"]
            total_precision += client_metrics["precision"]
            total_recall += client_metrics["recall"]
            total_f1 += client_metrics["f1"]
            num_eval_clients += 1
        except Exception as e:
            logger.warning(f"Failed to load or evaluate model at {path}: {e}")

    if num_eval_clients > 0:
        loss = total_loss / num_eval_clients
        metrics = {
            "accuracy": total_accuracy / num_eval_clients,
            "precision": total_precision / num_eval_clients,
            "recall": total_recall / num_eval_clients,
            "f1": total_f1 / num_eval_clients,
        }
    else:
        loss = 0.0
        metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    return loss, metrics
