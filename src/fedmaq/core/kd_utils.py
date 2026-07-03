"""Shared knowledge distillation helpers used by multiple strategy hooks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def run_server_side_kd(
    student_model: nn.Module,
    teachers: list[nn.Module],
    public_loader: DataLoader,
    temperature: float,
    learning_rate: float,
    momentum: float,
    device: torch.device,
    epochs: int = 1,
) -> None:
    """Run server-side knowledge distillation to transfer ensemble knowledge to student model.

    The student is updated in-place via SGD minimising KL divergence against the
    soft-label average of all teacher outputs.
    """
    optimizer = torch.optim.SGD(
        student_model.parameters(), lr=learning_rate, momentum=momentum
    )
    kl_criterion = nn.KLDivLoss(reduction="batchmean")

    student_model.train()
    for _ in range(epochs):
        for images, _ in public_loader:
            images = images.to(device)

            # Get soft targets from teachers
            with torch.no_grad():
                teacher_soft_preds_list = []
                for teacher in teachers:
                    t_out = teacher(images)
                    teacher_soft_preds_list.append(
                        F.softmax(t_out / temperature, dim=1)
                    )
                teacher_soft_preds = torch.stack(teacher_soft_preds_list).mean(dim=0)

            optimizer.zero_grad()
            student_logits = student_model(images)
            student_log_soft = F.log_softmax(student_logits / temperature, dim=1)

            # Distillation loss
            loss = kl_criterion(student_log_soft, teacher_soft_preds) * (temperature**2)
            loss.backward()
            optimizer.step()
