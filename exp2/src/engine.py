from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """完成一次训练轮次，返回平均损失与准确率。"""
    model.train()
    loss_sum = correct = sample_count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        sample_count += batch_size
    return loss_sum / sample_count, correct / sample_count


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    """关闭梯度完成评估，并返回标签以便画混淆矩阵。"""
    model.eval()
    loss_sum = correct = sample_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        predictions = logits.argmax(dim=1)
        batch_size = labels.size(0)
        loss_sum += loss_fn(logits, labels).item() * batch_size
        correct += (predictions == labels).sum().item()
        sample_count += batch_size
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predictions.cpu().tolist())
    return loss_sum / sample_count, correct / sample_count, y_true, y_pred
