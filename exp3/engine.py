from copy import deepcopy
import time

import torch
from torch import nn

from model import set_transfer_train_mode


def metrics_from_cm(cm):
    """从二维混淆矩阵计算准确率和宏平均 precision/recall/F1。"""
    cm = cm.to(torch.float64)
    accuracy = cm.diag().sum() / cm.sum().clamp_min(1)
    precision = cm.diag() / cm.sum(dim=0).clamp_min(1)
    recall = cm.diag() / cm.sum(dim=1).clamp_min(1)
    f1 = 2 * precision * recall / (
        precision + recall
    ).clamp_min(1e-12)
    return {
        "accuracy": accuracy.item(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
    }


def run_epoch(model, loader, criterion, optimizer=None):
    """optimizer 不为空时训练，否则只评估；两种模式共用同一统计口径。"""
    training = optimizer is not None
    set_transfer_train_mode(model) if training else model.eval()
    loss_sum, sample_count = 0.0, 0
    cm = torch.zeros((2, 2), dtype=torch.int64)
    for images, labels in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        predictions = logits.argmax(dim=1)
        loss_sum += loss.item() * labels.size(0)
        sample_count += labels.size(0)
        for true, pred in zip(labels, predictions):
            # 行表示真实类别，列表示预测类别。
            cm[true.long(), pred.long()] += 1
    stats = metrics_from_cm(cm)
    stats["loss"] = loss_sum / max(sample_count, 1)
    return stats, cm


def fit(model, loaders, strategy, epochs, output_path):
    """只依据验证集保存最佳权重，测试集不参与选模。"""
    lr = 1e-3 if strategy == "frozen" else 2e-4
    trainable = [
        p for p in model.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=lr, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()
    history, best_state, best_acc = [], None, -1.0
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_stats, _ = run_epoch(model, loaders["train"], criterion, optimizer)
        val_stats, _ = run_epoch(model, loaders["val"], criterion)
        history.append({"epoch": epoch, "train": train_stats, "val": val_stats})
        print(f"epoch={epoch}/{epochs} train_acc={train_stats['accuracy']:.3f} "
              f"val_acc={val_stats['accuracy']:.3f}")
        if val_stats["accuracy"] > best_acc:
            best_acc = val_stats["accuracy"]
            best_state = deepcopy(model.state_dict())
    elapsed = time.perf_counter() - started
    torch.save(best_state, output_path)
    model.load_state_dict(best_state)
    return history, elapsed


@torch.inference_mode()
def collect_mistakes(model, loader, limit=12):
    """收集高置信度错误样本，帮助判断模型到底错在哪里。"""
    model.eval()
    mistakes = []
    for images, labels in loader:
        probabilities = model(images).softmax(dim=1)
        confidence, predictions = probabilities.max(dim=1)
        for image, true, pred, conf in zip(images, labels, predictions, confidence):
            if true != pred:
                mistakes.append((conf.item(), image.cpu(), true.item(), pred.item()))
    mistakes.sort(key=lambda row: row[0], reverse=True)
    return mistakes[:limit]
