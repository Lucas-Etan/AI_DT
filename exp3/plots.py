from pathlib import Path

import matplotlib.pyplot as plt
import torch

from dataset import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD


def save_confusion_matrix(cm, path):
    """保存行是真实类别、列是预测类别的混淆矩阵。"""
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm.numpy(), cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, int(cm[row, col]), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASS_NAMES,
           yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="True")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def denormalize(image):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)


def save_mistakes(mistakes, path):
    """最多展示 12 个高置信度错误；若没有错误则写出说明图。"""
    count = max(len(mistakes), 1)
    rows = (count + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (confidence, image, true, pred) in zip(axes.flat, mistakes):
        ax.imshow(denormalize(image).permute(1, 2, 0).numpy())
        ax.set_title(f"true={CLASS_NAMES[true]} pred={CLASS_NAMES[pred]}\nconf={confidence:.2f}")
        ax.axis("off")
    if not mistakes:
        axes[0, 0].text(0.5, 0.5, "No mistakes", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

