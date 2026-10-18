from dataclasses import dataclass
import random

import torch


@dataclass(frozen=True)
class Profile:
    train_per_class: int
    val_per_class: int
    test_per_class: int
    epochs: int
    image_size: int
    batch_size: int


PROFILES = {
    "smoke": Profile(40, 20, 30, 1, 64, 32),
    "teaching": Profile(800, 200, 250, 3, 96, 64),
}


def seed_everything(seed: int = 42) -> None:
    """固定模型初始化、抽样和数据增强所用的随机数。"""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
