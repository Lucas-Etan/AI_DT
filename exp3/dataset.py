from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2


CLASS_MAP = {3: 0, 5: 1}  # CIFAR-10: cat->0, dog->1
CLASS_NAMES = ["cat", "dog"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BinarySubset(Dataset):
    """保存原始 CIFAR-10 索引，并在读取时应用指定变换。"""

    def __init__(self, base, indices, transform):
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        raw_index = self.indices[item]
        image, old_label = self.base[raw_index]
        new_label = CLASS_MAP[old_label]
        return self.transform(image), new_label


def stratified_indices(targets, seed):
    """分别打乱猫、狗索引，保证每个集合类别均衡。"""
    selected = {}
    generator = torch.Generator().manual_seed(seed)
    for old_label in CLASS_MAP:
        pool = torch.tensor([
            i for i, y in enumerate(targets) if y == old_label
        ])
        selected[old_label] = pool[
            torch.randperm(len(pool), generator=generator)
        ].tolist()
    return selected


def make_transforms(image_size):
    """训练集含随机增强；验证和测试集只做确定性预处理。"""
    common = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    eval_tf = v2.Compose([
        v2.Resize((image_size, image_size)), *common
    ])
    train_tf = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(
            (image_size, image_size), scale=(0.80, 1.0)
        ),
        v2.RandomHorizontalFlip(p=0.5),
        *common[1:],
    ])
    return train_tf, eval_tf


def build_datasets(root, profile, seed=42):
    """训练/验证来自官方训练集的不相交索引，测试来自官方测试集。"""
    root = Path(root)
    train_base = CIFAR10(root=root, train=True, download=True)
    test_base = CIFAR10(root=root, train=False, download=True)
    pools = stratified_indices(train_base.targets, seed)
    test_pools = stratified_indices(test_base.targets, seed + 1)
    train_idx, val_idx, test_idx = [], [], []
    for old_label in CLASS_MAP:
        split = profile.train_per_class
        train_idx += pools[old_label][:split]
        val_idx += pools[old_label][
            split:split + profile.val_per_class
        ]
        test_idx += test_pools[old_label][:profile.test_per_class]
    train_tf, eval_tf = make_transforms(profile.image_size)
    return (
        BinarySubset(train_base, train_idx, train_tf),
        BinarySubset(train_base, val_idx, eval_tf),
        BinarySubset(test_base, test_idx, eval_tf),
    )


def build_loaders(root, profile, seed=42):
    """Windows 使用 num_workers=0，错误栈清晰且不会复制进程内存。"""
    train_ds, val_ds, test_ds = build_datasets(root, profile, seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, profile.batch_size, shuffle=True,
        num_workers=0, generator=generator
    )
    val_loader = DataLoader(
        val_ds, profile.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, profile.batch_size, shuffle=False, num_workers=0
    )
    return {"train": train_loader, "val": val_loader,
            "test": test_loader}, test_ds
