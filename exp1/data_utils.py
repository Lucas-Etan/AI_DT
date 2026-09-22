"""加载内置数据，并以可复现、分层的方式划分数据集。"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from config import SEED, TARGET_COLUMN, TEST_SIZE


def load_dataset() -> pd.DataFrame:
    """返回含 30 个特征和一个二分类标签的 DataFrame。"""
    dataset = load_breast_cancer(as_frame=True)
    frame = dataset.frame.copy()

    # 原始类别 0 有 212 条、类别 1 有 357 条；反转后让较少类别成为正类 1。
    frame = frame.rename(columns={"target": TARGET_COLUMN})
    frame[TARGET_COLUMN] = 1 - frame[TARGET_COLUMN]
    # raise NotImplementedError("请完成当前教学填空")
    return frame


def make_split(frame: pd.DataFrame):
    """分离特征与标签，并保留类别比例完成训练/测试划分。"""
    features = frame.drop(columns=TARGET_COLUMN)
    target = frame[TARGET_COLUMN]

    return train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )
    raise NotImplementedError("请完成当前教学填空")
