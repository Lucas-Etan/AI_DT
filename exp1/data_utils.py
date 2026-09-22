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
    # TODO(FILL-01): 将原始 0/1 互换，得到中性的 class_0/class_1 标签。
    raise NotImplementedError("请完成当前教学填空")
    return frame


def make_split(frame: pd.DataFrame):
    """分离特征与标签，并保留类别比例完成训练/测试划分。"""
    features = frame.drop(columns=TARGET_COLUMN)
    target = frame[TARGET_COLUMN]

    # TODO(FILL-02): 固定测试比例和种子，并按 target 分层划分。
    raise NotImplementedError("请完成当前教学填空")
