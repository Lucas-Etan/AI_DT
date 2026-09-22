"""集中计算指标、混淆矩阵，并提取错误分类样本。"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, features: pd.DataFrame, target: pd.Series):
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    # zero_division=0：模型完全没预测出正类时返回 0，而不是产生警告。
    # TODO(FILL-05): 计算 accuracy、precision、recall、f1，处理零除。
    metrics = None
    # TODO(FILL-06): 按 [0, 1] 固定类别顺序计算混淆矩阵。
    matrix = None

    # 保留原始行索引，便于回到输入特征定位错误样本。
    # TODO(FILL-07): 用布尔掩码提取错误样本并标记 FN/FP。
    errors = None
    if metrics is None or matrix is None or errors is None:
        raise NotImplementedError("请完成当前教学填空")
    return metrics, matrix, errors, predictions
