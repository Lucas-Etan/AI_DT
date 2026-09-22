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
    metrics = {
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
    }
    # TODO(FILL-06): 按 [0, 1] 固定类别顺序计算混淆矩阵。
    matrix = confusion_matrix(target, predictions, labels=[0, 1])

    # 保留原始行索引，便于回到输入特征定位错误样本。
    # TODO(FILL-07): 用布尔掩码提取错误样本并标记 FN/FP。
    error_mask = target.to_numpy() != predictions
    errors = features.loc[error_mask].copy()
    errors["true_label"] = target.loc[error_mask].to_numpy()
    errors["predicted_label"] = predictions[error_mask]
    errors["positive_probability"] = probabilities[error_mask]
    errors["error_type"] = np.where(
        errors["true_label"].eq(1), "false_negative", "false_positive"
    )
    if metrics is None or matrix is None or errors is None:
        raise NotImplementedError("请完成当前教学填空")
    return metrics, matrix, errors, predictions
