"""训练两个模型，并把逐模型预测、错误样本和混淆矩阵写入磁盘。"""

import pandas as pd

from config import OUTPUT_DIR
from data_utils import load_dataset, make_split
from evaluation import evaluate_model
from models import build_models


def train_models():
    x_train, x_test, y_train, y_test = make_split(load_dataset())
    print(f"train_shape={x_train.shape}, test_shape={x_test.shape}")
    print(f"train_positive_rate={y_train.mean():.4f}, test_positive_rate={y_test.mean():.4f}")

    metric_rows, matrices, fitted_models = [], {}, {}
    for name, model in build_models().items():
        # TODO(FILL-08): 只用训练集拟合，再在测试集上调用 evaluate_model。
        model.fit(x_train, y_train)
        metrics, matrix, errors, predictions = evaluate_model(
            model, x_test, y_test
        )
        metric_rows.append({"model": name, **metrics})
        matrices[name] = matrix
        fitted_models[name] = model

        errors.to_csv(OUTPUT_DIR / f"{name}_errors.csv", encoding="utf-8-sig")
        pd.DataFrame(matrix).to_csv(
            OUTPUT_DIR / f"{name}_confusion.csv", index=False
        )
        pd.DataFrame({"true_label": y_test, name: predictions}).to_csv(
            OUTPUT_DIR / f"{name}_predictions.csv", encoding="utf-8-sig"
        )

    metrics_frame = pd.DataFrame(metric_rows).sort_values(
        "f1", ascending=False
    )
    if metrics_frame is None:
        raise NotImplementedError("请完成当前教学填空")
    metrics_frame.to_csv(OUTPUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    return metrics_frame, matrices, fitted_models
