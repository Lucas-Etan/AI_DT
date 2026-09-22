"""重新加载最佳模型，验证持久化前后的预测一致。"""

import joblib
import pandas as pd

from config import MODEL_DIR, OUTPUT_DIR
from data_utils import load_dataset, make_split


def main() -> None:
    _, x_test, _, y_test = make_split(load_dataset())
    # TODO(FILL-12): 从可信路径加载刚刚由本人训练的 Pipeline。
    model = joblib.load(MODEL_DIR / "best_pipeline.joblib")
    loaded_predictions = model.predict(x_test)

    # 最佳模型由 F1 排序得到；从 metrics.csv 恢复其名称。
    metrics = pd.read_csv(OUTPUT_DIR / "metrics.csv")
    best_name = str(metrics.iloc[0]["model"])
    saved_predictions = pd.read_csv(
        OUTPUT_DIR / f"{best_name}_predictions.csv", index_col=0
    )[best_name].to_numpy()

    print(f"loaded_model={type(model).__name__}")
    print(f"sample_count={len(loaded_predictions)}")
    print(f"same_predictions={(loaded_predictions == saved_predictions).all()}")
    preview = pd.DataFrame(
        {"true_label": y_test.iloc[:5], "prediction": loaded_predictions[:5]}
    )
    print(preview.to_string(index=True))


if __name__ == "__main__":
    main()
