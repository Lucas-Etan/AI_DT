"""阶段一：检查数据规模、类型、缺失值、重复值和类别分布。"""

from config import OUTPUT_DIR, TARGET_COLUMN
from data_utils import load_dataset


def main() -> None:
    frame = load_dataset()
    feature_frame = frame.drop(columns=TARGET_COLUMN)

    print(f"shape={frame.shape}")
    print(f"feature_count={feature_frame.shape[1]}")
    print(f"missing_values={int(frame.isna().sum().sum())}")
    print(f"duplicate_rows={int(frame.duplicated().sum())}")
    print("class_counts (0=class_0, 1=class_1):")
    print(frame[TARGET_COLUMN].value_counts().sort_index().to_string())

    # 保存完整描述统计，终端只展示前三个特征，避免输出过长。
    summary = feature_frame.describe().T
    summary.to_csv(OUTPUT_DIR / "feature_summary.csv", encoding="utf-8-sig")
    print("selected_summary:")
    print(summary.loc[["mean radius", "mean texture", "mean area"]].round(3))
    print("saved=outputs/feature_summary.csv")


if __name__ == "__main__":
    main()
