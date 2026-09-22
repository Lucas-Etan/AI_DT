"""使用无界面后端保存结果图，适合实验室和命令行环境。"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from config import CLASS_NAMES, OUTPUT_DIR


def save_confusion_plot(matrices: dict) -> None:
    fig, axes = plt.subplots(1, len(matrices), figsize=(10, 4))
    for axis, (name, matrix) in zip(axes, matrices.items(), strict=True):
        display = ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES)
        display.plot(ax=axis, colorbar=False, cmap="Blues")
        axis.set_title(name.replace("_", " "))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=160)
    plt.close(fig)


def save_metric_plot(metrics_frame: pd.DataFrame) -> None:
    chart = metrics_frame.set_index("model").plot.bar(
        figsize=(9, 5), ylim=(0.80, 1.00), rot=0
    )
    chart.set_ylabel("score")
    chart.set_title("Model metric comparison")
    chart.legend(loc="lower right")
    chart.figure.tight_layout()
    chart.figure.savefig(OUTPUT_DIR / "metric_comparison.png", dpi=160)
    plt.close(chart.figure)
