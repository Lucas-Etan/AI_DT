"""实验的公共配置：固定随机性并集中管理输出路径。"""

from pathlib import Path

# 固定随机种子，使划分和模型结果可以复现。
SEED = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "is_class_1"
CLASS_NAMES = ["class_0", "class_1"]

# 用脚本所在目录定位文件，避免从不同工作目录运行时写错位置。
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = PROJECT_DIR / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
