"""建立两个可比较的 Pipeline，所有预处理只在训练集上拟合。"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from config import SEED


def build_models() -> dict[str, Pipeline]:
    """返回名称到未训练 Pipeline 的映射。"""
    # TODO(FILL-03): 建立“标准化器 + 逻辑回归”的 Pipeline。
    logistic_pipeline = None

    # TODO(FILL-04): 建立深度和叶节点受限的决策树 Pipeline。
    tree_pipeline = None
    if logistic_pipeline is None or tree_pipeline is None:
        raise NotImplementedError("请完成当前教学填空")
    return {"logistic_regression": logistic_pipeline, "decision_tree": tree_pipeline}
