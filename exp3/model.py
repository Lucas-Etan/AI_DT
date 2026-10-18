from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(strategy, num_classes=2):
    """构建冻结骨干或部分解冻 layer4 的 ResNet-18。"""
    if strategy not in {"frozen", "partial"}:
        raise ValueError("strategy 必须是 frozen 或 partial")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    if strategy == "partial":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    return model


def set_transfer_train_mode(model):
    """启用训练，同时固定无可训练参数的 BatchNorm 统计量。"""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            trainable = any(
                p.requires_grad for p in module.parameters()
            )
            if not trainable:
                module.eval()


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
