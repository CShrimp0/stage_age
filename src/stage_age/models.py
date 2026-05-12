from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if name != "resnet18":
        raise ValueError(f"Unsupported model: {name}")

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
