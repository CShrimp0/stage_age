from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import torch.nn as nn


DEFAULT_USFM_ADAPTER_PATH = Path("/home/szdx/LNX/usage_predict_autoresearch/usfm_adapter.py")


def _load_adapter_module(adapter_path: str | Path = DEFAULT_USFM_ADAPTER_PATH):
    adapter_path = Path(adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"USFM adapter not found: {adapter_path}")
    spec = importlib.util.spec_from_file_location("stage_age_external_usfm_adapter", adapter_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import USFM adapter from {adapter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class USFMLinearProbe(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | Path,
        adapter_path: str | Path = DEFAULT_USFM_ADAPTER_PATH,
        image_size: int = 224,
        global_pool: str = "token",
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"USFM checkpoint not found: {checkpoint_path}")

        adapter_module = _load_adapter_module(adapter_path)
        self.backbone = adapter_module.USFMEncoderAdapter(
            image_size=image_size,
            pretrained_path=str(checkpoint_path),
            global_pool=global_pool,
        )
        self.feature_dim = int(self.backbone.feature_dim)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        return self.head(features)


class USFMClassifier(nn.Module):
    def __init__(
        self,
        checkpoint_path: str | Path,
        adapter_path: str | Path = DEFAULT_USFM_ADAPTER_PATH,
        image_size: int = 224,
        global_pool: str = "token",
        num_classes: int = 3,
        head_type: str = "linear",
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"USFM checkpoint not found: {checkpoint_path}")

        adapter_module = _load_adapter_module(adapter_path)
        self.backbone = adapter_module.USFMEncoderAdapter(
            image_size=image_size,
            pretrained_path=str(checkpoint_path),
            global_pool=global_pool,
        )
        self.feature_dim = int(self.backbone.feature_dim)
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_blocks = int(unfreeze_last_n_blocks)
        self.trainable_backbone_modules: list[str] = []

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            if self.unfreeze_last_n_blocks > 0:
                self._unfreeze_last_blocks(self.unfreeze_last_n_blocks)

        if head_type == "linear":
            self.head = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, num_classes),
            )
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes),
            )
        else:
            raise ValueError(f"Unsupported USFM head_type: {head_type}")

    def _unfreeze_last_blocks(self, last_n_blocks: int) -> None:
        encoder = getattr(self.backbone, "encoder", None)
        blocks = getattr(encoder, "blocks", None)
        if blocks is None or len(blocks) == 0:
            raise ValueError("USFM encoder does not expose transformer blocks for partial unfreezing.")
        if last_n_blocks > len(blocks):
            raise ValueError(f"Cannot unfreeze {last_n_blocks} blocks; encoder only has {len(blocks)} blocks.")

        start = len(blocks) - last_n_blocks
        for idx in range(start, len(blocks)):
            for param in blocks[idx].parameters():
                param.requires_grad = True
            self.trainable_backbone_modules.append(f"encoder.blocks.{idx}")

        for attr in ("norm", "fc_norm"):
            module = getattr(encoder, attr, None)
            if module is not None:
                trainable = False
                for param in module.parameters():
                    param.requires_grad = True
                    trainable = True
                if trainable:
                    self.trainable_backbone_modules.append(f"encoder.{attr}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone and self.unfreeze_last_n_blocks == 0:
            with torch.no_grad():
                features = self.backbone(x)
        else:
            features = self.backbone(x)
        return self.head(features)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable
