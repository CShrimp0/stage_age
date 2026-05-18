from __future__ import annotations

import argparse
import json
import math
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage_age.data import UltrasoundAgeDataset, build_manifest, make_transforms
from stage_age.metrics import image_level_metrics, subject_level_predictions
from stage_age.report import generate_report
from stage_age.run_multiseed_experiments import (
    check_finite,
    compute_class_weights,
    gpu_name,
    make_loader,
    report_to_csv,
    resolve_device,
    save_json,
    save_predictions,
    set_seed,
    smoke_forward,
    split_summary,
    train_one_epoch,
    unique_output_dir,
    write_model_summary,
)
from stage_age.usfm import DEFAULT_USFM_ADAPTER_PATH, USFMClassifier, count_parameters


USFM_CHECKPOINT = "/home/szdx/LNX/stage-age/USFM_latest.pth"
USFM_ADAPTER = str(DEFAULT_USFM_ADAPTER_PATH)
MAINLINE_SUBJECT_MACRO_F1 = 0.6471
MAINLINE_SUBJECT_BALANCED_ACCURACY = 0.6829


BASE_CONFIG: dict[str, Any] = {
    "data": {
        "image_dir": "/home/szdx/LNX/data/TA/Healthy/Images",
        "characteristics": "/home/szdx/LNX/data/TA/characteristics.xlsx",
        "sheet_name": "Blad1",
        "bins": [18, 45, 65, 101],
        "class_names": ["18-44", "45-64", "65-100"],
        "split": {"train": 0.70, "val": 0.15, "test": 0.15},
    },
    "model": {
        "pretrained": USFM_CHECKPOINT,
        "checkpoint_path": USFM_CHECKPOINT,
        "adapter_path": USFM_ADAPTER,
        "image_size": 224,
        "input_channels": 3,
        "input_mode": "grayscale images converted to RGB",
        "global_pool": "token",
        "head_type": "mlp",
        "dropout": 0.2,
        "freeze_backbone": True,
        "unfreeze_last_n_blocks": 1,
        "trainable": "encoder.blocks.11 + encoder.fc_norm + MLP classification head",
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    },
    "train": {
        "epochs": 30,
        "early_stopping_patience": 7,
        "batch_size": 32,
        "fallback_batch_size": 16,
        "num_workers": 8,
        "fallback_num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "use_class_weights": True,
        "device": "cuda",
        "amp": True,
        "grad_accum_steps": 1,
        "head_lr": 3e-4,
        "backbone_lr": 3e-5,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
        "save_best_by": "val_macro_f1",
        "test_checkpoint": "best.pt",
        "evaluated_checkpoint": "best.pt",
        "use_best_checkpoint_for_test": True,
        "checkpoint_state": "trainable_state_dict",
    },
}


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "partial_ls_lr1e5": {
        "model": {"name": "usfm_partial_ls_lr1e5", "num_outputs": 3, "task": "multiclass"},
        "train": {"backbone_lr": 1e-5},
    },
    "partial_ls_lr3e5": {
        "model": {"name": "usfm_partial_ls_lr3e5", "num_outputs": 3, "task": "multiclass"},
        "train": {"backbone_lr": 3e-5},
    },
    "partial_ls_lr5e5": {
        "model": {"name": "usfm_partial_ls_lr5e5", "num_outputs": 3, "task": "multiclass"},
        "train": {"backbone_lr": 5e-5},
    },
    "ordinal": {
        "model": {
            "name": "usfm_partial_ordinal_lr3e5",
            "num_outputs": 2,
            "task": "ordinal",
            "trainable": "encoder.blocks.11 + encoder.fc_norm + 2-logit ordinal MLP head",
        },
        "train": {"backbone_lr": 3e-5, "label_smoothing": 0.0, "use_class_weights": False},
    },
    "regression_binning": {
        "model": {
            "name": "usfm_partial_regression_binning",
            "num_outputs": 1,
            "task": "regression_binning",
            "trainable": "encoder.blocks.11 + encoder.fc_norm + 1-value regression MLP head",
        },
        "train": {
            "backbone_lr": 3e-5,
            "head_lr": 3e-4,
            "save_best_by": "val_mae",
            "regression_loss": "SmoothL1Loss",
            "use_class_weights": False,
        },
    },
    "focal_loss": {
        "model": {"name": "usfm_partial_focal_loss", "num_outputs": 3, "task": "focal_loss"},
        "train": {
            "backbone_lr": 3e-5,
            "head_lr": 3e-4,
            "save_best_by": "val_macro_f1",
            "focal_gamma": 2.0,
            "focal_alpha": "balanced_class_weights",
            "use_class_weights": True,
        },
    },
    "midclass_weight_1p3": {
        "model": {"name": "usfm_partial_midclass_weight_1p3", "num_outputs": 3, "task": "multiclass"},
        "train": {
            "backbone_lr": 3e-5,
            "head_lr": 3e-4,
            "save_best_by": "val_macro_f1",
            "use_class_weights": True,
            "midclass_weight_multiplier": 1.3,
        },
    },
    "ldl_boundary_w3": {
        "label_strategy": "ldl_boundary_w3",
        "model": {
            "name": "usfm_partial_ldl_boundary_w3",
            "num_outputs": 3,
            "task": "ldl_boundary",
            "trainable": "encoder.blocks.11 + encoder.fc_norm + LDL MLP classification head",
        },
        "train": {
            "backbone_lr": 3e-5,
            "head_lr": 3e-4,
            "save_best_by": "val_macro_f1",
            "use_class_weights": True,
            "soft_label_width": 3,
        },
    },
    "ldl_gaussian_sigma3": {
        "label_strategy": "ldl_gaussian_sigma3",
        "model": {
            "name": "usfm_partial_ldl_gaussian_sigma3",
            "num_outputs": 3,
            "task": "ldl_gaussian",
            "trainable": "encoder.blocks.11 + encoder.fc_norm + Gaussian LDL MLP classification head",
        },
        "train": {
            "backbone_lr": 3e-5,
            "head_lr": 3e-4,
            "save_best_by": "val_macro_f1",
            "use_class_weights": True,
            "ldl_sigma": 3.0,
        },
    },
    "ldl_multitask_w3_lam0p3": {
        "label_strategy": "ldl_multitask_w3_lam0p3",
        "model": {
            "name": "usfm_partial_ldl_multitask_w3_lam0p3",
            "num_outputs": 4,
            "task": "ldl_multitask",
            "trainable": "encoder.blocks.11 + encoder.fc_norm + 3-logit LDL classification output + 1-value age output",
        },
        "train": {
            "backbone_lr": 3e-5,
            "head_lr": 3e-4,
            "save_best_by": "val_macro_f1",
            "use_class_weights": True,
            "soft_label_width": 3,
            "regression_loss": "SmoothL1Loss",
            "multitask_regression_lambda": 0.3,
        },
    },
    "subject_mean_pool_k3": {
        "label_strategy": "hard_label_subject_pool",
        "model": {
            "name": "usfm_subject_mean_pool_k3",
            "num_outputs": 3,
            "task": "subject_pool",
            "pooling": "mean",
            "k_images": 3,
            "eval_uses_all_images": True,
            "trainable": "encoder.blocks.11 + encoder.fc_norm + mean-pooled MLP classification head",
        },
        "train": {
            "backbone_lr": 3e-5,
            "head_lr": 3e-4,
            "save_best_by": "val_macro_f1",
            "use_class_weights": True,
            "training_unit": "subject",
        },
    },
    "subject_attention_pool_k3": {
        "label_strategy": "hard_label_subject_pool",
        "model": {
            "name": "usfm_subject_attention_pool_k3",
            "num_outputs": 3,
            "task": "subject_pool",
            "pooling": "attention",
            "k_images": 3,
            "eval_uses_all_images": True,
            "trainable": "encoder.blocks.11 + encoder.fc_norm + attention-pooled MLP classification head",
        },
        "train": {
            "backbone_lr": 3e-5,
            "head_lr": 3e-4,
            "save_best_by": "val_macro_f1",
            "use_class_weights": True,
            "training_unit": "subject",
        },
    },
}


ALIASES = {
    "all": ["partial_ls_lr1e5", "partial_ls_lr3e5", "partial_ls_lr5e5", "ordinal"],
    "all_next": ["regression_binning", "focal_loss", "midclass_weight_1p3"],
    "partial_ls_lr1e5": ["partial_ls_lr1e5"],
    "partial_ls_lr3e5": ["partial_ls_lr3e5"],
    "partial_ls_lr5e5": ["partial_ls_lr5e5"],
    "ordinal": ["ordinal"],
    "regression_binning": ["regression_binning"],
    "focal_loss": ["focal_loss"],
    "midclass_weight_1p3": ["midclass_weight_1p3"],
    "all_ldl": ["ldl_boundary_w3", "ldl_gaussian_sigma3", "ldl_multitask_w3_lam0p3"],
    "ldl_boundary_w3": ["ldl_boundary_w3"],
    "ldl_gaussian_sigma3": ["ldl_gaussian_sigma3"],
    "ldl_multitask_w3_lam0p3": ["ldl_multitask_w3_lam0p3"],
    "all_subject_pool": ["subject_mean_pool_k3", "subject_attention_pool_k3"],
    "subject_mean_pool_k3": ["subject_mean_pool_k3"],
    "subject_attention_pool_k3": ["subject_attention_pool_k3"],
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def experiment_config(experiment: str) -> dict[str, Any]:
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unsupported experiment: {experiment}")
    config = deep_update(BASE_CONFIG, EXPERIMENTS[experiment])
    config["experiment_name"] = str(config["model"]["name"])
    config.setdefault("label_strategy", str(config["model"].get("task", "hard_label")))
    return config


def check_preflight() -> None:
    for path in [
        Path(BASE_CONFIG["data"]["image_dir"]),
        Path(BASE_CONFIG["data"]["characteristics"]),
        Path(USFM_CHECKPOINT),
        Path(USFM_ADAPTER),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    torch.backends.cudnn.benchmark = True
    print(f"cuda_available=True gpu0={torch.cuda.get_device_name(0)}")


def build_subject_manifest(config: dict[str, Any], seed: int) -> pd.DataFrame:
    manifest = build_manifest(
        image_dir=config["data"]["image_dir"],
        characteristics=config["data"]["characteristics"],
        sheet_name=config["data"]["sheet_name"],
        bins=config["data"]["bins"],
        class_names=config["data"]["class_names"],
        split=config["data"]["split"],
        seed=seed,
    )
    split_counts = manifest.groupby("subject_id")["split"].nunique()
    leaked = split_counts[split_counts > 1]
    if not leaked.empty:
        raise RuntimeError(f"Subject leakage across splits: {leaked.index.tolist()[:10]}")
    return manifest


class UltrasoundAgeRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, manifest: pd.DataFrame, transform=None):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        age = torch.tensor(float(row["age"]), dtype=torch.float32)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        subject_id = int(row["subject_id"])
        return image, age, label, subject_id


class SubjectImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        transform=None,
        k_images: int = 3,
        train: bool = True,
        eval_uses_all_images: bool = True,
    ):
        self.transform = transform
        self.k_images = int(k_images)
        self.train = train
        self.eval_uses_all_images = eval_uses_all_images
        subjects = []
        for subject_id, group in manifest.sort_values(["subject_id", "view"]).groupby("subject_id", sort=True):
            subjects.append(
                {
                    "subject_id": int(subject_id),
                    "age": float(group["age"].iloc[0]),
                    "label": int(group["label"].iloc[0]),
                    "image_paths": group["image_path"].tolist(),
                }
            )
        self.subjects = subjects

    def __len__(self) -> int:
        return len(self.subjects)

    def _select_paths(self, image_paths: list[str]) -> list[str]:
        if self.train:
            indices = np.random.choice(len(image_paths), size=self.k_images, replace=len(image_paths) < self.k_images)
            return [image_paths[int(idx)] for idx in indices]
        if self.eval_uses_all_images:
            return image_paths
        if len(image_paths) >= self.k_images:
            return image_paths[: self.k_images]
        repeats = [image_paths[idx % len(image_paths)] for idx in range(self.k_images)]
        return repeats

    def __getitem__(self, idx: int):
        item = self.subjects[idx]
        selected_paths = self._select_paths(item["image_paths"])
        images = []
        for image_path in selected_paths:
            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)
        return (
            torch.stack(images, dim=0),
            torch.ones(len(images), dtype=torch.bool),
            torch.tensor(int(item["label"]), dtype=torch.long),
            torch.tensor(int(item["subject_id"]), dtype=torch.long),
            torch.tensor(float(item["age"]), dtype=torch.float32),
            torch.tensor(len(item["image_paths"]), dtype=torch.long),
        )


def collate_subject_batch(batch):
    max_images = max(int(item[0].shape[0]) for item in batch)
    batch_size = len(batch)
    channels, height, width = batch[0][0].shape[1:]
    images = batch[0][0].new_zeros((batch_size, max_images, channels, height, width))
    masks = torch.zeros((batch_size, max_images), dtype=torch.bool)
    labels = torch.empty(batch_size, dtype=torch.long)
    subject_ids = torch.empty(batch_size, dtype=torch.long)
    ages = torch.empty(batch_size, dtype=torch.float32)
    image_counts = torch.empty(batch_size, dtype=torch.long)
    for idx, (item_images, item_mask, label, subject_id, age, image_count) in enumerate(batch):
        num_images = int(item_images.shape[0])
        images[idx, :num_images] = item_images
        masks[idx, :num_images] = item_mask
        labels[idx] = label
        subject_ids[idx] = subject_id
        ages[idx] = age
        image_counts[idx] = image_count
    return images, masks, labels, subject_ids, ages, image_counts


def exact_unfreeze_last_block(model: USFMClassifier) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = False
    encoder = getattr(model.backbone, "encoder", None)
    blocks = getattr(encoder, "blocks", None)
    if blocks is None or len(blocks) < 12:
        raise RuntimeError("Cannot safely locate encoder.blocks.11 for partial fine-tune.")
    trainable_modules: list[str] = []
    for param in blocks[11].parameters():
        param.requires_grad = True
    trainable_modules.append("encoder.blocks.11")
    fc_norm = getattr(encoder, "fc_norm", None)
    if fc_norm is None:
        raise RuntimeError("Cannot safely locate encoder.fc_norm for partial fine-tune.")
    trainable = False
    for param in fc_norm.parameters():
        param.requires_grad = True
        trainable = True
    if trainable:
        trainable_modules.append("encoder.fc_norm")
    model.trainable_backbone_modules = trainable_modules


def compute_subject_class_weights(train_manifest: pd.DataFrame, num_classes: int, device: torch.device) -> torch.Tensor:
    subject_labels = train_manifest.drop_duplicates("subject_id")["label"]
    counts = subject_labels.value_counts().reindex(range(num_classes), fill_value=0).to_numpy()
    if (counts == 0).any():
        raise ValueError(f"At least one class is missing from subject train split: {counts.tolist()}")
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


class USFMSubjectPoolingClassifier(nn.Module):
    def __init__(
        self,
        checkpoint_path: str,
        adapter_path: str,
        image_size: int,
        global_pool: str,
        num_classes: int,
        head_type: str,
        dropout: float,
        pooling: str,
    ) -> None:
        super().__init__()
        base = USFMClassifier(
            checkpoint_path=checkpoint_path,
            adapter_path=adapter_path,
            image_size=image_size,
            global_pool=global_pool,
            num_classes=num_classes,
            head_type=head_type,
            freeze_backbone=False,
            unfreeze_last_n_blocks=0,
            dropout=dropout,
        )
        self.backbone = base.backbone
        self.head = base.head
        self.feature_dim = base.feature_dim
        self.pooling = pooling
        self.trainable_backbone_modules: list[str] = []
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(self.feature_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )
        elif pooling != "mean":
            raise ValueError(f"Unsupported subject pooling: {pooling}")

    def pool_features(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(dtype=features.dtype, device=features.device)
        if self.pooling == "mean":
            denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
            return (features * mask_f.unsqueeze(-1)).sum(dim=1) / denom
        scores = self.attention(features).squeeze(-1)
        scores = scores.masked_fill(~mask.to(device=scores.device), torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return (features * weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        if x.ndim != 5:
            raise RuntimeError(f"Expected subject batch shape (B,K,C,H,W), got {tuple(x.shape)}")
        batch_size, k_images, channels, height, width = x.shape
        if mask is None:
            mask = torch.ones(batch_size, k_images, dtype=torch.bool, device=x.device)
        flat = x.reshape(batch_size * k_images, channels, height, width)
        features = self.backbone(flat).reshape(batch_size, k_images, self.feature_dim)
        pooled = self.pool_features(features, mask)
        return self.head(pooled)


def build_model(config: dict[str, Any], device: torch.device) -> USFMClassifier | USFMSubjectPoolingClassifier:
    if str(config["model"].get("task", "")) == "subject_pool":
        model = USFMSubjectPoolingClassifier(
            checkpoint_path=config["model"]["checkpoint_path"],
            adapter_path=config["model"]["adapter_path"],
            image_size=int(config["model"]["image_size"]),
            global_pool=str(config["model"]["global_pool"]),
            num_classes=int(config["model"]["num_outputs"]),
            head_type=str(config["model"]["head_type"]),
            dropout=float(config["model"]["dropout"]),
            pooling=str(config["model"]["pooling"]),
        ).to(device)
        exact_unfreeze_last_block(model)  # type: ignore[arg-type]
        return model

    model = USFMClassifier(
        checkpoint_path=config["model"]["checkpoint_path"],
        adapter_path=config["model"]["adapter_path"],
        image_size=int(config["model"]["image_size"]),
        global_pool=str(config["model"]["global_pool"]),
        num_classes=int(config["model"]["num_outputs"]),
        head_type=str(config["model"]["head_type"]),
        freeze_backbone=False,
        unfreeze_last_n_blocks=0,
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    exact_unfreeze_last_block(model)
    return model


def make_optimizer(model: USFMClassifier | USFMSubjectPoolingClassifier, config: dict[str, Any]) -> torch.optim.Optimizer:
    head_params = [param for param in model.head.parameters() if param.requires_grad]
    attention = getattr(model, "attention", None)
    if attention is not None:
        head_params.extend([param for param in attention.parameters() if param.requires_grad])
    backbone_params = [param for param in model.backbone.parameters() if param.requires_grad]
    if not backbone_params:
        raise RuntimeError("No trainable USFM backbone parameters found.")
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": float(config["train"]["backbone_lr"]), "name": "backbone"},
            {"params": head_params, "lr": float(config["train"]["head_lr"]), "name": "head"},
        ],
        weight_decay=float(config["train"]["weight_decay"]),
    )


class FocalLoss(nn.Module):
    def __init__(self, gamma: float, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else torch.empty(0))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(log_probs, labels, reduction="none")
        pt = torch.exp(-ce)
        loss = (1.0 - pt).pow(self.gamma) * ce
        if self.alpha.numel() > 0:
            loss = self.alpha[labels] * loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else torch.empty(0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        loss = -targets * log_probs
        if self.weight.numel() > 0:
            loss = loss * self.weight.view(1, -1)
        return loss.sum(dim=1).mean()


def autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def apply_midclass_multiplier(weights: torch.Tensor | None, config: dict[str, Any]) -> torch.Tensor | None:
    multiplier = float(config["train"].get("midclass_weight_multiplier", 1.0))
    if weights is not None and multiplier != 1.0:
        weights = weights.clone()
        weights[1] *= multiplier
    return weights


def age_to_bins(ages: np.ndarray) -> np.ndarray:
    preds = np.zeros(len(ages), dtype=int)
    preds[ages >= 45.0] = 1
    preds[ages >= 65.0] = 2
    return preds


def ldl_boundary_targets(ages: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    targets = F.one_hot(labels, num_classes=num_classes).float()
    age_years = torch.round(ages).long()
    rules = {
        42: [0.90, 0.10, 0.00],
        43: [0.80, 0.20, 0.00],
        44: [0.65, 0.35, 0.00],
        45: [0.35, 0.65, 0.00],
        46: [0.20, 0.80, 0.00],
        47: [0.10, 0.90, 0.00],
        62: [0.00, 0.90, 0.10],
        63: [0.00, 0.80, 0.20],
        64: [0.00, 0.65, 0.35],
        65: [0.00, 0.35, 0.65],
        66: [0.00, 0.20, 0.80],
        67: [0.00, 0.10, 0.90],
    }
    for age_year, values in rules.items():
        mask = age_years == age_year
        if bool(mask.any()):
            targets[mask] = torch.tensor(values, dtype=targets.dtype, device=targets.device)
    return targets


def ldl_gaussian_targets(ages: torch.Tensor, sigma: float, device: torch.device) -> torch.Tensor:
    axis = torch.arange(18, 101, dtype=torch.float32, device=device).view(1, -1)
    weights = torch.exp(-0.5 * ((axis - ages.float().view(-1, 1)) / float(sigma)).pow(2))
    bin0 = weights[:, :27].sum(dim=1)
    bin1 = weights[:, 27:47].sum(dim=1)
    bin2 = weights[:, 47:].sum(dim=1)
    targets = torch.stack([bin0, bin1, bin2], dim=1)
    return targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)


def make_ldl_targets(ages: torch.Tensor, labels: torch.Tensor, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    strategy = str(config.get("label_strategy", ""))
    num_classes = len(config["data"]["class_names"])
    if strategy == "ldl_boundary_w3" or strategy == "ldl_multitask_w3_lam0p3":
        return ldl_boundary_targets(ages.to(device), labels.to(device), num_classes)
    if strategy == "ldl_gaussian_sigma3":
        return ldl_gaussian_targets(ages.to(device), float(config["train"]["ldl_sigma"]), device)
    raise ValueError(f"Unsupported LDL label strategy: {strategy}")


def pearsonr_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or float(np.std(y_true)) == 0.0 or float(np.std(y_pred)) == 0.0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_pred.astype(float) - y_true.astype(float)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "pearson": pearsonr_np(y_true.astype(float), y_pred.astype(float)),
    }


def checkpoint_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name.startswith("head.")
        or name.startswith("attention.")
        or name.startswith("backbone.encoder.blocks.11.")
        or name.startswith("backbone.encoder.fc_norm.")
    }


def load_checkpoint_state(model: nn.Module, checkpoint: dict[str, Any]) -> None:
    state = checkpoint["model_state"]
    if checkpoint.get("checkpoint_state_type") == "trainable_state_dict":
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading best checkpoint: {unexpected}")
    else:
        model.load_state_dict(state)


def best_is_better(metric_name: str, value: float, best_value: float) -> bool:
    if metric_name in {"val_mae", "val_rmse", "val_loss"}:
        return value < best_value
    return value > best_value


def ordinal_targets(labels: torch.Tensor) -> torch.Tensor:
    return torch.stack([(labels >= 1).float(), (labels >= 2).float()], dim=1)


def ordinal_pos_weight(train_manifest: pd.DataFrame, device: torch.device) -> torch.Tensor:
    labels = train_manifest["label"].to_numpy()
    targets = np.stack([(labels >= 1).astype(np.float32), (labels >= 2).astype(np.float32)], axis=1)
    pos = targets.sum(axis=0)
    neg = targets.shape[0] - pos
    weights = neg / np.maximum(pos, 1.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def ordinal_probs_and_preds(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray, float]:
    probs = torch.sigmoid(logits.float()).cpu().numpy()
    invalid = probs[:, 1] > probs[:, 0]
    invalid_ratio = float(invalid.mean()) if len(probs) else 0.0
    probs[:, 1] = np.minimum(probs[:, 1], probs[:, 0])
    preds = np.zeros(probs.shape[0], dtype=int)
    preds[probs[:, 0] >= 0.5] = 1
    preds[probs[:, 1] >= 0.5] = 2
    return probs, preds, invalid_ratio


def ordinal_class_probs(threshold_probs: np.ndarray) -> np.ndarray:
    p45 = threshold_probs[:, 0]
    p65 = np.minimum(threshold_probs[:, 1], p45)
    return np.stack([1.0 - p45, p45 - p65, p65], axis=1)


def subject_level_predictions_ordinal(
    threshold_probs: np.ndarray,
    y_true: np.ndarray,
    subject_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.DataFrame({"subject_id": subject_ids, "label": y_true, "p45": threshold_probs[:, 0], "p65": threshold_probs[:, 1]})
    grouped = df.groupby("subject_id", sort=True)
    subject_true = grouped["label"].first().to_numpy(dtype=int)
    subject_probs = grouped[["p45", "p65"]].mean().to_numpy(dtype=float)
    subject_probs[:, 1] = np.minimum(subject_probs[:, 1], subject_probs[:, 0])
    subject_pred = np.zeros(subject_probs.shape[0], dtype=int)
    subject_pred[subject_probs[:, 0] >= 0.5] = 1
    subject_pred[subject_probs[:, 1] >= 0.5] = 2
    subject_ids_out = grouped["label"].first().index.to_numpy(dtype=int)
    return subject_true, subject_pred, subject_ids_out


def make_dataset_loader(
    manifest: pd.DataFrame,
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    config: dict[str, Any],
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": bool(config["train"]["pin_memory"]) and torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(config["train"]["persistent_workers"])
        kwargs["prefetch_factor"] = int(config["train"]["prefetch_factor"])
    task = str(config["model"].get("task", "multiclass"))
    if task == "subject_pool":
        dataset = SubjectImageDataset(
            manifest,
            transform,
            k_images=int(config["model"]["k_images"]),
            train=shuffle,
            eval_uses_all_images=bool(config["model"]["eval_uses_all_images"]),
        )
        return DataLoader(dataset, collate_fn=collate_subject_batch, **kwargs)
    age_target_tasks = {"regression_binning", "ldl_boundary", "ldl_gaussian", "ldl_multitask"}
    dataset = UltrasoundAgeRegressionDataset(manifest, transform) if task in age_target_tasks else UltrasoundAgeDataset(manifest, transform)
    return DataLoader(dataset, **kwargs)


def train_ordinal_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    amp_enabled: bool,
    grad_accum_steps: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for step, (images, labels, _subject_ids) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = ordinal_targets(labels)
        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
            loss_for_backward = loss / grad_accum_steps
        check_finite(float(loss.item()), "train_loss")
        scaler.scale(loss_for_backward).backward()
        if step % grad_accum_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        probs, preds, _invalid = ordinal_probs_and_preds(logits.detach())
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == labels.cpu().numpy()).sum())
        total_items += batch_size
        progress.set_postfix(loss=total_loss / total_items, acc=total_correct / total_items)
    return {"loss": total_loss / total_items, "accuracy": total_correct / total_items}


@torch.no_grad()
def predict_ordinal(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, amp_enabled: bool) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    invalid_weighted = 0.0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_subject_ids: list[np.ndarray] = []
    for images, labels, subject_ids in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = ordinal_targets(labels)
        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        check_finite(float(loss.item()), "eval_loss")
        threshold_probs, preds, invalid_ratio = ordinal_probs_and_preds(logits)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        invalid_weighted += invalid_ratio * batch_size
        all_probs.append(threshold_probs)
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
        all_subject_ids.append(subject_ids.numpy())
    probs_np = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)
    preds_np = np.concatenate(all_preds)
    subject_ids_np = np.concatenate(all_subject_ids)
    return {
        "loss": total_loss / total_items,
        "threshold_probs": probs_np,
        "class_probs": ordinal_class_probs(probs_np),
        "labels": labels_np,
        "preds": preds_np,
        "subject_ids": subject_ids_np,
        "invalid_order_ratio": invalid_weighted / total_items,
    }


@torch.no_grad()
def predict_multiclass(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, amp_enabled: bool) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_subject_ids: list[np.ndarray] = []
    for images, labels, subject_ids in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        check_finite(float(loss.item()), "eval_loss")
        probs = torch.softmax(logits.float(), dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_subject_ids.append(subject_ids.numpy())
    probs_np = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)
    subject_ids_np = np.concatenate(all_subject_ids)
    preds_np = probs_np.argmax(axis=1)
    return {"loss": total_loss / total_items, "probs": probs_np, "labels": labels_np, "preds": preds_np, "subject_ids": subject_ids_np}


def train_subject_pool_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    amp_enabled: bool,
    grad_accum_steps: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for step, (images, masks, labels, _subject_ids, _ages, _image_counts) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            logits = model(images, masks)
            loss = criterion(logits, labels)
            loss_for_backward = loss / grad_accum_steps
        check_finite(float(loss.item()), "train_loss")
        scaler.scale(loss_for_backward).backward()
        if step % grad_accum_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.detach().argmax(dim=1) == labels).sum().item())
        total_items += batch_size
        progress.set_postfix(loss=total_loss / total_items, acc=total_correct / total_items)
    return {"loss": total_loss / total_items, "accuracy": total_correct / total_items}


@torch.no_grad()
def predict_subject_pool(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, amp_enabled: bool) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_subject_ids: list[np.ndarray] = []
    all_ages: list[np.ndarray] = []
    all_image_counts: list[np.ndarray] = []
    for images, masks, labels, subject_ids, ages, image_counts in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            logits = model(images, masks)
            loss = criterion(logits, labels_device)
        check_finite(float(loss.item()), "eval_loss")
        probs = torch.softmax(logits.float(), dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_subject_ids.append(subject_ids.numpy())
        all_ages.append(ages.numpy())
        all_image_counts.append(image_counts.numpy())
    probs_np = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)
    subject_ids_np = np.concatenate(all_subject_ids)
    ages_np = np.concatenate(all_ages)
    image_counts_np = np.concatenate(all_image_counts)
    preds_np = probs_np.argmax(axis=1)
    return {
        "loss": total_loss / total_items,
        "probs": probs_np,
        "labels": labels_np,
        "preds": preds_np,
        "subject_ids": subject_ids_np,
        "ages": ages_np,
        "image_counts": image_counts_np,
    }


def save_subject_pool_predictions(
    output_dir: Path,
    probs: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    subject_ids: np.ndarray,
    ages: np.ndarray,
    image_counts: np.ndarray,
    class_names: list[str],
) -> None:
    subject_df = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "age": ages,
            "num_images": image_counts,
            "label": labels,
            "pred": preds,
        }
    )
    subject_df["label_name"] = [class_names[idx] for idx in labels]
    subject_df["pred_name"] = [class_names[idx] for idx in preds]
    for idx, name in enumerate(class_names):
        subject_df[f"prob_{name}"] = probs[:, idx]
    subject_df.to_csv(output_dir / "subject_test_predictions.csv", index=False)


def split_multitask_output(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if output.ndim != 2 or output.shape[1] != 4:
        raise RuntimeError(f"Expected multitask output shape (N, 4), got {tuple(output.shape)}")
    return output[:, :3], output[:, 3]


def train_ldl_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    amp_enabled: bool,
    grad_accum_steps: int,
    config: dict[str, Any],
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for step, (images, ages, labels, _subject_ids) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = make_ldl_targets(ages, labels, config, device)
        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
            loss_for_backward = loss / grad_accum_steps
        check_finite(float(loss.item()), "train_loss")
        scaler.scale(loss_for_backward).backward()
        if step % grad_accum_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        preds = logits.detach().argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == labels).sum().item())
        total_items += batch_size
        progress.set_postfix(loss=total_loss / total_items, acc=total_correct / total_items)
    return {"loss": total_loss / total_items, "accuracy": total_correct / total_items}


@torch.no_grad()
def predict_ldl(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_ages: list[np.ndarray] = []
    all_subject_ids: list[np.ndarray] = []
    for images, ages, labels, subject_ids in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        ages_device = ages.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        targets = make_ldl_targets(ages_device, labels_device, config, device)
        with autocast_context(device, amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        check_finite(float(loss.item()), "eval_loss")
        probs = torch.softmax(logits.float(), dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_ages.append(ages.numpy())
        all_subject_ids.append(subject_ids.numpy())
    probs_np = np.concatenate(all_probs)
    labels_np = np.concatenate(all_labels)
    ages_np = np.concatenate(all_ages)
    subject_ids_np = np.concatenate(all_subject_ids)
    preds_np = probs_np.argmax(axis=1)
    return {
        "loss": total_loss / total_items,
        "probs": probs_np,
        "labels": labels_np,
        "preds": preds_np,
        "true_age": ages_np,
        "subject_ids": subject_ids_np,
    }


def train_ldl_multitask_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    regression_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    amp_enabled: bool,
    grad_accum_steps: int,
    config: dict[str, Any],
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_ldl_loss = 0.0
    total_regression_loss = 0.0
    total_correct = 0
    total_items = 0
    lam = float(config["train"]["multitask_regression_lambda"])
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for step, (images, ages, labels, _subject_ids) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = make_ldl_targets(ages, labels, config, device)
        with autocast_context(device, amp_enabled):
            output = model(images)
            logits, pred_age = split_multitask_output(output)
            ldl_loss = criterion(logits, targets)
            regression_loss = regression_criterion(pred_age, ages)
            loss = ldl_loss + lam * regression_loss
            loss_for_backward = loss / grad_accum_steps
        check_finite(float(loss.item()), "train_loss")
        scaler.scale(loss_for_backward).backward()
        if step % grad_accum_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        preds = logits.detach().argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_ldl_loss += float(ldl_loss.item()) * batch_size
        total_regression_loss += float(regression_loss.item()) * batch_size
        total_correct += int((preds == labels).sum().item())
        total_items += batch_size
        progress.set_postfix(loss=total_loss / total_items, acc=total_correct / total_items)
    return {
        "loss": total_loss / total_items,
        "accuracy": total_correct / total_items,
        "ldl_loss": total_ldl_loss / total_items,
        "regression_loss": total_regression_loss / total_items,
    }


@torch.no_grad()
def predict_ldl_multitask(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    regression_criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    config: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_ldl_loss = 0.0
    total_regression_loss = 0.0
    total_items = 0
    lam = float(config["train"]["multitask_regression_lambda"])
    all_probs: list[np.ndarray] = []
    all_pred_age: list[np.ndarray] = []
    all_true_age: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_subject_ids: list[np.ndarray] = []
    for images, ages, labels, subject_ids in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        ages_device = ages.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        targets = make_ldl_targets(ages_device, labels_device, config, device)
        with autocast_context(device, amp_enabled):
            output = model(images)
            logits, pred_age = split_multitask_output(output)
            ldl_loss = criterion(logits, targets)
            regression_loss = regression_criterion(pred_age, ages_device)
            loss = ldl_loss + lam * regression_loss
        check_finite(float(loss.item()), "eval_loss")
        probs = torch.softmax(logits.float(), dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_ldl_loss += float(ldl_loss.item()) * batch_size
        total_regression_loss += float(regression_loss.item()) * batch_size
        total_items += batch_size
        all_probs.append(probs.cpu().numpy())
        all_pred_age.append(pred_age.float().cpu().numpy())
        all_true_age.append(ages.numpy())
        all_labels.append(labels.numpy())
        all_subject_ids.append(subject_ids.numpy())
    probs_np = np.concatenate(all_probs)
    pred_age_np = np.concatenate(all_pred_age)
    true_age_np = np.concatenate(all_true_age)
    labels_np = np.concatenate(all_labels)
    subject_ids_np = np.concatenate(all_subject_ids)
    preds_np = probs_np.argmax(axis=1)
    return {
        "loss": total_loss / total_items,
        "ldl_loss": total_ldl_loss / total_items,
        "regression_loss": total_regression_loss / total_items,
        "probs": probs_np,
        "pred_age": pred_age_np,
        "true_age": true_age_np,
        "labels": labels_np,
        "preds": preds_np,
        "subject_ids": subject_ids_np,
    }


def train_regression_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    amp_enabled: bool,
    grad_accum_steps: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for step, (images, ages, labels, _subject_ids) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True)
        labels_np = labels.numpy()
        with autocast_context(device, amp_enabled):
            pred_age = model(images).squeeze(1)
            loss = criterion(pred_age, ages)
            loss_for_backward = loss / grad_accum_steps
        check_finite(float(loss.item()), "train_loss")
        scaler.scale(loss_for_backward).backward()
        if step % grad_accum_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        preds_np = age_to_bins(pred_age.detach().float().cpu().numpy())
        batch_size = ages.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds_np == labels_np).sum())
        total_items += batch_size
        progress.set_postfix(loss=total_loss / total_items, acc=total_correct / total_items)
    return {"loss": total_loss / total_items, "accuracy": total_correct / total_items}


@torch.no_grad()
def predict_regression(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, amp_enabled: bool) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_pred_age: list[np.ndarray] = []
    all_true_age: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_subject_ids: list[np.ndarray] = []
    for images, ages, labels, subject_ids in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        ages = ages.to(device, non_blocking=True)
        with autocast_context(device, amp_enabled):
            pred_age = model(images).squeeze(1)
            loss = criterion(pred_age, ages)
        check_finite(float(loss.item()), "eval_loss")
        batch_size = ages.size(0)
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        all_pred_age.append(pred_age.float().cpu().numpy())
        all_true_age.append(ages.float().cpu().numpy())
        all_labels.append(labels.numpy())
        all_subject_ids.append(subject_ids.numpy())
    pred_age_np = np.concatenate(all_pred_age)
    true_age_np = np.concatenate(all_true_age)
    labels_np = np.concatenate(all_labels)
    subject_ids_np = np.concatenate(all_subject_ids)
    preds_np = age_to_bins(pred_age_np)
    return {
        "loss": total_loss / total_items,
        "pred_age": pred_age_np,
        "true_age": true_age_np,
        "labels": labels_np,
        "preds": preds_np,
        "subject_ids": subject_ids_np,
    }


def subject_level_regression(
    pred_age: np.ndarray,
    true_age: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.DataFrame({"subject_id": subject_ids, "pred_age": pred_age, "true_age": true_age, "label": labels})
    grouped = df.groupby("subject_id", sort=True)
    subject_ids_out = grouped["label"].first().index.to_numpy(dtype=int)
    subject_pred_age = grouped["pred_age"].mean().to_numpy(dtype=float)
    subject_true_age = grouped["true_age"].first().to_numpy(dtype=float)
    subject_true_label = grouped["label"].first().to_numpy(dtype=int)
    subject_pred_label = age_to_bins(subject_pred_age)
    return subject_true_age, subject_pred_age, subject_true_label, subject_pred_label, subject_ids_out


def subject_level_predictions_with_age(
    probs: np.ndarray,
    labels: np.ndarray,
    ages: np.ndarray,
    subject_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prob_cols = [f"p_{i}" for i in range(probs.shape[1])]
    df = pd.DataFrame({"subject_id": subject_ids, "label": labels, "age": ages})
    for idx, col in enumerate(prob_cols):
        df[col] = probs[:, idx]
    grouped = df.groupby("subject_id", sort=True)
    subject_ids_out = grouped["label"].first().index.to_numpy(dtype=int)
    subject_true = grouped["label"].first().to_numpy(dtype=int)
    subject_age = grouped["age"].first().to_numpy(dtype=float)
    subject_probs = grouped[prob_cols].mean().to_numpy(dtype=float)
    subject_pred = subject_probs.argmax(axis=1)
    return subject_true, subject_pred, subject_ids_out, subject_age


def miss_rates(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    if len(y_true) == 0:
        return math.nan, math.nan
    diff = np.abs(y_true.astype(int) - y_pred.astype(int))
    near = float(np.mean(diff == 1))
    far = float(np.mean(diff == 2))
    return near, far


def boundary_group_metrics(
    subject_true: np.ndarray,
    subject_pred: np.ndarray,
    subject_age: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, float]]:
    masks = {
        "boundary_45": (subject_age >= 42.0) & (subject_age <= 47.0),
        "boundary_65": (subject_age >= 62.0) & (subject_age <= 67.0),
    }
    masks["non_boundary"] = ~(masks["boundary_45"] | masks["boundary_65"])
    results: dict[str, dict[str, float]] = {}
    for name, mask in masks.items():
        y_true = subject_true[mask]
        y_pred = subject_pred[mask]
        near_rate, far_rate = miss_rates(y_true, y_pred)
        if len(y_true) == 0:
            results[name] = {
                "n_subjects": 0,
                "accuracy": math.nan,
                "macro_f1": math.nan,
                "near_miss_rate": math.nan,
                "far_miss_rate": math.nan,
            }
            continue
        accuracy = float(np.mean(y_true == y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(np.unique(y_true)) >= 2 else math.nan
        results[name] = {
            "n_subjects": int(len(y_true)),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "near_miss_rate": near_rate,
            "far_miss_rate": far_rate,
        }
    return results


def flatten_boundary_metrics(boundary_info: dict[str, Any]) -> dict[str, float]:
    row: dict[str, float] = {}
    for group in ("boundary_45", "boundary_65", "non_boundary"):
        metrics = boundary_info.get(group, {})
        row[f"{group}_n_subjects"] = metrics.get("n_subjects", math.nan)
        row[f"{group}_accuracy"] = metrics.get("accuracy", math.nan)
        row[f"{group}_macro_f1"] = metrics.get("macro_f1", math.nan)
        row[f"{group}_near_miss_rate"] = metrics.get("near_miss_rate", math.nan)
        row[f"{group}_far_miss_rate"] = metrics.get("far_miss_rate", math.nan)
    return row


def save_regression_predictions(
    output_dir: Path,
    test_manifest: pd.DataFrame,
    pred_age: np.ndarray,
    true_age: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    subject_ids: np.ndarray,
    class_names: list[str],
) -> None:
    image_df = test_manifest.reset_index(drop=True).copy()
    image_df["true_age"] = true_age
    image_df["pred_age"] = pred_age
    image_df["label"] = labels
    image_df["pred"] = preds
    image_df["label_name"] = [class_names[idx] for idx in labels]
    image_df["pred_name"] = [class_names[idx] for idx in preds]
    image_df.to_csv(output_dir / "image_regression_predictions.csv", index=False)
    image_df.to_csv(output_dir / "image_test_predictions.csv", index=False)

    subject_true_age, subject_pred_age, subject_true_label, subject_pred_label, subject_ids_out = subject_level_regression(
        pred_age, true_age, labels, subject_ids
    )
    subject_df = pd.DataFrame(
        {
            "subject_id": subject_ids_out,
            "true_age": subject_true_age,
            "pred_age": subject_pred_age,
            "label": subject_true_label,
            "pred": subject_pred_label,
        }
    )
    subject_df["label_name"] = [class_names[idx] for idx in subject_true_label]
    subject_df["pred_name"] = [class_names[idx] for idx in subject_pred_label]
    subject_df.to_csv(output_dir / "subject_regression_predictions.csv", index=False)
    subject_df.to_csv(output_dir / "subject_test_predictions.csv", index=False)


def save_aux_regression_predictions(
    output_dir: Path,
    test_manifest: pd.DataFrame,
    pred_age: np.ndarray,
    true_age: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    class_names: list[str],
) -> None:
    image_df = test_manifest.reset_index(drop=True).copy()
    image_df["true_age"] = true_age
    image_df["pred_age"] = pred_age
    image_df["label"] = labels
    image_df["label_name"] = [class_names[idx] for idx in labels]
    image_df.to_csv(output_dir / "image_regression_predictions.csv", index=False)

    subject_true_age, subject_pred_age, subject_true_label, _subject_pred_label, subject_ids_out = subject_level_regression(
        pred_age, true_age, labels, subject_ids
    )
    subject_df = pd.DataFrame(
        {
            "subject_id": subject_ids_out,
            "true_age": subject_true_age,
            "pred_age": subject_pred_age,
            "label": subject_true_label,
        }
    )
    subject_df["label_name"] = [class_names[idx] for idx in subject_true_label]
    subject_df.to_csv(output_dir / "subject_regression_predictions.csv", index=False)


def plot_regression_figures(output_dir: Path, true_age: np.ndarray, pred_age: np.ndarray, labels: np.ndarray, class_names: list[str]) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.scatter(true_age, pred_age, s=12, alpha=0.65)
    lo = float(min(true_age.min(), pred_age.min()))
    hi = float(max(true_age.max(), pred_age.max()))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("True age")
    ax.set_ylabel("Predicted age")
    ax.set_title("Predicted Age vs True Age")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "pred_age_vs_true_age.png")
    plt.close(fig)

    errors = pred_age - true_age
    data = [errors[labels == idx] for idx in range(len(class_names))]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    ax.boxplot(data, labels=class_names, showfliers=False)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Age bin")
    ax.set_ylabel("Prediction error")
    ax.set_title("Regression Error by Age Bin")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "regression_error_by_age_bin.png")
    plt.close(fig)


def save_ordinal_predictions(
    output_dir: Path,
    test_manifest: pd.DataFrame,
    threshold_probs: np.ndarray,
    class_probs: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    subject_ids: np.ndarray,
    class_names: list[str],
) -> None:
    image_df = test_manifest.reset_index(drop=True).copy()
    image_df["label"] = labels
    image_df["pred"] = preds
    image_df["label_name"] = [class_names[idx] for idx in labels]
    image_df["pred_name"] = [class_names[idx] for idx in preds]
    image_df["prob_age_ge_45"] = threshold_probs[:, 0]
    image_df["prob_age_ge_65"] = threshold_probs[:, 1]
    for idx, name in enumerate(class_names):
        image_df[f"prob_{name}"] = class_probs[:, idx]
    image_df.to_csv(output_dir / "image_test_predictions.csv", index=False)

    subject_true, subject_pred, subject_ids_out = subject_level_predictions_ordinal(threshold_probs, labels, subject_ids)
    subject_df = pd.DataFrame({"subject_id": subject_ids_out, "label": subject_true, "pred": subject_pred})
    subject_df["label_name"] = [class_names[idx] for idx in subject_true]
    subject_df["pred_name"] = [class_names[idx] for idx in subject_pred]
    probs_df = pd.DataFrame({"subject_id": subject_ids, "prob_age_ge_45": threshold_probs[:, 0], "prob_age_ge_65": threshold_probs[:, 1]})
    class_probs_df = pd.DataFrame({"subject_id": subject_ids})
    for idx, name in enumerate(class_names):
        class_probs_df[f"prob_{name}"] = class_probs[:, idx]
    subject_df = subject_df.merge(probs_df.groupby("subject_id", sort=True).mean().reset_index(), on="subject_id", how="left")
    subject_df = subject_df.merge(class_probs_df.groupby("subject_id", sort=True).mean().reset_index(), on="subject_id", how="left")
    subject_df.to_csv(output_dir / "subject_test_predictions.csv", index=False)


def class_subject_metrics(subject_report: dict[str, Any], class_names: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in class_names:
        key = name.replace("-", "_")
        values[f"subject_f1_{key}"] = float(subject_report[name]["f1-score"])
        values[f"subject_recall_{key}"] = float(subject_report[name]["recall"])
    return values


def append_result_notes(seed_dir: Path, lines: list[str]) -> None:
    result = seed_dir / "result.md"
    if not result.exists():
        return
    with result.open("a", encoding="utf-8") as f:
        f.write("\n## Notes\n\n")
        for line in lines:
            f.write(f"- {line}\n")


def run_seed(config: dict[str, Any], seed: int, seed_dir: Path, batch_size: int, num_workers: int) -> dict[str, Any]:
    set_seed(seed)
    torch.backends.cudnn.benchmark = True
    device = resolve_device(config["train"]["device"])
    config = deepcopy(config)
    config["train"]["seed"] = seed
    config["train"]["actual_batch_size"] = batch_size
    config["train"]["effective_batch_size"] = batch_size * int(config["train"]["grad_accum_steps"])
    config["train"]["num_workers"] = num_workers
    config["output_dir"] = str(seed_dir)
    config["runtime"] = {"cuda_available": torch.cuda.is_available(), "gpu_name": gpu_name(device)}
    if str(config["model"].get("task", "")) == "subject_pool":
        config["training_unit"] = "subject"
        config["pooling"] = str(config["model"]["pooling"])
        config["k_images"] = int(config["model"]["k_images"])
        config["eval_uses_all_images"] = bool(config["model"]["eval_uses_all_images"])

    seed_dir.mkdir(parents=True, exist_ok=False)
    manifest = build_subject_manifest(config, seed)
    manifest.to_csv(seed_dir / "manifest.csv", index=False)
    split_summary(manifest).to_csv(seed_dir / "split_summary.csv", index=False)
    save_json(config, seed_dir / "config_used.json")
    save_json(config, seed_dir / "config.json")

    train_tf, eval_tf = make_transforms(int(config["model"]["image_size"]))
    train_manifest = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    test_manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)
    train_loader = make_dataset_loader(train_manifest, train_tf, batch_size, num_workers, True, config)
    val_loader = make_dataset_loader(val_manifest, eval_tf, batch_size, num_workers, False, config)
    test_loader = make_dataset_loader(test_manifest, eval_tf, batch_size, num_workers, False, config)

    model = build_model(config, device)
    smoke_forward(model, device, int(config["model"]["num_outputs"]))
    write_model_summary(model, config, seed_dir / "model_summary.txt")
    with (seed_dir / "model_summary.txt").open("a", encoding="utf-8") as f:
        total, trainable = count_parameters(model)
        f.write(f"task: {config['model']['task']}\n")
        f.write(f"label_strategy: {config.get('label_strategy', '')}\n")
        f.write(f"soft_label_width: {config['train'].get('soft_label_width', '')}\n")
        f.write(f"ldl_sigma: {config['train'].get('ldl_sigma', '')}\n")
        f.write(f"multitask_regression_lambda: {config['train'].get('multitask_regression_lambda', '')}\n")
        f.write(f"training_unit: {config.get('training_unit', config['train'].get('training_unit', 'image'))}\n")
        f.write(f"pooling: {config['model'].get('pooling', '')}\n")
        f.write(f"k_images: {config['model'].get('k_images', '')}\n")
        f.write(f"eval_uses_all_images: {config['model'].get('eval_uses_all_images', '')}\n")
        f.write(f"exact_trainable_backbone_modules: {model.trainable_backbone_modules}\n")
        f.write(f"total_params_exact: {total}\n")
        f.write(f"trainable_params_exact: {trainable}\n")

    task = str(config["model"]["task"])
    regression_criterion: nn.Module | None = None
    if task == "ordinal":
        criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=ordinal_pos_weight(train_manifest, device))
    elif task == "regression_binning":
        criterion = nn.SmoothL1Loss()
    elif task == "focal_loss":
        weights = compute_class_weights(train_manifest, len(config["data"]["class_names"]), device)
        criterion = FocalLoss(gamma=float(config["train"]["focal_gamma"]), alpha=weights)
    elif task in {"ldl_boundary", "ldl_gaussian"}:
        weights = compute_class_weights(train_manifest, len(config["data"]["class_names"]), device) if config["train"]["use_class_weights"] else None
        criterion = SoftTargetCrossEntropy(weight=weights)
    elif task == "ldl_multitask":
        weights = compute_class_weights(train_manifest, len(config["data"]["class_names"]), device) if config["train"]["use_class_weights"] else None
        criterion = SoftTargetCrossEntropy(weight=weights)
        regression_criterion = nn.SmoothL1Loss()
    elif task == "subject_pool":
        weights = compute_subject_class_weights(train_manifest, len(config["data"]["class_names"]), device) if config["train"]["use_class_weights"] else None
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        weights = compute_class_weights(train_manifest, len(config["data"]["class_names"]), device) if config["train"]["use_class_weights"] else None
        weights = apply_midclass_multiplier(weights, config)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=float(config["train"]["label_smoothing"]))

    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["train"]["epochs"]))
    amp_enabled = bool(config["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
    save_best_by = str(config["train"]["save_best_by"])
    best_value = math.inf if save_best_by in {"val_mae", "val_rmse", "val_loss"} else -math.inf
    best_val_macro_f1 = -1.0
    best_val_mae = math.nan
    best_epoch = -1
    stale_epochs = 0
    stop_reason = "max_epochs"
    logs: list[dict[str, float | int | str]] = []
    class_names = config["data"]["class_names"]
    start = time.perf_counter()
    invalid_val_ratio = 0.0

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        if task == "ordinal":
            train_stats = train_ordinal_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                epoch,
                amp_enabled,
                int(config["train"]["grad_accum_steps"]),
            )
            val_pred = predict_ordinal(model, val_loader, criterion, device, amp_enabled)
            invalid_val_ratio = float(val_pred["invalid_order_ratio"])
        elif task == "regression_binning":
            train_stats = train_regression_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                epoch,
                amp_enabled,
                int(config["train"]["grad_accum_steps"]),
            )
            val_pred = predict_regression(model, val_loader, criterion, device, amp_enabled)
        elif task in {"ldl_boundary", "ldl_gaussian"}:
            train_stats = train_ldl_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                epoch,
                amp_enabled,
                int(config["train"]["grad_accum_steps"]),
                config,
            )
            val_pred = predict_ldl(model, val_loader, criterion, device, amp_enabled, config)
        elif task == "ldl_multitask":
            if regression_criterion is None:
                raise RuntimeError("regression_criterion is required for LDL multitask.")
            train_stats = train_ldl_multitask_epoch(
                model,
                train_loader,
                criterion,
                regression_criterion,
                optimizer,
                scaler,
                device,
                epoch,
                amp_enabled,
                int(config["train"]["grad_accum_steps"]),
                config,
            )
            val_pred = predict_ldl_multitask(model, val_loader, criterion, regression_criterion, device, amp_enabled, config)
        elif task == "subject_pool":
            train_stats = train_subject_pool_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                epoch,
                amp_enabled,
                int(config["train"]["grad_accum_steps"]),
            )
            val_pred = predict_subject_pool(model, val_loader, criterion, device, amp_enabled)
        else:
            train_stats = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                epoch,
                amp_enabled,
                int(config["train"]["grad_accum_steps"]),
            )
            val_pred = predict_multiclass(model, val_loader, criterion, device, amp_enabled)

        val_metrics = image_level_metrics(val_pred["labels"], val_pred["preds"], class_names)
        val_reg_metrics = regression_metrics(val_pred["true_age"], val_pred["pred_age"]) if task in {"regression_binning", "ldl_multitask"} else {}
        check_finite(float(train_stats["loss"]), "train_loss")
        check_finite(float(val_pred["loss"]), "val_loss")
        check_finite(float(val_metrics["macro_f1"]), "val_macro_f1")
        if val_reg_metrics:
            check_finite(float(val_reg_metrics["mae"]), "val_mae")
        scheduler.step()
        current_lrs = scheduler.get_last_lr()
        row = {
            "epoch": epoch,
            "lr": float(current_lrs[0]),
            "head_lr": float(current_lrs[-1]),
            "train_loss": float(train_stats["loss"]),
            "train_accuracy": float(train_stats["accuracy"]),
            "val_loss": float(val_pred["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_mae": float(val_reg_metrics.get("mae", math.nan)),
            "val_rmse": float(val_reg_metrics.get("rmse", math.nan)),
            "val_pearson": float(val_reg_metrics.get("pearson", math.nan)),
            "train_ldl_loss": float(train_stats.get("ldl_loss", math.nan)),
            "train_regression_loss": float(train_stats.get("regression_loss", math.nan)),
            "val_ldl_loss": float(val_pred.get("ldl_loss", math.nan)),
            "val_regression_loss": float(val_pred.get("regression_loss", math.nan)),
            "invalid_order_ratio": invalid_val_ratio if task == "ordinal" else 0.0,
        }
        logs.append(row)
        pd.DataFrame(logs).to_csv(seed_dir / "train_log.csv", index=False)
        if epoch % 5 == 0 or epoch == int(config["train"]["epochs"]):
            generate_report(seed_dir, epoch=epoch)
        metric_value = float(row[save_best_by])
        print(
            f"experiment={config['experiment_name']} seed={seed} epoch={epoch:03d} "
            f"val_macro_f1={row['val_macro_f1']:.4f} {save_best_by}={metric_value:.4f}"
        )

        if best_is_better(save_best_by, metric_value, best_value):
            best_value = metric_value
            best_val_macro_f1 = float(row["val_macro_f1"])
            best_val_mae = float(row["val_mae"]) if not math.isnan(float(row["val_mae"])) else math.nan
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": checkpoint_state_dict(model),
                    "checkpoint_state_type": str(config["train"]["checkpoint_state"]),
                    "config": config,
                    "best_value": best_value,
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_val_mae": best_val_mae,
                    "save_best_by": save_best_by,
                },
                seed_dir / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["train"]["early_stopping_patience"]):
                stop_reason = f"early_stopping_patience_{config['train']['early_stopping_patience']}"
                break

    elapsed = time.perf_counter() - start
    if best_epoch < 0:
        raise RuntimeError("No best checkpoint was saved.")
    checkpoint = torch.load(seed_dir / "best.pt", map_location=device, weights_only=False)
    load_checkpoint_state(model, checkpoint)
    boundary_info: dict[str, Any] = {}

    if task == "ordinal":
        test_pred = predict_ordinal(model, test_loader, criterion, device, amp_enabled)
        subject_true, subject_pred, _subject_ids = subject_level_predictions_ordinal(
            test_pred["threshold_probs"], test_pred["labels"], test_pred["subject_ids"]
        )
        save_ordinal_predictions(
            seed_dir,
            test_manifest,
            test_pred["threshold_probs"],
            test_pred["class_probs"],
            test_pred["labels"],
            test_pred["preds"],
            test_pred["subject_ids"],
            class_names,
        )
        ordinal_info = {"invalid_order_ratio": float(test_pred["invalid_order_ratio"]), "correction": "p65 clipped to p45 before thresholding"}
        regression_info = {}
    elif task == "regression_binning":
        test_pred = predict_regression(model, test_loader, criterion, device, amp_enabled)
        subject_true_age, subject_pred_age, subject_true, subject_pred, _subject_ids = subject_level_regression(
            test_pred["pred_age"], test_pred["true_age"], test_pred["labels"], test_pred["subject_ids"]
        )
        save_regression_predictions(
            seed_dir,
            test_manifest,
            test_pred["pred_age"],
            test_pred["true_age"],
            test_pred["labels"],
            test_pred["preds"],
            test_pred["subject_ids"],
            class_names,
        )
        plot_regression_figures(seed_dir, test_pred["true_age"], test_pred["pred_age"], test_pred["labels"], class_names)
        regression_info = {
            "image_level": regression_metrics(test_pred["true_age"], test_pred["pred_age"]),
            "subject_level": regression_metrics(subject_true_age, subject_pred_age),
        }
        ordinal_info = {}
    elif task in {"ldl_boundary", "ldl_gaussian"}:
        test_pred = predict_ldl(model, test_loader, criterion, device, amp_enabled, config)
        subject_true, subject_pred, _subject_ids, subject_age = subject_level_predictions_with_age(
            test_pred["probs"], test_pred["labels"], test_pred["true_age"], test_pred["subject_ids"]
        )
        save_predictions(
            seed_dir,
            test_manifest,
            test_pred["probs"],
            test_pred["labels"],
            test_pred["preds"],
            test_pred["subject_ids"],
            class_names,
        )
        boundary_info = boundary_group_metrics(subject_true, subject_pred, subject_age, class_names)
        ordinal_info = {}
        regression_info = {}
    elif task == "ldl_multitask":
        if regression_criterion is None:
            raise RuntimeError("regression_criterion is required for LDL multitask.")
        test_pred = predict_ldl_multitask(model, test_loader, criterion, regression_criterion, device, amp_enabled, config)
        subject_true, subject_pred, _subject_ids, subject_age = subject_level_predictions_with_age(
            test_pred["probs"], test_pred["labels"], test_pred["true_age"], test_pred["subject_ids"]
        )
        subject_true_age, subject_pred_age, _subject_true_label, _subject_pred_label, _reg_subject_ids = subject_level_regression(
            test_pred["pred_age"], test_pred["true_age"], test_pred["labels"], test_pred["subject_ids"]
        )
        save_predictions(
            seed_dir,
            test_manifest,
            test_pred["probs"],
            test_pred["labels"],
            test_pred["preds"],
            test_pred["subject_ids"],
            class_names,
        )
        save_aux_regression_predictions(
            seed_dir,
            test_manifest,
            test_pred["pred_age"],
            test_pred["true_age"],
            test_pred["labels"],
            test_pred["subject_ids"],
            class_names,
        )
        plot_regression_figures(seed_dir, test_pred["true_age"], test_pred["pred_age"], test_pred["labels"], class_names)
        boundary_info = boundary_group_metrics(subject_true, subject_pred, subject_age, class_names)
        regression_info = {
            "image_level": regression_metrics(test_pred["true_age"], test_pred["pred_age"]),
            "subject_level": regression_metrics(subject_true_age, subject_pred_age),
        }
        ordinal_info = {}
    elif task == "subject_pool":
        test_pred = predict_subject_pool(model, test_loader, criterion, device, amp_enabled)
        subject_true = test_pred["labels"]
        subject_pred = test_pred["preds"]
        _subject_ids = test_pred["subject_ids"]
        save_subject_pool_predictions(
            seed_dir,
            test_pred["probs"],
            test_pred["labels"],
            test_pred["preds"],
            test_pred["subject_ids"],
            test_pred["ages"],
            test_pred["image_counts"],
            class_names,
        )
        ordinal_info = {}
        regression_info = {}
    else:
        test_pred = predict_multiclass(model, test_loader, criterion, device, amp_enabled)
        subject_true, subject_pred, _subject_ids = subject_level_predictions(
            test_pred["probs"], test_pred["labels"], test_pred["subject_ids"]
        )
        save_predictions(
            seed_dir,
            test_manifest,
            test_pred["probs"],
            test_pred["labels"],
            test_pred["preds"],
            test_pred["subject_ids"],
            class_names,
        )
        ordinal_info = {}
        regression_info = {}

    test_image_metrics = image_level_metrics(test_pred["labels"], test_pred["preds"], class_names)
    test_subject_metrics = image_level_metrics(subject_true, subject_pred, class_names)
    metrics = {"image_level": test_image_metrics, "subject_level": test_subject_metrics}
    if ordinal_info:
        metrics["ordinal"] = ordinal_info
    if regression_info:
        metrics["regression"] = regression_info
    if boundary_info:
        metrics["boundary_subgroups"] = boundary_info
    save_json(metrics, seed_dir / "test_metrics.json")
    if regression_info:
        save_json(regression_info, seed_dir / "regression_test_metrics.json")
    report_to_csv(test_image_metrics["classification_report"], class_names, seed_dir / "image_classification_report.csv")
    report_to_csv(test_subject_metrics["classification_report"], class_names, seed_dir / "subject_classification_report.csv")
    config["runtime"]["train_elapsed_seconds"] = elapsed
    config["runtime"]["stop_reason"] = stop_reason
    config["train"]["epochs_ran"] = int(logs[-1]["epoch"])
    save_json(config, seed_dir / "config_used.json")
    save_json(config, seed_dir / "config.json")
    write_model_summary(model, config, seed_dir / "model_summary.txt", elapsed)
    with (seed_dir / "model_summary.txt").open("a", encoding="utf-8") as f:
        f.write(f"task: {task}\n")
        f.write(f"label_strategy: {config.get('label_strategy', '')}\n")
        f.write(f"soft_label_width: {config['train'].get('soft_label_width', '')}\n")
        f.write(f"ldl_sigma: {config['train'].get('ldl_sigma', '')}\n")
        f.write(f"multitask_regression_lambda: {config['train'].get('multitask_regression_lambda', '')}\n")
        f.write(f"training_unit: {config.get('training_unit', config['train'].get('training_unit', 'image'))}\n")
        f.write(f"pooling: {config['model'].get('pooling', '')}\n")
        f.write(f"k_images: {config['model'].get('k_images', '')}\n")
        f.write(f"eval_uses_all_images: {config['model'].get('eval_uses_all_images', '')}\n")
        f.write(f"exact_trainable_backbone_modules: {model.trainable_backbone_modules}\n")
        f.write(f"checkpoint_state: {config['train']['checkpoint_state']}\n")
        if ordinal_info:
            f.write(f"ordinal_invalid_order_ratio: {ordinal_info['invalid_order_ratio']:.6f}\n")
            f.write("ordinal_correction: p65 clipped to p45 before thresholding\n")
        if regression_info:
            f.write(f"image_mae: {regression_info['image_level']['mae']:.6f}\n")
            f.write(f"subject_mae: {regression_info['subject_level']['mae']:.6f}\n")
    generate_report(seed_dir, epoch=int(logs[-1]["epoch"]))
    append_result_notes(seed_dir, [f"Test metrics are evaluated from best.pt selected by the configured validation metric ({save_best_by})."])
    if regression_info:
        append_result_notes(
            seed_dir,
            [
                f"Regression image MAE/RMSE/Pearson: {regression_info['image_level']['mae']:.4f} / {regression_info['image_level']['rmse']:.4f} / {regression_info['image_level']['pearson']:.4f}.",
                f"Regression subject MAE/RMSE/Pearson: {regression_info['subject_level']['mae']:.4f} / {regression_info['subject_level']['rmse']:.4f} / {regression_info['subject_level']['pearson']:.4f}.",
            ],
        )
    if ordinal_info:
        append_result_notes(
            seed_dir,
            [
                f"Ordinal invalid order ratio on test images: {ordinal_info['invalid_order_ratio']:.6f}.",
                "Ordinal probabilities are corrected with p(age>=65) <= p(age>=45) before class prediction.",
            ],
        )
    if task in {"ldl_boundary", "ldl_gaussian", "ldl_multitask"}:
        ldl_lines = [
            "Test metrics are evaluated from best.pt selected by highest val_macro_f1.",
            f"label_strategy: {config.get('label_strategy', '')}.",
            f"use_class_weights: {config['train']['use_class_weights']}.",
        ]
        if "soft_label_width" in config["train"]:
            ldl_lines.append(f"soft_label_width: {config['train']['soft_label_width']} years.")
        if "ldl_sigma" in config["train"]:
            ldl_lines.append(f"gaussian_sigma: {config['train']['ldl_sigma']} years.")
        if boundary_info:
            for group_name in ("boundary_45", "boundary_65", "non_boundary"):
                group = boundary_info[group_name]
                ldl_lines.append(
                    f"{group_name}: n={group['n_subjects']}, accuracy={group['accuracy']:.4f}, "
                    f"near_miss_rate={group['near_miss_rate']:.4f}, far_miss_rate={group['far_miss_rate']:.4f}."
                )
        append_result_notes(seed_dir, ldl_lines)
    if task == "subject_pool":
        append_result_notes(
            seed_dir,
            [
                "Test metrics are evaluated from best.pt selected by highest val_macro_f1.",
                "This run uses subject-level training: each batch item is one subject with multiple images.",
                f"pooling: {config['model']['pooling']}.",
                f"k_images: {config['model']['k_images']} for training.",
                f"eval_uses_all_images: {config['model']['eval_uses_all_images']}.",
                "For report compatibility, image-level tables mirror subject-level predictions because the prediction unit is subject.",
            ],
        )

    row = {
        "experiment_name": config["experiment_name"],
        "status": "completed",
        "seed": seed,
        "seed_run_dir": str(seed_dir),
        "pretrained": config["model"]["pretrained"],
        "best_epoch": best_epoch,
        "epochs_ran": int(logs[-1]["epoch"]),
        "stop_reason": stop_reason,
        "evaluated_checkpoint": "best.pt",
        "use_best_checkpoint_for_test": True,
        "save_best_by": save_best_by,
        "best_value": best_value,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_mae": best_val_mae,
        "image_accuracy": test_image_metrics["accuracy"],
        "image_balanced_accuracy": test_image_metrics["balanced_accuracy"],
        "image_macro_f1": test_image_metrics["macro_f1"],
        "subject_accuracy": test_subject_metrics["accuracy"],
        "subject_balanced_accuracy": test_subject_metrics["balanced_accuracy"],
        "subject_macro_f1": test_subject_metrics["macro_f1"],
        "image_mae": regression_info.get("image_level", {}).get("mae", math.nan),
        "image_rmse": regression_info.get("image_level", {}).get("rmse", math.nan),
        "image_pearson": regression_info.get("image_level", {}).get("pearson", math.nan),
        "subject_mae": regression_info.get("subject_level", {}).get("mae", math.nan),
        "subject_rmse": regression_info.get("subject_level", {}).get("rmse", math.nan),
        "subject_pearson": regression_info.get("subject_level", {}).get("pearson", math.nan),
        "invalid_order_ratio": ordinal_info.get("invalid_order_ratio", 0.0),
        "log_path": "",
        "failure_reason": "",
    }
    row.update(class_subject_metrics(test_subject_metrics["classification_report"], class_names))
    row.update(flatten_boundary_metrics(boundary_info))
    return row


def run_seed_with_fallback(config: dict[str, Any], seed: int, seed_dir: Path) -> dict[str, Any]:
    batch_size = int(config["train"]["batch_size"])
    fallback_batch_size = int(config["train"]["fallback_batch_size"])
    num_workers = int(config["train"]["num_workers"])
    fallback_num_workers = int(config["train"]["fallback_num_workers"])
    try:
        return run_seed(config, seed, seed_dir, batch_size, num_workers)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg and batch_size > fallback_batch_size:
            if seed_dir.exists():
                import shutil

                shutil.rmtree(seed_dir)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return run_seed(config, seed, seed_dir, fallback_batch_size, num_workers)
        if ("worker" in msg or "dataloader" in msg) and num_workers > fallback_num_workers:
            if seed_dir.exists():
                import shutil

                shutil.rmtree(seed_dir)
            return run_seed(config, seed, seed_dir, batch_size, fallback_num_workers)
        raise


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = pd.DataFrame([row for row in rows if row.get("status") == "completed"])
    if completed.empty:
        return rows
    stats: dict[str, float] = {}
    for metric in [
        "subject_macro_f1",
        "subject_balanced_accuracy",
        "subject_f1_45_64",
        "subject_recall_45_64",
        "image_mae",
        "image_rmse",
        "image_pearson",
        "subject_mae",
        "subject_rmse",
        "subject_pearson",
        "boundary_45_accuracy",
        "boundary_45_macro_f1",
        "boundary_45_near_miss_rate",
        "boundary_45_far_miss_rate",
        "boundary_65_accuracy",
        "boundary_65_macro_f1",
        "boundary_65_near_miss_rate",
        "boundary_65_far_miss_rate",
        "non_boundary_accuracy",
        "non_boundary_macro_f1",
        "non_boundary_near_miss_rate",
        "non_boundary_far_miss_rate",
    ]:
        if metric in completed.columns and completed[metric].notna().any():
            stats[f"{metric}_mean"] = float(completed[metric].mean())
            stats[f"{metric}_std"] = float(completed[metric].std(ddof=1)) if len(completed) > 1 else 0.0
    for row in rows:
        row.update(stats)
    return rows


def failure_row(experiment_name: str, seed: int, seed_dir: Path, log_path: Path, reason: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment_name": experiment_name,
        "status": "failed",
        "seed": seed,
        "seed_run_dir": str(seed_dir),
        "pretrained": USFM_CHECKPOINT,
        "best_epoch": "",
        "epochs_ran": "",
        "stop_reason": "failed",
        "evaluated_checkpoint": "",
        "use_best_checkpoint_for_test": False,
        "save_best_by": "",
        "best_value": math.nan,
        "best_val_macro_f1": math.nan,
        "best_val_mae": math.nan,
        "image_accuracy": math.nan,
        "image_balanced_accuracy": math.nan,
        "image_macro_f1": math.nan,
        "subject_accuracy": math.nan,
        "subject_balanced_accuracy": math.nan,
        "subject_macro_f1": math.nan,
        "image_mae": math.nan,
        "image_rmse": math.nan,
        "image_pearson": math.nan,
        "subject_mae": math.nan,
        "subject_rmse": math.nan,
        "subject_pearson": math.nan,
        "subject_f1_18_44": math.nan,
        "subject_recall_18_44": math.nan,
        "subject_f1_45_64": math.nan,
        "subject_recall_45_64": math.nan,
        "subject_f1_65_100": math.nan,
        "subject_recall_65_100": math.nan,
        "invalid_order_ratio": math.nan,
        "boundary_45_n_subjects": math.nan,
        "boundary_45_accuracy": math.nan,
        "boundary_45_macro_f1": math.nan,
        "boundary_45_near_miss_rate": math.nan,
        "boundary_45_far_miss_rate": math.nan,
        "boundary_65_n_subjects": math.nan,
        "boundary_65_accuracy": math.nan,
        "boundary_65_macro_f1": math.nan,
        "boundary_65_near_miss_rate": math.nan,
        "boundary_65_far_miss_rate": math.nan,
        "non_boundary_n_subjects": math.nan,
        "non_boundary_accuracy": math.nan,
        "non_boundary_macro_f1": math.nan,
        "non_boundary_near_miss_rate": math.nan,
        "non_boundary_far_miss_rate": math.nan,
        "log_path": str(log_path),
        "failure_reason": reason,
    }
    return row


def row_from_seed_dir(experiment_name: str, seed: int, seed_dir: Path, log_path: Path) -> dict[str, Any]:
    if not (seed_dir / "test_metrics.json").exists():
        return failure_row(experiment_name, seed, seed_dir, log_path, "test_metrics.json missing")
    metrics = json.loads((seed_dir / "test_metrics.json").read_text())
    train_log = pd.read_csv(seed_dir / "train_log.csv")
    cfg = json.loads((seed_dir / "config_used.json").read_text())
    save_best_by = str(cfg["train"].get("save_best_by", "val_macro_f1"))
    if save_best_by not in train_log.columns:
        save_best_by = "val_macro_f1"
    best_idx = train_log[save_best_by].idxmin() if save_best_by in {"val_mae", "val_rmse", "val_loss"} else train_log[save_best_by].idxmax()
    best = train_log.loc[best_idx]
    subject_report = metrics["subject_level"]["classification_report"]
    regression_info = metrics.get("regression", {})
    row = {
        "experiment_name": experiment_name,
        "status": "completed",
        "seed": seed,
        "seed_run_dir": str(seed_dir),
        "pretrained": cfg["model"]["pretrained"],
        "best_epoch": int(best["epoch"]),
        "epochs_ran": int(train_log.iloc[-1]["epoch"]),
        "stop_reason": cfg.get("runtime", {}).get("stop_reason", ""),
        "evaluated_checkpoint": cfg["train"].get("evaluated_checkpoint", "best.pt"),
        "use_best_checkpoint_for_test": bool(cfg["train"].get("use_best_checkpoint_for_test", True)),
        "save_best_by": save_best_by,
        "best_value": float(best[save_best_by]),
        "best_val_macro_f1": float(best["val_macro_f1"]),
        "best_val_mae": float(best["val_mae"]) if "val_mae" in best and pd.notna(best["val_mae"]) else math.nan,
        "image_accuracy": metrics["image_level"]["accuracy"],
        "image_balanced_accuracy": metrics["image_level"]["balanced_accuracy"],
        "image_macro_f1": metrics["image_level"]["macro_f1"],
        "subject_accuracy": metrics["subject_level"]["accuracy"],
        "subject_balanced_accuracy": metrics["subject_level"]["balanced_accuracy"],
        "subject_macro_f1": metrics["subject_level"]["macro_f1"],
        "image_mae": regression_info.get("image_level", {}).get("mae", math.nan),
        "image_rmse": regression_info.get("image_level", {}).get("rmse", math.nan),
        "image_pearson": regression_info.get("image_level", {}).get("pearson", math.nan),
        "subject_mae": regression_info.get("subject_level", {}).get("mae", math.nan),
        "subject_rmse": regression_info.get("subject_level", {}).get("rmse", math.nan),
        "subject_pearson": regression_info.get("subject_level", {}).get("pearson", math.nan),
        "invalid_order_ratio": metrics.get("ordinal", {}).get("invalid_order_ratio", 0.0),
        "log_path": str(log_path),
        "failure_reason": "",
    }
    row.update(class_subject_metrics(subject_report, cfg["data"]["class_names"]))
    row.update(flatten_boundary_metrics(metrics.get("boundary_subgroups", {})))
    return row


def write_summary(root_dir: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    rows = aggregate_rows(rows)
    df = pd.DataFrame(rows)
    df.to_csv(root_dir / "summary.csv", index=False)
    save_json(config, root_dir / "config_multiseed.json")
    cols = [
        "experiment_name",
        "status",
        "seed",
        "best_epoch",
        "save_best_by",
        "evaluated_checkpoint",
        "best_val_macro_f1",
        "best_val_mae",
        "image_accuracy",
        "image_balanced_accuracy",
        "image_macro_f1",
        "subject_accuracy",
        "subject_balanced_accuracy",
        "subject_macro_f1",
        "subject_f1_18_44",
        "subject_recall_18_44",
        "subject_f1_45_64",
        "subject_recall_45_64",
        "subject_f1_65_100",
        "subject_recall_65_100",
        "image_mae",
        "image_rmse",
        "image_pearson",
        "subject_mae",
        "subject_rmse",
        "subject_pearson",
        "invalid_order_ratio",
        "boundary_45_n_subjects",
        "boundary_45_accuracy",
        "boundary_45_macro_f1",
        "boundary_45_near_miss_rate",
        "boundary_45_far_miss_rate",
        "boundary_65_n_subjects",
        "boundary_65_accuracy",
        "boundary_65_macro_f1",
        "boundary_65_near_miss_rate",
        "boundary_65_far_miss_rate",
        "non_boundary_n_subjects",
        "non_boundary_accuracy",
        "non_boundary_macro_f1",
        "non_boundary_near_miss_rate",
        "non_boundary_far_miss_rate",
    ]
    existing_cols = [col for col in cols if col in df.columns]
    view = df[existing_cols]
    table_lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for record in view.to_dict(orient="records"):
        values = []
        for col in view.columns:
            value = record[col]
            values.append(f"{value:.4f}" if isinstance(value, float) and math.isfinite(value) else str(value))
        table_lines.append("| " + " | ".join(values) + " |")
    completed = df[df["status"] == "completed"].copy()
    lines = [f"# {config['experiment_name']} Summary", "", "\n".join(table_lines), ""]
    if not completed.empty:
        lines.extend(
            [
                f"subject_macro_f1_mean: {completed['subject_macro_f1'].mean():.4f}",
                f"subject_macro_f1_std: {completed['subject_macro_f1'].std(ddof=1):.4f}",
                f"subject_balanced_accuracy_mean: {completed['subject_balanced_accuracy'].mean():.4f}",
                f"subject_balanced_accuracy_std: {completed['subject_balanced_accuracy'].std(ddof=1):.4f}",
                f"45-64_subject_f1_mean: {completed['subject_f1_45_64'].mean():.4f}",
                f"45-64_subject_f1_std: {completed['subject_f1_45_64'].std(ddof=1):.4f}",
                f"45-64_subject_recall_mean: {completed['subject_recall_45_64'].mean():.4f}",
                f"45-64_subject_recall_std: {completed['subject_recall_45_64'].std(ddof=1):.4f}",
            ]
        )
        for group_name in ("boundary_45", "boundary_65", "non_boundary"):
            accuracy_col = f"{group_name}_accuracy"
            far_col = f"{group_name}_far_miss_rate"
            near_col = f"{group_name}_near_miss_rate"
            if accuracy_col in completed.columns and completed[accuracy_col].notna().any():
                lines.extend(
                    [
                        f"{group_name}_accuracy_mean: {completed[accuracy_col].mean():.4f}",
                        f"{group_name}_far_miss_rate_mean: {completed[far_col].mean():.4f}",
                        f"{group_name}_near_miss_rate_mean: {completed[near_col].mean():.4f}",
                    ]
                )
        if "subject_mae" in completed.columns and completed["subject_mae"].notna().any():
            lines.extend(
                [
                    f"subject_mae_mean: {completed['subject_mae'].mean():.4f}",
                    f"subject_rmse_mean: {completed['subject_rmse'].mean():.4f}",
                    f"subject_pearson_mean: {completed['subject_pearson'].mean():.4f}",
                ]
            )
        if str(config["model"].get("task", "")) == "subject_pool":
            subject_macro_mean = float(completed["subject_macro_f1"].mean())
            subject_balanced_mean = float(completed["subject_balanced_accuracy"].mean())
            mid_f1_mean = float(completed["subject_f1_45_64"].mean())
            mid_recall_mean = float(completed["subject_recall_45_64"].mean())
            mainline_summary = Path("outputs/20260511_162523_usfm_partial_last_block_age45_65_multiseed/summary.csv")
            mainline_mid_f1 = math.nan
            mainline_mid_recall = math.nan
            if mainline_summary.exists():
                mainline_df = pd.read_csv(mainline_summary)
                mainline_done = mainline_df[mainline_df["status"] == "completed"] if "status" in mainline_df.columns else mainline_df
                if "subject_f1_45_64" in mainline_done.columns:
                    mainline_mid_f1 = float(mainline_done["subject_f1_45_64"].mean())
                if "subject_recall_45_64" in mainline_done.columns:
                    mainline_mid_recall = float(mainline_done["subject_recall_45_64"].mean())
            lines.extend(
                [
                    "",
                    "## Subject Pooling Comparison",
                    "",
                    f"current_mainline_subject_macro_f1: {MAINLINE_SUBJECT_MACRO_F1:.4f}",
                    f"current_mainline_subject_balanced_accuracy: {MAINLINE_SUBJECT_BALANCED_ACCURACY:.4f}",
                    f"exceeds_current_mainline_subject_macro_f1: {subject_macro_mean > MAINLINE_SUBJECT_MACRO_F1}",
                    f"exceeds_0.68_subject_macro_f1: {subject_macro_mean > 0.68}",
                    f"subject_balanced_accuracy_delta_vs_mainline: {subject_balanced_mean - MAINLINE_SUBJECT_BALANCED_ACCURACY:.4f}",
                    f"45-64_subject_f1_delta_vs_mainline: {mid_f1_mean - mainline_mid_f1:.4f}" if math.isfinite(mainline_mid_f1) else "45-64_subject_f1_delta_vs_mainline: 待核验",
                    f"45-64_subject_recall_delta_vs_mainline: {mid_recall_mean - mainline_mid_recall:.4f}" if math.isfinite(mainline_mid_recall) else "45-64_subject_recall_delta_vs_mainline: 待核验",
                ]
            )
        lines.extend(
            [
                "",
                "All test metrics are evaluated from best.pt selected by the configured validation metric.",
            ]
        )
    failed = df[df["status"] == "failed"].copy()
    if not failed.empty:
        lines.extend(["", "## Failed Tasks", ""])
        for row in failed.to_dict(orient="records"):
            lines.append(f"- seed {row['seed']}: {row.get('failure_reason', '')}; log: {row.get('log_path', '')}")
    (root_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


TRAINLOG_NOTES = {
    "usfm_partial_ldl_boundary_w3": "LDL 边界软标签 w3；只软化 45/65 岁附近样本，主要看能否减少边界错分并改善 45-64。",
    "usfm_partial_ldl_gaussian_sigma3": "LDL 高斯年龄分布 sigma=3；把连续年龄概率累加到三类，检验连续年龄信息是否比手写边界规则更稳。",
    "usfm_partial_ldl_multitask_w3_lam0p3": "LDL 边界软标签 + 年龄回归辅助；分类仍按三分类评估，回归头用来约束年龄连续性。",
    "usfm_subject_mean_pool_k3": "subject-level mean pooling；每个受试者作为一个样本，训练时采 3 张图，验证/测试用全部图像均值池化。",
    "usfm_subject_attention_pool_k3": "subject-level attention pooling；每个受试者作为一个样本，让模型学习多张图的权重后再分类。",
}


def format_mean_std_from_df(df: pd.DataFrame, col: str) -> str:
    completed = df[df["status"] == "completed"].copy()
    if completed.empty or col not in completed.columns or not completed[col].notna().any():
        return "待核验"
    mean = float(completed[col].mean())
    std = float(completed[col].std(ddof=1)) if len(completed) > 1 else 0.0
    return f"{mean:.4f} ± {std:.4f}"


def format_mean_from_df(df: pd.DataFrame, col: str) -> str:
    completed = df[df["status"] == "completed"].copy()
    if completed.empty or col not in completed.columns or not completed[col].notna().any():
        return "待核验"
    return f"{float(completed[col].mean()):.4f}"


def update_trainlog_xlsx(trainlog_path: Path) -> None:
    rows = []
    if not trainlog_path.exists():
        return
    for line in trainlog_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split(" | ", 3)
        if len(parts) != 4:
            continue
        rows.append({"subject_macro_f1": parts[0], "status": parts[1], "path": parts[2], "亮点/结论": parts[3]})
    if not rows:
        return
    try:
        pd.DataFrame(rows).to_excel(trainlog_path.with_suffix(".xlsx"), index=False)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: failed to update trainlog.xlsx: {exc}")


def append_outputs_trainlog(roots: dict[str, Path]) -> None:
    trainlog_path = Path("outputs") / "trainlog"
    trainlog_path.parent.mkdir(parents=True, exist_ok=True)
    if not trainlog_path.exists():
        trainlog_path.write_text("# stage-age outputs trainlog\n# 格式：subject_macro_f1 | 状态 | 路径 | 亮点/结论\n\n", encoding="utf-8")
    existing = trainlog_path.read_text(encoding="utf-8")
    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for _experiment, root_dir in roots.items():
        root_str = str(root_dir)
        if root_str in existing:
            continue
        summary_path = root_dir / "summary.csv"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        experiment_name = str(df["experiment_name"].dropna().iloc[0]) if "experiment_name" in df.columns and not df.empty else root_dir.name
        completed = df[df["status"] == "completed"].copy() if "status" in df.columns else pd.DataFrame()
        if not completed.empty and "subject_" in experiment_name:
            status = "active_subject_pool"
        else:
            status = "active_ldl" if not completed.empty else "failed"
        subject_macro = format_mean_std_from_df(df, "subject_macro_f1") if not completed.empty else "failed"
        subject_bal = format_mean_std_from_df(df, "subject_balanced_accuracy")
        mid_f1 = format_mean_std_from_df(df, "subject_f1_45_64")
        mid_recall = format_mean_std_from_df(df, "subject_recall_45_64")
        b45_acc = format_mean_from_df(df, "boundary_45_accuracy")
        b45_far = format_mean_from_df(df, "boundary_45_far_miss_rate")
        b65_acc = format_mean_from_df(df, "boundary_65_accuracy")
        b65_far = format_mean_from_df(df, "boundary_65_far_miss_rate")
        note = TRAINLOG_NOTES.get(experiment_name, f"{experiment_name} 实验。")
        failed_count = int((df["status"] == "failed").sum()) if "status" in df.columns else 0
        if failed_count:
            note += f" 有 {failed_count} 个 seed 失败，请看对应 logs。"
        note += (
            f" balanced accuracy={subject_bal}；45-64 F1={mid_f1}，recall={mid_recall}；"
            f"boundary45 acc/far={b45_acc}/{b45_far}，boundary65 acc/far={b65_acc}/{b65_far}；"
            f"记录时间 {timestamp}。"
        )
        lines.append(f"{subject_macro} | {status} | {root_str} | {note}")
    if lines:
        with trainlog_path.open("a", encoding="utf-8") as f:
            if not existing.endswith("\n"):
                f.write("\n")
            f.write("\n".join(lines) + "\n")
        update_trainlog_xlsx(trainlog_path)


def run_one_seed(args: argparse.Namespace) -> None:
    config = experiment_config(args.run_one_experiment)
    config["multiseed"] = {"seeds": [args.run_one_seed], "root_dir": args.root_dir}
    row = run_seed_with_fallback(config, args.run_one_seed, Path(args.root_dir) / f"seed{args.run_one_seed}")
    save_json(row, Path(args.root_dir) / f"seed{args.run_one_seed}" / "seed_summary.json")


def prepare_roots(experiments: list[str], seeds: list[int]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for experiment in experiments:
        config = experiment_config(experiment)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_dir = unique_output_dir(Path("outputs") / f"{timestamp}_{config['experiment_name']}_age45_65_multiseed")
        root_dir.mkdir(parents=True, exist_ok=False)
        (root_dir / "logs").mkdir(parents=True, exist_ok=True)
        config["multiseed"] = {"seeds": seeds, "root_dir": str(root_dir)}
        save_json(config, root_dir / "config_multiseed.json")
        roots[experiment] = root_dir
    return roots


def task_command(experiment: str, seed: int, root_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "stage_age.run_next_stage_experiments",
        "--run-one-experiment",
        experiment,
        "--run-one-seed",
        str(seed),
        "--root-dir",
        str(root_dir),
    ]


def worker_loop(
    gpu_id: str,
    tasks: queue.Queue[tuple[str, int, Path]],
    results: list[dict[str, Any]],
    results_lock: threading.Lock,
    repo_root: Path,
) -> None:
    while True:
        try:
            experiment, seed, root_dir = tasks.get_nowait()
        except queue.Empty:
            return
        config = experiment_config(experiment)
        log_path = root_dir / "logs" / f"{config['experiment_name']}_seed{seed}_gpu{gpu_id}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONPATH"] = str(repo_root / "src")
        seed_dir = root_dir / f"seed{seed}"
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                task_command(experiment, seed, root_dir),
                cwd=str(repo_root),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            returncode = proc.wait()
        if returncode == 0:
            row = row_from_seed_dir(config["experiment_name"], seed, seed_dir, log_path)
        else:
            reason = f"subprocess exited with code {returncode}"
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-10:]
                if tail:
                    reason += ": " + " | ".join(tail[-3:])
            except OSError:
                pass
            row = failure_row(config["experiment_name"], seed, seed_dir, log_path, reason)
        with results_lock:
            results.append(row)
        tasks.task_done()


def aggregate_all(roots: dict[str, Path], seeds: list[int]) -> dict[str, pd.DataFrame]:
    summaries: dict[str, pd.DataFrame] = {}
    for experiment, root_dir in roots.items():
        config = experiment_config(experiment)
        config["multiseed"] = {"seeds": seeds, "root_dir": str(root_dir)}
        rows = []
        for seed in seeds:
            log_candidates = sorted((root_dir / "logs").glob(f"{config['experiment_name']}_seed{seed}_gpu*.log"))
            log_path = log_candidates[0] if log_candidates else root_dir / "logs" / f"{config['experiment_name']}_seed{seed}_gpuunknown.log"
            rows.append(row_from_seed_dir(config["experiment_name"], seed, root_dir / f"seed{seed}", log_path))
        write_summary(root_dir, rows, config)
        summaries[experiment] = pd.read_csv(root_dir / "summary.csv")
    return summaries


def append_subject_pool_cross_comparison(roots: dict[str, Path]) -> None:
    subject_roots = {name: root for name, root in roots.items() if name in {"subject_mean_pool_k3", "subject_attention_pool_k3"}}
    if len(subject_roots) < 2:
        return
    rows = []
    for experiment, root in subject_roots.items():
        df = pd.read_csv(root / "summary.csv")
        completed = df[df["status"] == "completed"].copy()
        if completed.empty:
            continue
        rows.append(
            {
                "experiment": experiment,
                "root": root,
                "subject_macro_f1_mean": float(completed["subject_macro_f1"].mean()),
                "subject_macro_f1_std": float(completed["subject_macro_f1"].std(ddof=1)) if len(completed) > 1 else 0.0,
                "subject_balanced_accuracy_mean": float(completed["subject_balanced_accuracy"].mean()),
                "subject_f1_45_64_mean": float(completed["subject_f1_45_64"].mean()),
                "subject_recall_45_64_mean": float(completed["subject_recall_45_64"].mean()),
            }
        )
    if len(rows) < 2:
        return
    best_macro = max(rows, key=lambda row: row["subject_macro_f1_mean"])
    most_stable = min(rows, key=lambda row: row["subject_macro_f1_std"])
    table = [
        "| experiment | subject_macro_f1_mean | subject_macro_f1_std | subject_balanced_accuracy_mean | 45-64_f1_mean | 45-64_recall_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        table.append(
            f"| {row['experiment']} | {row['subject_macro_f1_mean']:.4f} | {row['subject_macro_f1_std']:.4f} | "
            f"{row['subject_balanced_accuracy_mean']:.4f} | {row['subject_f1_45_64_mean']:.4f} | {row['subject_recall_45_64_mean']:.4f} |"
        )
    section = [
        "",
        "## Mean vs Attention Pooling",
        "",
        "\n".join(table),
        "",
        f"best_subject_macro_f1: {best_macro['experiment']}",
        f"more_stable_by_subject_macro_f1_std: {most_stable['experiment']}",
    ]
    for row in rows:
        with (row["root"] / "summary.md").open("a", encoding="utf-8") as f:
            f.write("\n".join(section) + "\n")


def run_scheduler(args: argparse.Namespace) -> dict[str, Path]:
    check_preflight()
    if int(args.parallel_per_gpu) != 1:
        raise ValueError("This scheduler supports --parallel_per_gpu 1 only.")
    experiments = ALIASES[args.experiment]
    roots = prepare_roots(experiments, args.seeds)
    tasks: queue.Queue[tuple[str, int, Path]] = queue.Queue()
    for experiment in experiments:
        for seed in args.seeds:
            tasks.put((experiment, seed, roots[experiment]))

    results: list[dict[str, Any]] = []
    results_lock = threading.Lock()
    repo_root = Path(__file__).resolve().parents[2]
    worker_count = min(int(args.max_parallel), len(args.gpus), tasks.qsize())
    threads = []
    for idx in range(worker_count):
        gpu_id = str(args.gpus[idx % len(args.gpus)])
        thread = threading.Thread(target=worker_loop, args=(gpu_id, tasks, results, results_lock, repo_root), daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    # Re-read from disk so summaries are independent of scheduler memory state.
    for experiment, root_dir in roots.items():
        config = experiment_config(experiment)
        config["multiseed"] = {
            "seeds": args.seeds,
            "root_dir": str(root_dir),
            "gpus": [str(gpu) for gpu in args.gpus],
            "parallel_per_gpu": int(args.parallel_per_gpu),
            "max_parallel": int(args.max_parallel),
            "scheduler": "subprocess_gpu_worker_queue",
        }
        rows = []
        for seed in args.seeds:
            log_candidates = sorted((root_dir / "logs").glob(f"{config['experiment_name']}_seed{seed}_gpu*.log"))
            log_path = log_candidates[0] if log_candidates else root_dir / "logs" / f"{config['experiment_name']}_seed{seed}_gpuunknown.log"
            seed_dir = root_dir / f"seed{seed}"
            if (seed_dir / "test_metrics.json").exists():
                rows.append(row_from_seed_dir(config["experiment_name"], seed, seed_dir, log_path))
            else:
                rows.append(failure_row(config["experiment_name"], seed, seed_dir, log_path, "test_metrics.json missing"))
        write_summary(root_dir, rows, config)
    append_subject_pool_cross_comparison(roots)
    append_outputs_trainlog(roots)
    return roots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(ALIASES), default="all")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3", "4", "5"])
    parser.add_argument("--parallel_per_gpu", type=int, default=1)
    parser.add_argument("--max_parallel", type=int, default=6)
    parser.add_argument("--run-one-experiment", choices=sorted(EXPERIMENTS), default=None)
    parser.add_argument("--run-one-seed", type=int, default=None)
    parser.add_argument("--root-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_one_experiment is not None:
        if args.run_one_seed is None or args.root_dir is None:
            raise ValueError("--run-one-seed and --root-dir are required with --run-one-experiment")
        run_one_seed(args)
        return
    roots = run_scheduler(args)
    for experiment, root_dir in roots.items():
        print(f"{experiment}_run_dir={root_dir}")


if __name__ == "__main__":
    main()
