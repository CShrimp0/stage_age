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


def build_model(config: dict[str, Any], device: torch.device) -> USFMClassifier:
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


def make_optimizer(model: USFMClassifier, config: dict[str, Any]) -> torch.optim.Optimizer:
    head_params = [param for param in model.head.parameters() if param.requires_grad]
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
        if name.startswith("head.") or name.startswith("backbone.encoder.blocks.11.") or name.startswith("backbone.encoder.fc_norm.")
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
    dataset = UltrasoundAgeRegressionDataset(manifest, transform) if task == "regression_binning" else UltrasoundAgeDataset(manifest, transform)
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
        f.write(f"exact_trainable_backbone_modules: {model.trainable_backbone_modules}\n")
        f.write(f"total_params_exact: {total}\n")
        f.write(f"trainable_params_exact: {trainable}\n")

    task = str(config["model"]["task"])
    if task == "ordinal":
        criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=ordinal_pos_weight(train_manifest, device))
    elif task == "regression_binning":
        criterion = nn.SmoothL1Loss()
    elif task == "focal_loss":
        weights = compute_class_weights(train_manifest, len(config["data"]["class_names"]), device)
        criterion = FocalLoss(gamma=float(config["train"]["focal_gamma"]), alpha=weights)
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
        val_reg_metrics = regression_metrics(val_pred["true_age"], val_pred["pred_age"]) if task == "regression_binning" else {}
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
        if "subject_mae" in completed.columns and completed["subject_mae"].notna().any():
            lines.extend(
                [
                    f"subject_mae_mean: {completed['subject_mae'].mean():.4f}",
                    f"subject_rmse_mean: {completed['subject_rmse'].mean():.4f}",
                    f"subject_pearson_mean: {completed['subject_pearson'].mean():.4f}",
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
