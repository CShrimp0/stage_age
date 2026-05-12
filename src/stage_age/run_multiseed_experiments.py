from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage_age.data import UltrasoundAgeDataset, build_manifest, make_transforms
from stage_age.metrics import image_level_metrics, subject_level_predictions
from stage_age.models import build_model as build_torchvision_model
from stage_age.report import generate_report
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
    "train": {
        "epochs": 30,
        "num_workers": 8,
        "fallback_num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "use_class_weights": True,
        "device": "cuda",
        "amp": True,
        "grad_accum_steps": 1,
        "weight_decay": 0.0001,
        "save_best_by": "val_macro_f1",
        "test_checkpoint": "best.pt",
        "evaluated_checkpoint": "best.pt",
        "use_best_checkpoint_for_test": True,
    },
}


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "resnet18_baseline": {
        "model": {
            "name": "resnet18_baseline",
            "backbone": "resnet18",
            "pretrained": "ImageNet",
            "num_classes": 3,
            "image_size": 224,
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        },
        "train": {
            "batch_size": 64,
            "fallback_batch_size": 32,
            "lr": 0.0003,
            "head_lr": 0.0003,
            "backbone_lr": 0.0003,
        },
    },
    "usfm_linear_probe": {
        "model": {
            "name": "usfm_linear_probe",
            "pretrained": USFM_CHECKPOINT,
            "checkpoint_path": USFM_CHECKPOINT,
            "adapter_path": USFM_ADAPTER,
            "image_size": 224,
            "input_channels": 3,
            "input_mode": "grayscale images converted to RGB",
            "global_pool": "token",
            "head_type": "linear",
            "dropout": 0.0,
            "freeze_backbone": True,
            "unfreeze_last_n_blocks": 0,
            "trainable": "LayerNorm + Linear classification head",
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        },
        "train": {
            "batch_size": 64,
            "fallback_batch_size": 32,
            "lr": 0.0003,
            "head_lr": 0.0003,
            "backbone_lr": 0.0,
        },
    },
    "usfm_mlp_probe": {
        "model": {
            "name": "usfm_mlp_probe",
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
            "unfreeze_last_n_blocks": 0,
            "trainable": "MLP classification head only",
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        },
        "train": {
            "batch_size": 64,
            "fallback_batch_size": 32,
            "lr": 0.0003,
            "head_lr": 0.0003,
            "backbone_lr": 0.0,
        },
    },
    "usfm_partial_last_block": {
        "model": {
            "name": "usfm_partial_last_block",
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
            "trainable": "last USFM transformer block + final norm if present + MLP classification head",
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        },
        "train": {
            "batch_size": 32,
            "fallback_batch_size": 16,
            "lr": 0.0003,
            "head_lr": 0.0003,
            "backbone_lr": 0.00003,
        },
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def unique_output_dir(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        return path
    idx = 2
    while True:
        candidate = Path(f"{path}_v{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def model_config(model_name: str) -> dict[str, Any]:
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")
    return deep_update(BASE_CONFIG, MODEL_CONFIGS[model_name])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but device=cuda was requested.")
    return torch.device(device_name)


def gpu_name(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    return torch.cuda.get_device_name(device)


def check_preflight() -> None:
    paths = [
        Path(BASE_CONFIG["data"]["image_dir"]),
        Path(BASE_CONFIG["data"]["characteristics"]),
        Path(USFM_CHECKPOINT),
        Path(USFM_ADAPTER),
    ]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
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


def split_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    subjects = manifest.drop_duplicates("subject_id")
    return (
        subjects.groupby(["split", "class_name"])
        .size()
        .rename("subjects")
        .reset_index()
        .merge(
            manifest.groupby(["split", "class_name"]).size().rename("images").reset_index(),
            on=["split", "class_name"],
        )
    )


def make_loader(
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
    return DataLoader(UltrasoundAgeDataset(manifest, transform), **kwargs)


def build_model(config: dict[str, Any], device: torch.device) -> nn.Module:
    name = config["model"]["name"]
    if name == "resnet18_baseline":
        return build_torchvision_model("resnet18", num_classes=3, pretrained=True).to(device)
    model = USFMClassifier(
        checkpoint_path=config["model"]["checkpoint_path"],
        adapter_path=config["model"]["adapter_path"],
        image_size=int(config["model"]["image_size"]),
        global_pool=str(config["model"]["global_pool"]),
        num_classes=len(config["data"]["class_names"]),
        head_type=str(config["model"]["head_type"]),
        freeze_backbone=bool(config["model"]["freeze_backbone"]),
        unfreeze_last_n_blocks=int(config["model"]["unfreeze_last_n_blocks"]),
        dropout=float(config["model"]["dropout"]),
    ).to(device)
    return model


def make_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    if config["model"]["name"] == "resnet18_baseline":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(config["train"]["lr"]),
            weight_decay=float(config["train"]["weight_decay"]),
        )
    head_params = [param for param in model.head.parameters() if param.requires_grad]
    backbone_params = [param for param in model.backbone.parameters() if param.requires_grad]
    groups = [{"params": head_params, "lr": float(config["train"]["head_lr"])}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(config["train"]["backbone_lr"])})
    return torch.optim.AdamW(groups, weight_decay=float(config["train"]["weight_decay"]))


def compute_class_weights(train_manifest: pd.DataFrame, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = train_manifest["label"].value_counts().reindex(range(num_classes), fill_value=0).to_numpy()
    if (counts == 0).any():
        raise ValueError(f"At least one class is missing from train split: {counts.tolist()}")
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


def check_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise FloatingPointError(f"{name} is not finite: {value}")


def train_one_epoch(
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
        with autocast_context(device, amp_enabled):
            logits = model(images)
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
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_items += batch_size
        progress.set_postfix(loss=total_loss / total_items, acc=total_correct / total_items)
    return {"loss": total_loss / total_items, "accuracy": total_correct / total_items}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, amp_enabled: bool) -> dict[str, Any]:
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


def save_json(obj: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def report_to_csv(report: dict[str, Any], class_names: list[str], path: Path) -> None:
    rows = []
    for name in class_names:
        item = report[name]
        rows.append({"class": name, "precision": item["precision"], "recall": item["recall"], "f1": item["f1-score"], "support": int(item["support"])})
    pd.DataFrame(rows).to_csv(path, index=False)


def save_predictions(
    output_dir: Path,
    test_manifest: pd.DataFrame,
    probs: np.ndarray,
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
    for idx, name in enumerate(class_names):
        image_df[f"prob_{name}"] = probs[:, idx]
    image_df.to_csv(output_dir / "image_test_predictions.csv", index=False)

    subject_true, subject_pred, subject_ids_out = subject_level_predictions(probs, labels, subject_ids)
    subject_df = pd.DataFrame({"subject_id": subject_ids_out, "label": subject_true, "pred": subject_pred})
    subject_df["label_name"] = [class_names[idx] for idx in subject_true]
    subject_df["pred_name"] = [class_names[idx] for idx in subject_pred]
    probs_df = pd.DataFrame({"subject_id": subject_ids})
    for idx, name in enumerate(class_names):
        probs_df[f"prob_{name}"] = probs[:, idx]
    subject_df = subject_df.merge(probs_df.groupby("subject_id", sort=True).mean().reset_index(), on="subject_id", how="left")
    subject_df.to_csv(output_dir / "subject_test_predictions.csv", index=False)


def smoke_forward(model: nn.Module, device: torch.device, num_classes: int) -> None:
    model.eval()
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        logits = model(x)
    if tuple(logits.shape) != (2, num_classes):
        raise RuntimeError(f"Smoke forward failed: expected (2, {num_classes}), got {tuple(logits.shape)}")
    print(f"smoke_forward_ok shape={tuple(logits.shape)}")


def model_param_summary(model: nn.Module) -> dict[str, Any]:
    total, trainable = count_parameters(model)
    summary: dict[str, Any] = {"total_params": total, "trainable_params": trainable}
    if hasattr(model, "backbone"):
        summary["backbone_params"] = sum(param.numel() for param in model.backbone.parameters())
        summary["trainable_backbone_params"] = sum(param.numel() for param in model.backbone.parameters() if param.requires_grad)
        summary["trainable_backbone_modules"] = getattr(model, "trainable_backbone_modules", [])
    if hasattr(model, "head"):
        summary["head_params"] = sum(param.numel() for param in model.head.parameters())
    return summary


def write_model_summary(model: nn.Module, config: dict[str, Any], path: Path, elapsed: float | None = None) -> None:
    params = model_param_summary(model)
    lines = [
        f"model: {config['model']['name']}",
        f"pretrained: {config['model']['pretrained']}",
        f"checkpoint_path: {config['model'].get('checkpoint_path', '')}",
        f"adapter_path: {config['model'].get('adapter_path', '')}",
        f"input_size: {config['model']['image_size']}x{config['model']['image_size']}",
        "input_channels: 3",
        f"normalization_mean: {config['model']['normalization']['mean']}",
        f"normalization_std: {config['model']['normalization']['std']}",
        f"save_best_by: {config['train']['save_best_by']}",
        f"test_checkpoint: {config['train']['test_checkpoint']}",
        f"evaluated_checkpoint: {config['train']['evaluated_checkpoint']}",
        f"use_best_checkpoint_for_test: {config['train']['use_best_checkpoint_for_test']}",
        f"gpu_name: {config['runtime']['gpu_name']}",
        f"amp_enabled: {config['train']['amp']}",
        f"batch_size: {config['train']['actual_batch_size']}",
        f"effective_batch_size: {config['train']['effective_batch_size']}",
        f"num_workers: {config['train']['num_workers']}",
    ]
    if config["model"]["name"].startswith("usfm"):
        lines.extend(
            [
                f"global_pool: {config['model']['global_pool']}",
                f"head_type: {config['model']['head_type']}",
                f"dropout: {config['model']['dropout']}",
                f"frozen_backbone: {config['model']['freeze_backbone']}",
                f"unfreeze_last_n_blocks: {config['model']['unfreeze_last_n_blocks']}",
                f"trainable_backbone_modules: {params.get('trainable_backbone_modules', [])}",
                f"trainable_layers: {config['model']['trainable']}",
            ]
        )
    for key, value in params.items():
        lines.append(f"{key}: {value}")
    if elapsed is not None:
        lines.append(f"train_elapsed_seconds: {elapsed:.2f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_seed(config: dict[str, Any], seed: int, seed_dir: Path, batch_size: int, num_workers: int) -> dict[str, Any]:
    set_seed(seed)
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
    train_loader = make_loader(train_manifest, train_tf, batch_size, num_workers, True, config)
    val_loader = make_loader(val_manifest, eval_tf, batch_size, num_workers, False, config)
    test_loader = make_loader(test_manifest, eval_tf, batch_size, num_workers, False, config)

    model = build_model(config, device)
    smoke_forward(model, device, len(config["data"]["class_names"]))
    write_model_summary(model, config, seed_dir / "model_summary.txt")
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_manifest, len(config["data"]["class_names"]), device)
        if config["train"]["use_class_weights"]
        else None
    )
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["train"]["epochs"]))
    amp_enabled = bool(config["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
    best_val_macro_f1 = -1.0
    best_epoch = -1
    logs: list[dict[str, float]] = []
    class_names = config["data"]["class_names"]
    start = time.perf_counter()

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
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
        val_pred = predict(model, val_loader, criterion, device, amp_enabled)
        val_metrics = image_level_metrics(val_pred["labels"], val_pred["preds"], class_names)
        check_finite(float(val_metrics["macro_f1"]), "val_macro_f1")
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": val_pred["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        logs.append(row)
        pd.DataFrame(logs).to_csv(seed_dir / "train_log.csv", index=False)
        if epoch % 5 == 0 or epoch == int(config["train"]["epochs"]):
            generate_report(seed_dir, epoch=epoch)
        print(f"seed={seed} epoch={epoch:03d} val_macro_f1={row['val_macro_f1']:.4f}")
        if row["val_macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = row["val_macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "config": config,
                    "best_val_macro_f1": best_val_macro_f1,
                    "save_best_by": "val_macro_f1",
                },
                seed_dir / "best.pt",
            )

    elapsed = time.perf_counter() - start
    checkpoint = torch.load(seed_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_pred = predict(model, test_loader, criterion, device, amp_enabled)
    test_image_metrics = image_level_metrics(test_pred["labels"], test_pred["preds"], class_names)
    subject_true, subject_pred, _subject_ids = subject_level_predictions(
        test_pred["probs"], test_pred["labels"], test_pred["subject_ids"]
    )
    test_subject_metrics = image_level_metrics(subject_true, subject_pred, class_names)
    metrics = {"image_level": test_image_metrics, "subject_level": test_subject_metrics}
    save_json(metrics, seed_dir / "test_metrics.json")
    save_predictions(
        seed_dir,
        test_manifest,
        test_pred["probs"],
        test_pred["labels"],
        test_pred["preds"],
        test_pred["subject_ids"],
        class_names,
    )
    report_to_csv(test_image_metrics["classification_report"], class_names, seed_dir / "image_classification_report.csv")
    report_to_csv(test_subject_metrics["classification_report"], class_names, seed_dir / "subject_classification_report.csv")
    config["runtime"]["train_elapsed_seconds"] = elapsed
    save_json(config, seed_dir / "config_used.json")
    save_json(config, seed_dir / "config.json")
    write_model_summary(model, config, seed_dir / "model_summary.txt", elapsed)
    generate_report(seed_dir, epoch=int(config["train"]["epochs"]))

    return {
        "model": config["model"]["name"],
        "seed": seed,
        "seed_run_dir": str(seed_dir),
        "pretrained": config["model"]["pretrained"],
        "best_epoch": best_epoch,
        "evaluated_checkpoint": "best.pt",
        "use_best_checkpoint_for_test": True,
        "best_val_macro_f1": best_val_macro_f1,
        "image_accuracy": test_image_metrics["accuracy"],
        "image_balanced_accuracy": test_image_metrics["balanced_accuracy"],
        "image_macro_f1": test_image_metrics["macro_f1"],
        "subject_accuracy": test_subject_metrics["accuracy"],
        "subject_balanced_accuracy": test_subject_metrics["balanced_accuracy"],
        "subject_macro_f1": test_subject_metrics["macro_f1"],
    }


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
                shutil.rmtree(seed_dir)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return run_seed(config, seed, seed_dir, fallback_batch_size, num_workers)
        if ("worker" in msg or "dataloader" in msg) and num_workers > fallback_num_workers:
            if seed_dir.exists():
                shutil.rmtree(seed_dir)
            return run_seed(config, seed, seed_dir, batch_size, fallback_num_workers)
        raise


def add_aggregate_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    subj_f1_mean = float(df["subject_macro_f1"].mean())
    subj_f1_std = float(df["subject_macro_f1"].std(ddof=1)) if len(df) > 1 else 0.0
    subj_bal_mean = float(df["subject_balanced_accuracy"].mean())
    subj_bal_std = float(df["subject_balanced_accuracy"].std(ddof=1)) if len(df) > 1 else 0.0
    for row in rows:
        row["subject_macro_f1_mean"] = subj_f1_mean
        row["subject_macro_f1_std"] = subj_f1_std
        row["subject_balanced_accuracy_mean"] = subj_bal_mean
        row["subject_balanced_accuracy_std"] = subj_bal_std
    return rows


def write_summary(root_dir: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    rows = add_aggregate_columns(rows)
    df = pd.DataFrame(rows)
    df.to_csv(root_dir / "summary.csv", index=False)
    save_json(config, root_dir / "config_multiseed.json")
    metric_cols = [
        "model",
        "seed",
        "best_epoch",
        "best_val_macro_f1",
        "image_macro_f1",
        "subject_balanced_accuracy",
        "subject_macro_f1",
    ]
    summary_table = df[metric_cols].copy()
    table_lines = [
        "| " + " | ".join(summary_table.columns) + " |",
        "| " + " | ".join(["---"] * len(summary_table.columns)) + " |",
    ]
    for record in summary_table.to_dict(orient="records"):
        values = []
        for col in summary_table.columns:
            value = record[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        table_lines.append("| " + " | ".join(values) + " |")
    lines = [
        "# Multi-Seed Summary",
        "",
        "\n".join(table_lines),
        "",
        f"subject_macro_f1_mean: {df['subject_macro_f1'].mean():.4f}",
        f"subject_macro_f1_std: {df['subject_macro_f1'].std(ddof=1):.4f}",
        f"subject_balanced_accuracy_mean: {df['subject_balanced_accuracy'].mean():.4f}",
        f"subject_balanced_accuracy_std: {df['subject_balanced_accuracy'].std(ddof=1):.4f}",
        "",
        "All test metrics are evaluated from best.pt selected by highest val_macro_f1.",
    ]
    (root_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_multiseed(model_name: str, seeds: list[int], epochs: int | None = None) -> Path:
    check_preflight()
    config = model_config(model_name)
    if epochs is not None:
        config["train"]["epochs"] = epochs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = unique_output_dir(Path("outputs") / f"{timestamp}_{model_name}_age45_65_multiseed")
    root_dir.mkdir(parents=True, exist_ok=False)
    config["multiseed"] = {"seeds": seeds, "root_dir": str(root_dir)}
    save_json(config, root_dir / "config_multiseed.json")
    device = resolve_device(config["train"]["device"])
    probe_model = build_model(config, device)
    smoke_forward(probe_model, device, len(config["data"]["class_names"]))
    del probe_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"starting model={model_name} seed={seed}")
        row = run_seed_with_fallback(config, seed, root_dir / f"seed{seed}")
        rows.append(row)
        write_summary(root_dir, rows, config)
    return root_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_CONFIGS.keys()),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = run_multiseed(args.model, args.seeds, epochs=args.epochs)
    print(f"multiseed_run_dir={root}")


if __name__ == "__main__":
    main()
