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

from stage_age.data import UltrasoundAgeDataset, make_transforms
from stage_age.metrics import image_level_metrics, subject_level_predictions
from stage_age.report import generate_report
from stage_age.usfm import DEFAULT_USFM_ADAPTER_PATH, USFMClassifier, count_parameters


FIXED_MANIFEST = "/home/szdx/LNX/stage-age/outputs/20260511_152330_resnet18_age45_65/manifest.csv"
USFM_CHECKPOINT = "/home/szdx/LNX/stage-age/USFM_latest.pth"
USFM_ADAPTER = str(DEFAULT_USFM_ADAPTER_PATH)


BASE_CONFIG: dict[str, Any] = {
    "data": {
        "image_dir": "/home/szdx/LNX/data/TA/Healthy/Images",
        "characteristics": "/home/szdx/LNX/data/TA/characteristics.xlsx",
        "sheet_name": "Blad1",
        "manifest_path": FIXED_MANIFEST,
        "bins": [18, 45, 65, 101],
        "class_names": ["18-44", "45-64", "65-100"],
    },
    "model": {
        "pretrained": USFM_CHECKPOINT,
        "checkpoint_path": USFM_CHECKPOINT,
        "adapter_path": USFM_ADAPTER,
        "image_size": 224,
        "input_channels": 3,
        "input_mode": "grayscale images converted to RGB",
        "global_pool": "token",
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    },
    "train": {
        "epochs": 30,
        "num_workers": 8,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "use_class_weights": True,
        "seed": 42,
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


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "mlp_probe": {
        "output_root": "/home/szdx/LNX/stage-age/outputs",
        "run_name": "usfm_mlp_probe_age45_65",
        "model": {
            "name": "usfm_mlp_probe",
            "head_type": "mlp",
            "dropout": 0.2,
            "freeze_backbone": True,
            "unfreeze_last_n_blocks": 0,
            "trainable": "MLP classification head only",
        },
        "train": {
            "batch_size": 64,
            "fallback_batch_size": 32,
            "lr": 0.0003,
            "head_lr": 0.0003,
            "backbone_lr": 0.0,
        },
    },
    "partial_last_block": {
        "output_root": "/home/szdx/LNX/stage-age/outputs",
        "run_name": "usfm_partial_last_block_age45_65",
        "model": {
            "name": "usfm_partial_last_block",
            "head_type": "mlp",
            "dropout": 0.2,
            "freeze_backbone": True,
            "unfreeze_last_n_blocks": 1,
            "trainable": "last USFM transformer block + final norm if present + MLP classification head",
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


def make_config(name: str) -> dict[str, Any]:
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}")
    return deep_update(BASE_CONFIG, EXPERIMENTS[name])


def unique_output_dir(base_dir: str | Path) -> Path:
    base = Path(base_dir)
    if not base.exists():
        return base
    idx = 2
    while True:
        candidate = Path(f"{base}_v{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def resolve_output_dir(config: dict[str, Any]) -> Path:
    if "output_dir" in config:
        return unique_output_dir(config["output_dir"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return unique_output_dir(Path(config.get("output_root", "outputs")) / f"{timestamp}_{config.get('run_name', 'usfm')}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def gpu_name(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    return torch.cuda.get_device_name(device)


def check_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise FloatingPointError(f"{name} is not finite: {value}")


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fixed manifest not found: {path}")
    manifest = pd.read_csv(path)
    required = {"image_path", "subject_id", "label", "class_name", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
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


def compute_class_weights(train_manifest: pd.DataFrame, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = train_manifest["label"].value_counts().reindex(range(num_classes), fill_value=0).to_numpy()
    if (counts == 0).any():
        raise ValueError(f"At least one class is missing from train split: {counts.tolist()}")
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_model(config: dict[str, Any], device: torch.device) -> USFMClassifier:
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


def make_optimizer(model: USFMClassifier, config: dict[str, Any]) -> torch.optim.Optimizer:
    head_params = [param for param in model.head.parameters() if param.requires_grad]
    backbone_params = [param for param in model.backbone.parameters() if param.requires_grad]
    groups = [{"params": head_params, "lr": float(config["train"]["head_lr"])}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(config["train"]["backbone_lr"])})
    return torch.optim.AdamW(groups, weight_decay=float(config["train"]["weight_decay"]))


def autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")


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
def predict(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, Any]:
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
    return {
        "loss": total_loss / total_items,
        "probs": probs_np,
        "labels": labels_np,
        "preds": preds_np,
        "subject_ids": subject_ids_np,
    }


def save_json(obj: object, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def report_to_csv(report: dict[str, Any], class_names: list[str], path: Path) -> None:
    rows = []
    for name in class_names:
        item = report[name]
        rows.append(
            {
                "class": name,
                "precision": item["precision"],
                "recall": item["recall"],
                "f1": item["f1-score"],
                "support": int(item["support"]),
            }
        )
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
    means = probs_df.groupby("subject_id", sort=True).mean().reset_index()
    subject_df = subject_df.merge(means, on="subject_id", how="left")
    subject_df.to_csv(output_dir / "subject_test_predictions.csv", index=False)


def write_model_summary(
    model: USFMClassifier,
    config: dict[str, Any],
    output_path: Path,
    elapsed_seconds: float | None = None,
) -> None:
    total_params, trainable_params = count_parameters(model)
    backbone_params = sum(param.numel() for param in model.backbone.parameters())
    trainable_backbone_params = sum(param.numel() for param in model.backbone.parameters() if param.requires_grad)
    head_params = sum(param.numel() for param in model.head.parameters())
    lines = [
        f"model: {config['model']['name']}",
        f"checkpoint_path: {config['model']['checkpoint_path']}",
        f"adapter_path: {config['model']['adapter_path']}",
        f"input_size: {config['model']['image_size']}x{config['model']['image_size']}",
        "input_channels: 3",
        f"input_mode: {config['model']['input_mode']}",
        f"normalization_mean: {config['model']['normalization']['mean']}",
        f"normalization_std: {config['model']['normalization']['std']}",
        f"global_pool: {config['model']['global_pool']}",
        f"head_type: {config['model']['head_type']}",
        f"dropout: {config['model']['dropout']}",
        f"frozen_backbone: {config['model']['freeze_backbone']}",
        f"unfreeze_last_n_blocks: {config['model']['unfreeze_last_n_blocks']}",
        f"trainable_backbone_modules: {getattr(model, 'trainable_backbone_modules', [])}",
        f"trainable_layers: {config['model']['trainable']}",
        f"feature_dim: {model.feature_dim}",
        f"total_params: {total_params}",
        f"trainable_params: {trainable_params}",
        f"backbone_params: {backbone_params}",
        f"trainable_backbone_params: {trainable_backbone_params}",
        f"head_params: {head_params}",
        f"gpu_name: {config['runtime']['gpu_name']}",
        f"amp_enabled: {config['train']['amp']}",
        f"batch_size: {config['train']['actual_batch_size']}",
        f"effective_batch_size: {config['train']['effective_batch_size']}",
        f"num_workers: {config['train']['num_workers']}",
        f"pin_memory: {config['train']['pin_memory']}",
        f"persistent_workers: {config['train']['persistent_workers']}",
        f"prefetch_factor: {config['train']['prefetch_factor']}",
    ]
    if elapsed_seconds is not None:
        lines.append(f"train_elapsed_seconds: {elapsed_seconds:.2f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def smoke_forward(model: nn.Module, device: torch.device, num_classes: int) -> None:
    model.eval()
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        logits = model(x)
    if tuple(logits.shape) != (2, num_classes):
        raise RuntimeError(f"Smoke forward failed: expected (2, {num_classes}), got {tuple(logits.shape)}")
    print(f"smoke_forward_ok shape={tuple(logits.shape)}")


def prepare_run(config: dict[str, Any], batch_size: int) -> tuple[dict[str, Any], Path, pd.DataFrame]:
    config = deepcopy(config)
    output_dir = resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=False)
    config["output_dir"] = str(output_dir)
    config["train"]["actual_batch_size"] = batch_size
    config["train"]["effective_batch_size"] = batch_size * int(config["train"]["grad_accum_steps"])

    manifest_path = Path(config["data"]["manifest_path"])
    manifest = load_manifest(manifest_path)
    shutil.copy2(manifest_path, output_dir / "manifest.csv")
    split_summary(manifest).to_csv(output_dir / "split_summary.csv", index=False)
    return config, output_dir, manifest


def run_experiment(config: dict[str, Any], batch_size: int) -> Path:
    set_seed(int(config["train"]["seed"]))
    device = resolve_device(str(config["train"]["device"]))
    config, output_dir, manifest = prepare_run(config, batch_size)
    config["runtime"] = {
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name(device),
    }
    save_json(config, output_dir / "config_used.json")
    save_json(config, output_dir / "config.json")
    print(f"run_dir={output_dir}")
    print(f"gpu={config['runtime']['gpu_name']}")

    train_tf, eval_tf = make_transforms(int(config["model"]["image_size"]))
    train_manifest = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    test_manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)
    num_workers = int(config["train"]["num_workers"])
    train_loader = make_loader(train_manifest, train_tf, batch_size, num_workers, True, config)
    val_loader = make_loader(val_manifest, eval_tf, batch_size, num_workers, False, config)
    test_loader = make_loader(test_manifest, eval_tf, batch_size, num_workers, False, config)

    model = build_model(config, device)
    smoke_forward(model, device, len(config["data"]["class_names"]))
    write_model_summary(model, config, output_dir / "model_summary.txt")

    if config["train"]["use_class_weights"]:
        weights = compute_class_weights(train_manifest, len(config["data"]["class_names"]), device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["train"]["epochs"]))
    scaler = torch.amp.GradScaler(device="cuda", enabled=bool(config["train"]["amp"]) and device.type == "cuda")
    amp_enabled = bool(config["train"]["amp"]) and device.type == "cuda"
    grad_accum_steps = int(config["train"]["grad_accum_steps"])
    class_names = config["data"]["class_names"]
    best_val_macro_f1 = -1.0
    best_path = output_dir / "best.pt"
    logs: list[dict[str, float]] = []
    start_time = time.perf_counter()

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
            grad_accum_steps,
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
        pd.DataFrame(logs).to_csv(output_dir / "train_log.csv", index=False)
        if epoch % 5 == 0 or epoch == int(config["train"]["epochs"]):
            generate_report(output_dir, epoch=epoch)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_macro_f1={row['val_macro_f1']:.4f}"
        )
        if row["val_macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = row["val_macro_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "config": config,
                    "best_val_macro_f1": best_val_macro_f1,
                },
                best_path,
            )

    elapsed = time.perf_counter() - start_time
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_pred = predict(model, test_loader, criterion, device, amp_enabled)
    test_image_metrics = image_level_metrics(test_pred["labels"], test_pred["preds"], class_names)
    subject_true, subject_pred, _subject_ids = subject_level_predictions(
        test_pred["probs"], test_pred["labels"], test_pred["subject_ids"]
    )
    test_subject_metrics = image_level_metrics(subject_true, subject_pred, class_names)
    for level, metrics in (("image", test_image_metrics), ("subject", test_subject_metrics)):
        for metric_name in ("accuracy", "balanced_accuracy", "macro_f1"):
            check_finite(float(metrics[metric_name]), f"{level}_{metric_name}")

    metrics_obj = {"image_level": test_image_metrics, "subject_level": test_subject_metrics}
    save_json(metrics_obj, output_dir / "test_metrics.json")
    save_predictions(
        output_dir,
        test_manifest,
        test_pred["probs"],
        test_pred["labels"],
        test_pred["preds"],
        test_pred["subject_ids"],
        class_names,
    )
    report_to_csv(test_image_metrics["classification_report"], class_names, output_dir / "image_classification_report.csv")
    report_to_csv(
        test_subject_metrics["classification_report"],
        class_names,
        output_dir / "subject_classification_report.csv",
    )
    config["runtime"]["train_elapsed_seconds"] = elapsed
    save_json(config, output_dir / "config_used.json")
    save_json(config, output_dir / "config.json")
    write_model_summary(model, config, output_dir / "model_summary.txt", elapsed_seconds=elapsed)
    generate_report(output_dir, epoch=int(config["train"]["epochs"]))
    print(
        f"done run_dir={output_dir} image_macro_f1={test_image_metrics['macro_f1']:.4f} "
        f"subject_macro_f1={test_subject_metrics['macro_f1']:.4f} elapsed={elapsed:.1f}s"
    )
    return output_dir


def run_with_fallback(config: dict[str, Any]) -> Path:
    batch_size = int(config["train"]["batch_size"])
    fallback = int(config["train"]["fallback_batch_size"])
    try:
        return run_experiment(config, batch_size)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" not in message or batch_size <= fallback:
            raise
        print(f"CUDA OOM with batch_size={batch_size}; retrying with batch_size={fallback}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return run_experiment(config, fallback)


def preflight() -> None:
    manifest = Path(FIXED_MANIFEST)
    checkpoint = Path(USFM_CHECKPOINT)
    adapter = Path(USFM_ADAPTER)
    if not manifest.exists():
        raise FileNotFoundError(f"Fixed manifest not found: {manifest}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"USFM checkpoint not found: {checkpoint}")
    if not adapter.exists():
        raise FileNotFoundError(f"USFM adapter not found: {adapter}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; requested device=cuda with RTX 4090 priority.")
    print(f"cuda_available=True gpu0={torch.cuda.get_device_name(0)}")
    for name in ("mlp_probe", "partial_last_block"):
        config = make_config(name)
        print(f"{name}_output_dir={resolve_output_dir(config)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["all", "mlp_probe", "partial_last_block"],
        default="all",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preflight()
    names = ["mlp_probe", "partial_last_block"] if args.experiment == "all" else [args.experiment]
    for name in names:
        config = make_config(name)
        if args.num_workers is not None:
            config["train"]["num_workers"] = args.num_workers
        if args.epochs is not None:
            config["train"]["epochs"] = args.epochs
        print(f"starting_experiment={name}")
        run_with_fallback(config)


if __name__ == "__main__":
    main()
