from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage_age.config import add_config_args, apply_cli_overrides, load_config
from stage_age.data import UltrasoundAgeDataset, build_manifest, make_transforms
from stage_age.metrics import image_level_metrics, subject_level_predictions
from stage_age.models import build_model
from stage_age.report import generate_report, plot_training_curves


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def make_loader(
    manifest: pd.DataFrame,
    transform,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = UltrasoundAgeDataset(manifest, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def compute_class_weights(train_manifest: pd.DataFrame, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = train_manifest["label"].value_counts().reindex(range(num_classes), fill_value=0).to_numpy()
    if (counts == 0).any():
        raise ValueError(f"At least one class is missing from train split: {counts.tolist()}")
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for images, labels, _subject_ids in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_items += batch_size
        progress.set_postfix(loss=total_loss / total_items, acc=total_correct / total_items)
    return {"loss": total_loss / total_items, "accuracy": total_correct / total_items}


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_subject_ids: list[np.ndarray] = []
    for images, labels, subject_ids in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)

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


def summarize_split(manifest: pd.DataFrame) -> pd.DataFrame:
    subject_summary = manifest.drop_duplicates("subject_id")
    return (
        subject_summary.groupby(["split", "class_name"])
        .size()
        .rename("subjects")
        .reset_index()
        .merge(
            manifest.groupby(["split", "class_name"]).size().rename("images").reset_index(),
            on=["split", "class_name"],
        )
    )


def resolve_output_dir(config: dict[str, Any]) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "output_dir" in config:
        legacy_output = Path(config["output_dir"])
        return legacy_output.parent / f"{timestamp}_{legacy_output.name}"

    output_root = Path(config.get("output_root", "outputs"))
    run_name = str(config.get("run_name", "run"))
    return output_root / f"{timestamp}_{run_name}"


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser())
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config), args)

    set_seed(int(config["train"]["seed"]))
    output_dir = resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    config["resolved_output_dir"] = str(output_dir)
    save_json(config, output_dir / "config.json")

    manifest = build_manifest(
        image_dir=config["data"]["image_dir"],
        characteristics=config["data"]["characteristics"],
        sheet_name=config["data"]["sheet_name"],
        bins=config["data"]["bins"],
        class_names=config["data"]["class_names"],
        split=config["data"]["split"],
        seed=int(config["train"]["seed"]),
    )
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summarize_split(manifest).to_csv(output_dir / "split_summary.csv", index=False)

    train_tf, eval_tf = make_transforms(int(config["train"]["image_size"]))
    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"]["num_workers"])
    train_loader = make_loader(manifest[manifest["split"] == "train"], train_tf, batch_size, num_workers, True)
    val_loader = make_loader(manifest[manifest["split"] == "val"], eval_tf, batch_size, num_workers, False)
    test_loader = make_loader(manifest[manifest["split"] == "test"], eval_tf, batch_size, num_workers, False)

    device = resolve_device(str(config["train"]["device"]))
    model = build_model(
        name=config["model"]["name"],
        num_classes=int(config["model"]["num_classes"]),
        pretrained=bool(config["model"]["pretrained"]),
    ).to(device)

    if config["train"]["use_class_weights"]:
        weights = compute_class_weights(manifest[manifest["split"] == "train"], int(config["model"]["num_classes"]), device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["train"]["epochs"]))

    logs: list[dict[str, float]] = []
    best_val_macro_f1 = -1.0
    best_path = output_dir / "best.pt"
    class_names = config["data"]["class_names"]

    for epoch in range(1, int(config["train"]["epochs"]) + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_pred = predict(model, val_loader, criterion, device)
        val_metrics = image_level_metrics(val_pred["labels"], val_pred["preds"], class_names)
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
            plot_training_curves(output_dir, epoch=epoch)
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

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_pred = predict(model, test_loader, criterion, device)
    test_image_metrics = image_level_metrics(test_pred["labels"], test_pred["preds"], class_names)
    subj_true, subj_pred, subj_ids = subject_level_predictions(
        test_pred["probs"],
        test_pred["labels"],
        test_pred["subject_ids"],
    )
    test_subject_metrics = image_level_metrics(subj_true, subj_pred, class_names)

    save_json({"image_level": test_image_metrics, "subject_level": test_subject_metrics}, output_dir / "test_metrics.json")
    pd.DataFrame(
        {
            "subject_id": subj_ids,
            "label": subj_true,
            "pred": subj_pred,
            "label_name": [class_names[i] for i in subj_true],
            "pred_name": [class_names[i] for i in subj_pred],
        }
    ).to_csv(output_dir / "test_subject_predictions.csv", index=False)
    generate_report(output_dir, epoch=int(config["train"]["epochs"]))
    print(f"best_checkpoint={best_path}")
    print(
        f"test image_acc={test_image_metrics['accuracy']:.4f} "
        f"subject_acc={test_subject_metrics['accuracy']:.4f} "
        f"subject_macro_f1={test_subject_metrics['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
