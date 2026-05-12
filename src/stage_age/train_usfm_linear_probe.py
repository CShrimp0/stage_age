from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from stage_age.data import UltrasoundAgeDataset, make_transforms
from stage_age.metrics import image_level_metrics, subject_level_predictions
from stage_age.report import generate_report
from stage_age.usfm import DEFAULT_USFM_ADAPTER_PATH, USFMLinearProbe, count_parameters


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "manifest_path": "outputs/20260511_152330_resnet18_age45_65/manifest.csv",
        "image_dir": "/home/szdx/LNX/data/TA/Healthy/Images",
        "characteristics": "/home/szdx/LNX/data/TA/characteristics.xlsx",
        "sheet_name": "Blad1",
        "bins": [18, 45, 65, 101],
        "class_names": ["18-44", "45-64", "65-100"],
    },
    "model": {
        "name": "usfm_linear_probe",
        "pretrained": "/home/szdx/LNX/stage-age/USFM_latest.pth",
        "checkpoint_path": "/home/szdx/LNX/stage-age/USFM_latest.pth",
        "adapter_path": str(DEFAULT_USFM_ADAPTER_PATH),
        "image_size": 224,
        "input_channels": 3,
        "global_pool": "token",
        "freeze_backbone": True,
        "trainable": "LayerNorm + Linear classification head",
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    },
    "train": {
        "epochs": 30,
        "batch_size": 32,
        "fallback_batch_size": 16,
        "num_workers": 4,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "use_class_weights": True,
        "seed": 42,
        "device": "cuda",
        "save_best_by": "val_macro_f1",
    },
    "output_root": "outputs",
    "run_name": "usfm_linear_probe_age45_65",
    "resnet_run_dir": "outputs/20260511_152330_resnet18_age45_65",
}


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
    return unique_output_dir(Path(config.get("output_root", "outputs")) / f"{timestamp}_{config.get('run_name', 'usfm_linear_probe')}")


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


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required ResNet manifest not found: {path}")
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
) -> DataLoader:
    return DataLoader(
        UltrasoundAgeDataset(manifest, transform),
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
    prob_cols = {f"prob_{class_names[idx]}": probs[:, idx] for idx in range(len(class_names))}
    image_df = test_manifest.reset_index(drop=True).copy()
    image_df["label"] = labels
    image_df["pred"] = preds
    image_df["label_name"] = [class_names[idx] for idx in labels]
    image_df["pred_name"] = [class_names[idx] for idx in preds]
    for col, values in prob_cols.items():
        image_df[col] = values
    image_df.to_csv(output_dir / "image_test_predictions.csv", index=False)

    subject_true, subject_pred, subject_ids_out = subject_level_predictions(probs, labels, subject_ids)
    subject_df = pd.DataFrame({"subject_id": subject_ids_out, "label": subject_true, "pred": subject_pred})
    subject_df["label_name"] = [class_names[idx] for idx in subject_true]
    subject_df["pred_name"] = [class_names[idx] for idx in subject_pred]
    for idx, name in enumerate(class_names):
        means = (
            pd.DataFrame({"subject_id": subject_ids, f"prob_{name}": probs[:, idx]})
            .groupby("subject_id", sort=True)[f"prob_{name}"]
            .mean()
            .to_numpy()
        )
        subject_df[f"prob_{name}"] = means
    subject_df.to_csv(output_dir / "subject_test_predictions.csv", index=False)


def write_model_summary(model: USFMLinearProbe, config: dict[str, Any], output_path: Path) -> None:
    total_params, trainable_params = count_parameters(model)
    backbone_params = sum(param.numel() for param in model.backbone.parameters())
    head_params = sum(param.numel() for param in model.head.parameters())
    lines = [
        "model: usfm_linear_probe",
        f"checkpoint_path: {config['model']['checkpoint_path']}",
        f"adapter_path: {config['model']['adapter_path']}",
        f"input_size: {config['model']['image_size']}x{config['model']['image_size']}",
        "input_channels: 3",
        "input_mode: grayscale images converted to RGB",
        f"normalization_mean: {config['model']['normalization']['mean']}",
        f"normalization_std: {config['model']['normalization']['std']}",
        f"global_pool: {config['model']['global_pool']}",
        "frozen_layers: full USFM backbone / encoder",
        "trainable_layers: LayerNorm + Linear classification head",
        f"feature_dim: {model.feature_dim}",
        f"total_params: {total_params}",
        f"trainable_params: {trainable_params}",
        f"backbone_params: {backbone_params}",
        f"head_params: {head_params}",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def smoke_test(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> None:
    model.eval()
    images, _labels, _subject_ids = next(iter(loader))
    images = images[: min(2, images.size(0))].to(device)
    with torch.no_grad():
        logits = model(images)
    if tuple(logits.shape) != (images.size(0), num_classes):
        raise RuntimeError(f"Unexpected USFM linear probe output shape: {tuple(logits.shape)}")


def run_training(config: dict[str, Any], batch_size: int) -> Path:
    set_seed(int(config["train"]["seed"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(json.dumps(config))
    config["output_dir"] = str(output_dir)
    config["train"]["actual_batch_size"] = batch_size
    config["train"]["effective_batch_size"] = batch_size
    config["train"]["save_best_by"] = "val_macro_f1"
    config["train"]["test_checkpoint"] = "best.pt"
    config["train"]["evaluated_checkpoint"] = "best.pt"
    config["train"]["use_best_checkpoint_for_test"] = True

    manifest_path = Path(config["data"]["manifest_path"])
    manifest = load_manifest(manifest_path)
    shutil.copy2(manifest_path, output_dir / "manifest.csv")
    split_summary(manifest).to_csv(output_dir / "split_summary.csv", index=False)

    save_json(config, output_dir / "config_used.json")
    save_json(config, output_dir / "config.json")

    train_tf, eval_tf = make_transforms(int(config["model"]["image_size"]))
    num_workers = int(config["train"]["num_workers"])
    train_manifest = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_manifest = manifest[manifest["split"] == "val"].reset_index(drop=True)
    test_manifest = manifest[manifest["split"] == "test"].reset_index(drop=True)

    train_loader = make_loader(train_manifest, train_tf, batch_size, num_workers, True)
    val_loader = make_loader(val_manifest, eval_tf, batch_size, num_workers, False)
    test_loader = make_loader(test_manifest, eval_tf, batch_size, num_workers, False)

    device = resolve_device(str(config["train"]["device"]))
    model = USFMLinearProbe(
        checkpoint_path=config["model"]["checkpoint_path"],
        adapter_path=config["model"]["adapter_path"],
        image_size=int(config["model"]["image_size"]),
        global_pool=str(config["model"]["global_pool"]),
        num_classes=len(config["data"]["class_names"]),
    ).to(device)
    write_model_summary(model, config, output_dir / "model_summary.txt")
    smoke_test(model, train_loader, device, len(config["data"]["class_names"]))

    if config["train"]["use_class_weights"]:
        weights = compute_class_weights(train_manifest, len(config["data"]["class_names"]), device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
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

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_pred = predict(model, test_loader, criterion, device)
    test_image_metrics = image_level_metrics(test_pred["labels"], test_pred["preds"], class_names)
    subject_true, subject_pred, _subject_ids = subject_level_predictions(
        test_pred["probs"], test_pred["labels"], test_pred["subject_ids"]
    )
    test_subject_metrics = image_level_metrics(subject_true, subject_pred, class_names)
    metrics = {"image_level": test_image_metrics, "subject_level": test_subject_metrics}
    save_json(metrics, output_dir / "test_metrics.json")

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
    generate_report(output_dir, epoch=int(config["train"]["epochs"]))
    write_comparison(config["resnet_run_dir"], output_dir, manifest_path)
    print(f"best_checkpoint={best_path}")
    print(
        f"test image_acc={test_image_metrics['accuracy']:.4f} "
        f"subject_acc={test_subject_metrics['accuracy']:.4f} "
        f"subject_macro_f1={test_subject_metrics['macro_f1']:.4f}"
    )
    return output_dir


def run_forward_smoke(config: dict[str, Any], batch_size: int) -> None:
    set_seed(int(config["train"]["seed"]))
    manifest = load_manifest(config["data"]["manifest_path"])
    train_manifest = manifest[manifest["split"] == "train"].reset_index(drop=True)
    _train_tf, eval_tf = make_transforms(int(config["model"]["image_size"]))
    loader = make_loader(
        train_manifest,
        eval_tf,
        min(batch_size, 2),
        int(config["train"]["num_workers"]),
        False,
    )
    device = resolve_device(str(config["train"]["device"]))
    model = USFMLinearProbe(
        checkpoint_path=config["model"]["checkpoint_path"],
        adapter_path=config["model"]["adapter_path"],
        image_size=int(config["model"]["image_size"]),
        global_pool=str(config["model"]["global_pool"]),
        num_classes=len(config["data"]["class_names"]),
    ).to(device)
    smoke_test(model, loader, device, len(config["data"]["class_names"]))
    total_params, trainable_params = count_parameters(model)
    print("smoke_forward_ok")
    print(f"output_shape=(2,{len(config['data']['class_names'])})")
    print(f"total_params={total_params}")
    print(f"trainable_params={trainable_params}")


def best_val_macro_f1(run_dir: str | Path) -> float:
    log = pd.read_csv(Path(run_dir) / "train_log.csv")
    return float(log["val_macro_f1"].max())


def load_metrics(run_dir: str | Path) -> dict[str, Any]:
    with (Path(run_dir) / "test_metrics.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def metrics_row(name: str, run_dir: Path, metrics: dict[str, Any]) -> list[object]:
    return [
        name,
        str(run_dir),
        metrics["image_level"]["accuracy"],
        metrics["image_level"]["balanced_accuracy"],
        metrics["image_level"]["macro_f1"],
        metrics["subject_level"]["accuracy"],
        metrics["subject_level"]["balanced_accuracy"],
        metrics["subject_level"]["macro_f1"],
        best_val_macro_f1(run_dir),
    ]


def markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(item) for item in row) + " |")
    return "\n".join(lines)


def write_comparison(resnet_run_dir: str | Path, usfm_run_dir: str | Path, manifest_path: str | Path) -> None:
    resnet_run_dir = Path(resnet_run_dir)
    usfm_run_dir = Path(usfm_run_dir)
    if not resnet_run_dir.exists():
        raise FileNotFoundError(f"ResNet run directory not found: {resnet_run_dir}")

    resnet_metrics = load_metrics(resnet_run_dir)
    usfm_metrics = load_metrics(usfm_run_dir)
    delta = usfm_metrics["subject_level"]["macro_f1"] - resnet_metrics["subject_level"]["macro_f1"]
    conclusion = "USFM linear probe 优于 ResNet18 baseline。" if delta > 0 else "USFM linear probe 未优于 ResNet18 baseline。"

    resnet_manifest = Path(manifest_path)
    usfm_manifest = usfm_run_dir / "manifest.csv"
    same_manifest = resnet_manifest.exists() and usfm_manifest.exists() and resnet_manifest.read_bytes() == usfm_manifest.read_bytes()

    rows = [
        metrics_row("ResNet18 age45_65", resnet_run_dir, resnet_metrics),
        metrics_row("USFM linear probe", usfm_run_dir, usfm_metrics),
    ]
    headers = [
        "model",
        "run_dir",
        "image_acc",
        "image_bal_acc",
        "image_macro_f1",
        "subject_acc",
        "subject_bal_acc",
        "subject_macro_f1",
        "best_val_macro_f1",
    ]
    content = [
        "# ResNet18 vs USFM",
        "",
        markdown_table(rows, headers),
        "",
        f"subject_macro_f1_delta_usfm_minus_resnet18: {delta:.4f}",
        f"same_manifest_csv: {same_manifest}",
        f"manifest_csv: {resnet_manifest}",
        f"conclusion: {conclusion}",
        "",
        "## Figures",
        "",
        f"- ResNet18: {resnet_run_dir / 'figures' / 'confusion_matrices.png'}",
        f"- USFM: {usfm_run_dir / 'figures' / 'confusion_matrices.png'}",
    ]
    Path("outputs/comparison_resnet18_vs_usfm.md").write_text("\n".join(content) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--manifest-path", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.manifest_path is not None:
        config["data"]["manifest_path"] = args.manifest_path
    if args.checkpoint_path is not None:
        config["model"]["checkpoint_path"] = args.checkpoint_path
    if args.adapter_path is not None:
        config["model"]["adapter_path"] = args.adapter_path

    if args.smoke_only:
        run_forward_smoke(config, min(int(config["train"]["batch_size"]), 2))
        return

    config["output_dir"] = str(resolve_output_dir(config))

    batch_size = int(config["train"]["batch_size"])
    try:
        run_training(config, batch_size)
    except RuntimeError as exc:
        message = str(exc).lower()
        fallback = int(config["train"]["fallback_batch_size"])
        if "out of memory" not in message or batch_size <= fallback:
            raise
        print(f"CUDA OOM with batch_size={batch_size}; retrying with batch_size={fallback}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_training(config, fallback)


if __name__ == "__main__":
    main()
