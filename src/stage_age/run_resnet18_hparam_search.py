from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn

from stage_age.data import make_transforms
from stage_age.metrics import image_level_metrics, subject_level_predictions
from stage_age.models import build_model as build_torchvision_model
from stage_age.report import generate_report
from stage_age.run_multiseed_experiments import (
    BASE_CONFIG,
    build_subject_manifest,
    check_finite,
    compute_class_weights,
    deep_update,
    gpu_name,
    make_loader,
    predict,
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


CONFIGS: dict[str, dict[str, Any]] = {
    "resnet18_reg_a": {
        "train": {
            "lr": 1e-4,
            "weight_decay": 5e-4,
            "label_smoothing": 0.05,
            "finetune_strategy": "full_finetune",
        },
        "model": {"trainable": "full fine-tune"},
    },
    "resnet18_reg_b": {
        "train": {
            "lr": 1e-4,
            "weight_decay": 1e-3,
            "label_smoothing": 0.05,
            "finetune_strategy": "full_finetune",
        },
        "model": {"trainable": "full fine-tune"},
    },
    "resnet18_freeze12": {
        "train": {
            "lr": 3e-4,
            "weight_decay": 5e-4,
            "label_smoothing": 0.05,
            "finetune_strategy": "train_layer3_layer4_fc",
        },
        "model": {"trainable": "layer3 + layer4 + fc"},
    },
    "resnet18_discriminative_lr": {
        "train": {
            "backbone_lr": 1e-5,
            "head_lr": 3e-4,
            "weight_decay": 5e-4,
            "label_smoothing": 0.05,
            "finetune_strategy": "discriminative_full_finetune",
        },
        "model": {"trainable": "full fine-tune with backbone/head param groups"},
    },
    "resnet18_layer4_only": {
        "train": {
            "layer4_lr": 3e-5,
            "head_lr": 3e-4,
            "weight_decay": 5e-4,
            "label_smoothing": 0.05,
            "finetune_strategy": "train_layer4_fc",
        },
        "model": {"trainable": "layer4 + fc"},
    },
    "resnet18_layer4_only_strongwd": {
        "train": {
            "layer4_lr": 3e-5,
            "head_lr": 3e-4,
            "weight_decay": 1e-3,
            "label_smoothing": 0.05,
            "finetune_strategy": "train_layer4_fc",
        },
        "model": {"trainable": "layer4 + fc"},
    },
}


def base_resnet_config(config_name: str) -> dict[str, Any]:
    if config_name not in CONFIGS:
        raise ValueError(f"Unsupported config: {config_name}")
    config = deep_update(
        BASE_CONFIG,
        {
            "model": {
                "name": config_name,
                "backbone": "resnet18",
                "pretrained": "ImageNet",
                "num_classes": 3,
                "image_size": 224,
                "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            },
            "train": {
                "epochs": 30,
                "early_stopping_patience": 7,
                "batch_size": 64,
                "fallback_batch_size": 32,
                "use_class_weights": True,
                "amp": True,
                "weight_decay": 5e-4,
                "label_smoothing": 0.05,
                "save_best_by": "val_macro_f1",
                "test_checkpoint": "best.pt",
                "evaluated_checkpoint": "best.pt",
                "use_best_checkpoint_for_test": True,
            },
        },
    )
    return deep_update(config, CONFIGS[config_name])


def check_preflight() -> None:
    for path in [Path(BASE_CONFIG["data"]["image_dir"]), Path(BASE_CONFIG["data"]["characteristics"])]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    torch.backends.cudnn.benchmark = True
    print(f"cuda_available=True gpu0={torch.cuda.get_device_name(0)}")


def set_trainable(model: nn.Module, strategy: str) -> list[str]:
    for param in model.parameters():
        param.requires_grad = False

    if strategy in {"full_finetune", "discriminative_full_finetune"}:
        for param in model.parameters():
            param.requires_grad = True
        return ["all"]

    trainable_modules: list[str]
    if strategy == "train_layer3_layer4_fc":
        trainable_modules = ["layer3", "layer4", "fc"]
    elif strategy == "train_layer4_fc":
        trainable_modules = ["layer4", "fc"]
    else:
        raise ValueError(f"Unsupported finetune_strategy: {strategy}")

    for module_name in trainable_modules:
        module = getattr(model, module_name)
        for param in module.parameters():
            param.requires_grad = True
    return trainable_modules


def build_resnet(config: dict[str, Any], device: torch.device) -> nn.Module:
    model = build_torchvision_model("resnet18", num_classes=len(config["data"]["class_names"]), pretrained=True)
    trainable = set_trainable(model, str(config["train"]["finetune_strategy"]))
    model.trainable_modules = trainable  # type: ignore[attr-defined]
    return model.to(device)


def make_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    strategy = str(config["train"]["finetune_strategy"])
    weight_decay = float(config["train"]["weight_decay"])
    if strategy == "discriminative_full_finetune":
        head_params = [p for p in model.fc.parameters() if p.requires_grad]
        head_ids = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": float(config["train"]["backbone_lr"]), "name": "backbone"},
                {"params": head_params, "lr": float(config["train"]["head_lr"]), "name": "head"},
            ],
            weight_decay=weight_decay,
        )
    if strategy == "train_layer4_fc":
        layer4_params = [p for p in model.layer4.parameters() if p.requires_grad]
        fc_params = [p for p in model.fc.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            [
                {"params": layer4_params, "lr": float(config["train"]["layer4_lr"]), "name": "layer4"},
                {"params": fc_params, "lr": float(config["train"]["head_lr"]), "name": "head"},
            ],
            weight_decay=weight_decay,
        )
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=float(config["train"]["lr"]), weight_decay=weight_decay)


def param_summary(model: nn.Module) -> dict[str, Any]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_modules": getattr(model, "trainable_modules", []),
    }


def append_resnet_summary(model: nn.Module, config: dict[str, Any], path: Path) -> None:
    summary = param_summary(model)
    extra = [
        f"finetune_strategy: {config['train']['finetune_strategy']}",
        f"trainable_modules: {summary['trainable_modules']}",
        f"label_smoothing: {config['train']['label_smoothing']}",
        f"early_stopping_patience: {config['train']['early_stopping_patience']}",
        f"total_params_resnet: {summary['total_params']}",
        f"trainable_params_resnet: {summary['trainable_params']}",
    ]
    with path.open("a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(extra) + "\n")


def class_subject_metrics(subject_report: dict[str, Any], class_names: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in class_names:
        key = name.replace("-", "_")
        values[f"subject_f1_{key}"] = float(subject_report[name]["f1-score"])
        values[f"subject_recall_{key}"] = float(subject_report[name]["recall"])
    return values


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
    train_loader = make_loader(train_manifest, train_tf, batch_size, num_workers, True, config)
    val_loader = make_loader(val_manifest, eval_tf, batch_size, num_workers, False, config)
    test_loader = make_loader(test_manifest, eval_tf, batch_size, num_workers, False, config)

    model = build_resnet(config, device)
    smoke_forward(model, device, len(config["data"]["class_names"]))
    write_model_summary(model, config, seed_dir / "model_summary.txt")
    append_resnet_summary(model, config, seed_dir / "model_summary.txt")

    weights = compute_class_weights(train_manifest, len(config["data"]["class_names"]), device) if config["train"]["use_class_weights"] else None
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=float(config["train"]["label_smoothing"]))
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["train"]["epochs"]))
    amp_enabled = bool(config["train"]["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=amp_enabled)
    best_val_macro_f1 = -1.0
    best_epoch = -1
    stale_epochs = 0
    stop_reason = "max_epochs"
    logs: list[dict[str, float | int | str]] = []
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
        check_finite(float(train_stats["loss"]), "train_loss")
        check_finite(float(val_pred["loss"]), "val_loss")
        check_finite(float(val_metrics["macro_f1"]), "val_macro_f1")
        scheduler.step()
        current_lrs = scheduler.get_last_lr()
        row = {
            "epoch": epoch,
            "lr": float(current_lrs[0]),
            "train_loss": float(train_stats["loss"]),
            "train_accuracy": float(train_stats["accuracy"]),
            "val_loss": float(val_pred["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        }
        logs.append(row)
        pd.DataFrame(logs).to_csv(seed_dir / "train_log.csv", index=False)
        if epoch % 5 == 0 or epoch == int(config["train"]["epochs"]):
            generate_report(seed_dir, epoch=epoch)
        print(f"config={config['model']['name']} seed={seed} epoch={epoch:03d} val_macro_f1={row['val_macro_f1']:.4f}")
        if row["val_macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = float(row["val_macro_f1"])
            best_epoch = epoch
            stale_epochs = 0
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
        else:
            stale_epochs += 1
            if stale_epochs >= int(config["train"]["early_stopping_patience"]):
                stop_reason = f"early_stopping_patience_{config['train']['early_stopping_patience']}"
                break

    elapsed = time.perf_counter() - start
    if best_epoch < 0:
        raise RuntimeError("No best checkpoint was saved.")
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
    config["runtime"]["stop_reason"] = stop_reason
    config["train"]["epochs_ran"] = int(logs[-1]["epoch"])
    save_json(config, seed_dir / "config_used.json")
    save_json(config, seed_dir / "config.json")
    write_model_summary(model, config, seed_dir / "model_summary.txt", elapsed)
    append_resnet_summary(model, config, seed_dir / "model_summary.txt")
    generate_report(seed_dir, epoch=int(logs[-1]["epoch"]))

    row = {
        "config_name": config["model"]["name"],
        "model": "resnet18",
        "seed": seed,
        "seed_run_dir": str(seed_dir),
        "pretrained": config["model"]["pretrained"],
        "best_epoch": best_epoch,
        "epochs_ran": int(logs[-1]["epoch"]),
        "stop_reason": stop_reason,
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
                shutil.rmtree(seed_dir)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return run_seed(config, seed, seed_dir, fallback_batch_size, num_workers)
        if ("worker" in msg or "dataloader" in msg) and num_workers > fallback_num_workers:
            if seed_dir.exists():
                shutil.rmtree(seed_dir)
            return run_seed(config, seed, seed_dir, batch_size, fallback_num_workers)
        raise
    except FloatingPointError as exc:
        failure_path = seed_dir.parent / "failure.json"
        save_json({"config_name": config["model"]["name"], "seed": seed, "reason": str(exc)}, failure_path)
        raise


def aggregate_rows(rows: list[dict[str, Any]], class_names: list[str]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    df = pd.DataFrame(rows)
    metrics = ["subject_macro_f1", "subject_balanced_accuracy", "subject_f1_45_64"]
    stats = {}
    for metric in metrics:
        stats[f"{metric}_mean"] = float(df[metric].mean())
        stats[f"{metric}_std"] = float(df[metric].std(ddof=1)) if len(df) > 1 else 0.0
    for row in rows:
        row.update(stats)
    return rows


def write_summary(root_dir: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    rows = aggregate_rows(rows, config["data"]["class_names"])
    df = pd.DataFrame(rows)
    df.to_csv(root_dir / "summary.csv", index=False)
    save_json(config, root_dir / "config_multiseed.json")
    cols = [
        "config_name",
        "seed",
        "best_epoch",
        "epochs_ran",
        "best_val_macro_f1",
        "image_macro_f1",
        "subject_macro_f1",
        "subject_balanced_accuracy",
        "subject_f1_18_44",
        "subject_recall_18_44",
        "subject_f1_45_64",
        "subject_recall_45_64",
        "subject_f1_65_100",
        "subject_recall_65_100",
    ]
    view = df[cols]
    table_lines = [
        "| " + " | ".join(view.columns) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for record in view.to_dict(orient="records"):
        values = []
        for col in view.columns:
            value = record[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        table_lines.append("| " + " | ".join(values) + " |")
    lines = [
        f"# {config['model']['name']} Summary",
        "",
        "\n".join(table_lines),
        "",
        f"subject_macro_f1_mean: {df['subject_macro_f1'].mean():.4f}",
        f"subject_macro_f1_std: {df['subject_macro_f1'].std(ddof=1):.4f}",
        f"subject_balanced_accuracy_mean: {df['subject_balanced_accuracy'].mean():.4f}",
        f"subject_balanced_accuracy_std: {df['subject_balanced_accuracy'].std(ddof=1):.4f}",
        f"45-64_subject_f1_mean: {df['subject_f1_45_64'].mean():.4f}",
        f"45-64_subject_f1_std: {df['subject_f1_45_64'].std(ddof=1):.4f}",
        "",
        "All test metrics are evaluated from best.pt selected by highest val_macro_f1.",
    ]
    (root_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_config(config_name: str, seeds: list[int], epochs: int | None = None) -> Path:
    check_preflight()
    config = base_resnet_config(config_name)
    if epochs is not None:
        config["train"]["epochs"] = epochs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = unique_output_dir(Path("outputs") / f"{timestamp}_{config_name}_age45_65_hparam")
    root_dir.mkdir(parents=True, exist_ok=False)
    config["hparam_search"] = {"config_name": config_name, "seeds": seeds, "root_dir": str(root_dir)}
    save_json(config, root_dir / "config_multiseed.json")

    device = resolve_device(config["train"]["device"])
    probe = build_resnet(config, device)
    smoke_forward(probe, device, len(config["data"]["class_names"]))
    del probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"starting config={config_name} seed={seed}")
        row = run_seed_with_fallback(config, seed, root_dir / f"seed{seed}")
        rows.append(row)
        write_summary(root_dir, rows, config)
    return root_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", choices=sorted(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = []
    for config_name in args.configs:
        root = run_config(config_name, args.seeds, epochs=args.epochs)
        print(f"hparam_run_dir={root}")
        roots.append(str(root))
    print("completed_hparam_run_dirs=" + json.dumps(roots, ensure_ascii=False))


if __name__ == "__main__":
    main()
