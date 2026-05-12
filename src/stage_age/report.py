from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(rows: list[list[object]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(item) for item in row) + " |")
    return "\n".join(lines)


def _pretrained_label(config: dict[str, Any]) -> str:
    model_cfg = config.get("model", {})
    value = model_cfg.get("pretrained")
    if value not in (None, ""):
        if value is True and str(model_cfg.get("name", "")).lower().startswith("resnet"):
            return "ImageNet"
        return str(value)
    checkpoint = model_cfg.get("checkpoint_path")
    if checkpoint:
        return str(checkpoint)
    return ""


def plot_training_curves(run_dir: str | Path, epoch: int | None = None) -> Path | None:
    run_dir = Path(run_dir)
    train_log_path = run_dir / "train_log.csv"
    if not train_log_path.exists():
        return None

    log = pd.read_csv(train_log_path)
    if log.empty:
        return None
    if epoch is not None:
        log = log[log["epoch"] <= epoch].copy()
        if log.empty:
            return None

    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    axes = axes.flatten()

    axes[0].plot(log["epoch"], log["train_loss"], label="train_loss")
    axes[0].plot(log["epoch"], log["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(log["epoch"], log["train_accuracy"], label="train_accuracy")
    axes[1].plot(log["epoch"], log["val_accuracy"], label="val_accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(log["epoch"], log["val_balanced_accuracy"], label="val_balanced_accuracy")
    axes[2].plot(log["epoch"], log["val_macro_f1"], label="val_macro_f1")
    axes[2].set_title("Validation Metrics")
    axes[2].set_xlabel("epoch")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    axes[3].plot(log["epoch"], log["lr"], label="lr")
    axes[3].set_title("Learning Rate")
    axes[3].set_xlabel("epoch")
    axes[3].grid(alpha=0.3)
    axes[3].legend()

    fig.tight_layout()
    latest_path = figures_dir / "training_curves_latest.png"
    fig.savefig(latest_path)
    plt.close(fig)
    return latest_path


def _metrics_rows(metrics: dict[str, Any], section: str) -> list[list[object]]:
    item = metrics[section]
    return [
        [section, "accuracy", item["accuracy"]],
        [section, "balanced_accuracy", item["balanced_accuracy"]],
        [section, "macro_f1", item["macro_f1"]],
    ]


def _class_report_rows(report: dict[str, Any], class_names: list[str]) -> list[list[object]]:
    rows = []
    for name in class_names:
        item = report[name]
        rows.append([name, item["precision"], item["recall"], item["f1-score"], int(item["support"])])
    return rows


def plot_confusion_matrix(
    matrix: list[list[int]],
    class_names: list[str],
    title: str,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(matrix, dtype=int)

    fig, ax = plt.subplots(figsize=(5.8, 4.8), dpi=160)
    im = ax.imshow(values, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)

    threshold = values.max() / 2 if values.size and values.max() > 0 else 0
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            color = "white" if values[row, col] > threshold else "black"
            ax.text(col, row, str(values[row, col]), ha="center", va="center", color=color)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_combined_confusion_matrices(
    image_matrix: list[list[int]],
    subject_matrix: list[list[int]],
    class_names: list[str],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrices = [
        ("Image Level", np.asarray(image_matrix, dtype=int)),
        ("Subject Level", np.asarray(subject_matrix, dtype=int)),
    ]
    vmax = max(int(values.max()) for _title, values in matrices)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=160, constrained_layout=True)
    for ax, (title, values) in zip(axes, matrices):
        im = ax.imshow(values, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(np.arange(len(class_names)), labels=class_names)
        ax.set_yticks(np.arange(len(class_names)), labels=class_names)

        threshold = vmax / 2 if vmax > 0 else 0
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                color = "white" if values[row, col] > threshold else "black"
                ax.text(col, row, str(values[row, col]), ha="center", va="center", color=color)

    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_confusion_matrices(run_dir: str | Path, metrics: dict[str, Any], class_names: list[str]) -> dict[str, Path]:
    run_dir = Path(run_dir)
    figures_dir = run_dir / "figures"
    return {
        "image_level": plot_confusion_matrix(
            metrics["image_level"]["confusion_matrix"],
            class_names,
            "Image Level Confusion Matrix",
            figures_dir / "confusion_matrix_image_level.png",
        ),
        "subject_level": plot_confusion_matrix(
            metrics["subject_level"]["confusion_matrix"],
            class_names,
            "Subject Level Confusion Matrix",
            figures_dir / "confusion_matrix_subject_level.png",
        ),
        "combined": plot_combined_confusion_matrices(
            metrics["image_level"]["confusion_matrix"],
            metrics["subject_level"]["confusion_matrix"],
            class_names,
            figures_dir / "confusion_matrices.png",
        ),
    }


def write_result_markdown(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    config = _load_json(run_dir / "config.json") or _load_json(run_dir / "config_used.json") or {}
    metrics = _load_json(run_dir / "test_metrics.json")
    train_log_path = run_dir / "train_log.csv"
    split_summary_path = run_dir / "split_summary.csv"

    train_log = pd.read_csv(train_log_path) if train_log_path.exists() else pd.DataFrame()
    split_summary = pd.read_csv(split_summary_path) if split_summary_path.exists() else pd.DataFrame()
    class_names = config.get("data", {}).get("class_names", ["18-44", "45-59", "60-100"])

    lines: list[str] = ["# Result", ""]
    lines.extend(
        [
            "## Run",
            "",
            _markdown_table(
                [
                    ["run_dir", str(run_dir)],
                    ["model", config.get("model", {}).get("name", "")],
                    ["pretrained", _pretrained_label(config)],
                    ["epochs", config.get("train", {}).get("epochs", "")],
                    ["batch_size", config.get("train", {}).get("batch_size", "")],
                    ["lr", config.get("train", {}).get("lr", "")],
                    ["seed", config.get("train", {}).get("seed", "")],
                    ["bins", config.get("data", {}).get("bins", "")],
                ],
                ["item", "value"],
            ),
            "",
        ]
    )

    if not split_summary.empty:
        lines.extend(
            [
                "## Split",
                "",
                _markdown_table(split_summary.values.tolist(), split_summary.columns.tolist()),
                "",
            ]
        )

    if not train_log.empty:
        save_best_by = str(config.get("train", {}).get("save_best_by", "val_macro_f1"))
        if save_best_by not in train_log.columns:
            save_best_by = "val_macro_f1"
        best_idx = train_log[save_best_by].idxmin() if save_best_by in {"val_mae", "val_rmse", "val_loss"} else train_log[save_best_by].idxmax()
        best = train_log.loc[best_idx]
        final = train_log.iloc[-1]
        lines.extend(
            [
                "## Training",
                "",
                _markdown_table(
                    [
                        [
                            f"best_{save_best_by}",
                            int(best["epoch"]),
                            best["train_loss"],
                            best["val_loss"],
                            best["val_accuracy"],
                            best["val_balanced_accuracy"],
                            best["val_macro_f1"],
                        ],
                        [
                            "final",
                            int(final["epoch"]),
                            final["train_loss"],
                            final["val_loss"],
                            final["val_accuracy"],
                            final["val_balanced_accuracy"],
                            final["val_macro_f1"],
                        ],
                    ],
                    ["row", "epoch", "train_loss", "val_loss", "val_acc", "val_bal_acc", "val_macro_f1"],
                ),
                "",
                "Test metrics are evaluated from best.pt selected by the configured validation metric.",
                "",
                f"![training curves](figures/training_curves_latest.png)",
                "",
            ]
        )

    if metrics:
        plot_confusion_matrices(run_dir, metrics, class_names)
        metric_rows = _metrics_rows(metrics, "image_level") + _metrics_rows(metrics, "subject_level")
        lines.extend(
            [
                "## Test",
                "",
                _markdown_table(metric_rows, ["level", "metric", "value"]),
                "",
                "### Image Level Class Report",
                "",
                _markdown_table(
                    _class_report_rows(metrics["image_level"]["classification_report"], class_names),
                    ["class", "precision", "recall", "f1", "support"],
                ),
                "",
                "### Subject Level Class Report",
                "",
                _markdown_table(
                    _class_report_rows(metrics["subject_level"]["classification_report"], class_names),
                    ["class", "precision", "recall", "f1", "support"],
                ),
                "",
                "### Confusion Matrices",
                "",
                "![confusion matrices](figures/confusion_matrices.png)",
                "",
            ]
        )

    result_path = run_dir / "result.md"
    result_path.write_text("\n".join(lines), encoding="utf-8")
    return result_path


def generate_report(run_dir: str | Path, epoch: int | None = None) -> None:
    plot_training_curves(run_dir, epoch=epoch)
    write_result_markdown(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    generate_report(args.run_dir)


if __name__ == "__main__":
    main()
