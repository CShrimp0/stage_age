from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "image_dir": "/home/szdx/LNX/data/TA/Healthy/Images",
        "characteristics": "/home/szdx/LNX/data/TA/characteristics.xlsx",
        "sheet_name": "Blad1",
        "bins": [18, 45, 60, 101],
        "class_names": ["18-44", "45-59", "60-100"],
        "split": {"train": 0.70, "val": 0.15, "test": 0.15},
    },
    "model": {
        "name": "resnet18",
        "pretrained": True,
        "num_classes": 3,
    },
    "train": {
        "epochs": 30,
        "batch_size": 32,
        "num_workers": 4,
        "image_size": 224,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "use_class_weights": True,
        "seed": 42,
        "device": "cuda",
    },
    "output_root": "outputs",
    "run_name": "resnet18_baseline",
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path | None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config
    with Path(path).open("r", encoding="utf-8") as f:
        user_config = json.load(f)
    return deep_update(config, user_config)


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet18.json",
        help="Path to a JSON config file.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = deepcopy(config)
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers
    if args.seed is not None:
        config["train"]["seed"] = args.seed
    if args.no_pretrained:
        config["model"]["pretrained"] = False
    return config
