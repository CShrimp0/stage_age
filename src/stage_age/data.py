from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_RE = re.compile(r"^anon_(\d+)_(\d+)\.png$")


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15


def age_to_label(age: float, bins: list[float]) -> int | None:
    for idx, (lower, upper) in enumerate(zip(bins[:-1], bins[1:])):
        if lower <= age < upper:
            return idx
    return None


def read_healthy_characteristics(path: str | Path, sheet_name: str = "Blad1") -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    healthy = raw.iloc[2:, 0:5].copy()
    healthy.columns = ["subject_id", "age", "length", "weight", "sex"]
    healthy["subject_id"] = pd.to_numeric(healthy["subject_id"], errors="coerce")
    healthy["age"] = pd.to_numeric(healthy["age"], errors="coerce")
    healthy = healthy.dropna(subset=["subject_id", "age"]).copy()
    healthy["subject_id"] = healthy["subject_id"].astype(int)
    return healthy


def build_manifest(
    image_dir: str | Path,
    characteristics: str | Path,
    sheet_name: str,
    bins: list[float],
    class_names: list[str],
    split: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    image_dir = Path(image_dir)
    characteristics_df = read_healthy_characteristics(characteristics, sheet_name)
    meta = characteristics_df.set_index("subject_id").to_dict(orient="index")

    rows: list[dict[str, object]] = []
    for image_path in sorted(image_dir.glob("*.png")):
        match = IMAGE_RE.match(image_path.name)
        if not match:
            continue
        subject_id = int(match.group(1))
        view = int(match.group(2))
        if subject_id not in meta:
            continue
        age = float(meta[subject_id]["age"])
        label = age_to_label(age, bins)
        if label is None:
            continue
        rows.append(
            {
                "image_path": str(image_path),
                "subject_id": subject_id,
                "view": view,
                "age": age,
                "sex": meta[subject_id].get("sex"),
                "label": label,
                "class_name": class_names[label],
            }
        )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("No usable images were found after matching metadata and age bins.")

    subject_df = (
        manifest[["subject_id", "label", "class_name"]]
        .drop_duplicates("subject_id")
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    split_cfg = SplitConfig(**split)
    holdout = split_cfg.val + split_cfg.test
    if holdout <= 0 or split_cfg.train <= 0:
        raise ValueError("Split fractions must include train and at least one holdout split.")

    train_subjects, holdout_subjects = train_test_split(
        subject_df,
        test_size=holdout,
        random_state=seed,
        stratify=subject_df["label"],
    )
    val_fraction_in_holdout = split_cfg.val / holdout
    val_subjects, test_subjects = train_test_split(
        holdout_subjects,
        test_size=1.0 - val_fraction_in_holdout,
        random_state=seed,
        stratify=holdout_subjects["label"],
    )

    split_map = {sid: "train" for sid in train_subjects["subject_id"]}
    split_map.update({sid: "val" for sid in val_subjects["subject_id"]})
    split_map.update({sid: "test" for sid in test_subjects["subject_id"]})
    manifest["split"] = manifest["subject_id"].map(split_map)

    ordered_cols = [
        "image_path",
        "subject_id",
        "view",
        "age",
        "sex",
        "label",
        "class_name",
        "split",
    ]
    return manifest[ordered_cols].sort_values(["split", "subject_id", "view"]).reset_index(drop=True)


class UltrasoundAgeDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform: transforms.Compose | None = None):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.manifest.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        subject_id = int(row["subject_id"])
        return image, label, subject_id


def make_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform
