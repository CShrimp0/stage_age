from __future__ import annotations

import argparse
from pathlib import Path

from stage_age.config import load_config
from stage_age.data import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/resnet18.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = build_manifest(
        image_dir=config["data"]["image_dir"],
        characteristics=config["data"]["characteristics"],
        sheet_name=config["data"]["sheet_name"],
        bins=config["data"]["bins"],
        class_names=config["data"]["class_names"],
        split=config["data"]["split"],
        seed=int(config["train"]["seed"]),
    )
    print("images:", len(manifest))
    print("subjects:", manifest["subject_id"].nunique())
    print("\nsubjects by split/class")
    print(
        manifest.drop_duplicates("subject_id")
        .groupby(["split", "class_name"])
        .size()
        .rename("subjects")
        .to_string()
    )
    print("\nimages by split/class")
    print(manifest.groupby(["split", "class_name"]).size().rename("images").to_string())
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(path, index=False)
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
