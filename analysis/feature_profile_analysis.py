from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats


CLASS_NAMES = ["18-44", "45-64", "65-100"]
COARSE_LABELS = ["18-44", "45-64", "65-100"]
FINE_LABELS = ["18-34", "35-44", "45-54", "55-64", "65-74", "75-100"]
DEFAULT_EXPERIMENT_DIR = Path("/home/szdx/LNX/stage-age/outputs/20260511_162523_usfm_partial_last_block_age45_65_multiseed")
DEFAULT_OUTPUT_ROOT = Path("/home/szdx/LNX/stage-age/outputs")


def unique_output_dir(path: Path) -> Path:
    if not path.exists():
        return path
    idx = 2
    while True:
        candidate = Path(f"{path}_v{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def resolve_output_dir(output_dir: str | None) -> Path:
    if output_dir:
        return unique_output_dir(Path(output_dir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return unique_output_dir(DEFAULT_OUTPUT_ROOT / f"{timestamp}_feature_profile_analysis")


def resolve_experiment_dir(experiment_dir: str | None) -> Path:
    if experiment_dir:
        path = Path(experiment_dir)
        if path.exists():
            return path
        print(f"warning: requested experiment_dir not found: {path}; falling back to {DEFAULT_EXPERIMENT_DIR}")
    if not DEFAULT_EXPERIMENT_DIR.exists():
        raise FileNotFoundError(f"Fallback experiment_dir does not exist: {DEFAULT_EXPERIMENT_DIR}")
    return DEFAULT_EXPERIMENT_DIR


def coarse_age_group(age: float) -> str:
    if age < 45:
        return "18-44"
    if age < 65:
        return "45-64"
    return "65-100"


def fine_age_group(age: float) -> str:
    if age < 35:
        return "18-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    if age < 75:
        return "65-74"
    return "75-100"


def label_to_name(label: int) -> str:
    return CLASS_NAMES[int(label)]


def pred_direction(label: int, pred: int) -> str:
    if int(label) == int(pred):
        return "correct"
    return "predicted_younger" if int(pred) < int(label) else "predicted_older"


def read_seed_manifests(experiment_dir: Path) -> tuple[pd.DataFrame, list[int]]:
    rows = []
    seeds = []
    for seed_dir in sorted(experiment_dir.glob("seed*")):
        manifest_path = seed_dir / "manifest.csv"
        if not manifest_path.exists():
            continue
        seed_text = seed_dir.name.replace("seed", "")
        seed = int(seed_text) if seed_text.isdigit() else len(seeds)
        df = pd.read_csv(manifest_path)
        df["seed"] = seed
        rows.append(df)
        seeds.append(seed)
    if not rows:
        raise FileNotFoundError(f"No seed*/manifest.csv files found under {experiment_dir}")
    manifest = pd.concat(rows, ignore_index=True)
    return manifest, seeds


def read_seed_predictions(experiment_dir: Path) -> pd.DataFrame:
    rows = []
    for seed_dir in sorted(experiment_dir.glob("seed*")):
        pred_path = seed_dir / "subject_test_predictions.csv"
        if not pred_path.exists():
            continue
        seed_text = seed_dir.name.replace("seed", "")
        seed = int(seed_text) if seed_text.isdigit() else len(rows)
        df = pd.read_csv(pred_path)
        df["seed"] = seed
        if "label" not in df.columns and "true_class" in df.columns:
            df["label"] = df["true_class"]
        if "pred" not in df.columns and "pred_class" in df.columns:
            df["pred"] = df["pred_class"]
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No seed*/subject_test_predictions.csv files found under {experiment_dir}")
    return pd.concat(rows, ignore_index=True)


def valid_crop(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    valid = gray > 5.0 / 255.0
    if not bool(valid.any()):
        valid = np.ones_like(gray, dtype=bool)
    ys, xs = np.where(valid)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = gray[y0:y1, x0:x1]
    crop_valid = valid[y0:y1, x0:x1]
    height, width = crop.shape
    geom = {
        "valid_area_ratio": float(valid.mean()),
        "valid_width": float(width),
        "valid_height": float(height),
        "aspect_ratio": float(width / height) if height > 0 else math.nan,
    }
    return crop, crop_valid, geom


def glcm_features(crop: np.ndarray, mask: np.ndarray, levels: int = 16) -> dict[str, float]:
    values = np.clip((crop * levels).astype(np.int32), 0, levels - 1)
    glcm = np.zeros((levels, levels), dtype=np.float64)
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dy, dx in directions:
        if dy >= 0:
            y_src = slice(0, crop.shape[0] - dy)
            y_dst = slice(dy, crop.shape[0])
        else:
            y_src = slice(-dy, crop.shape[0])
            y_dst = slice(0, crop.shape[0] + dy)
        if dx >= 0:
            x_src = slice(0, crop.shape[1] - dx)
            x_dst = slice(dx, crop.shape[1])
        else:
            x_src = slice(-dx, crop.shape[1])
            x_dst = slice(0, crop.shape[1] + dx)
        src_mask = mask[y_src, x_src] & mask[y_dst, x_dst]
        src = values[y_src, x_src][src_mask]
        dst = values[y_dst, x_dst][src_mask]
        if len(src):
            np.add.at(glcm, (src, dst), 1)
            np.add.at(glcm, (dst, src), 1)
    if glcm.sum() == 0:
        return {
            "glcm_contrast": math.nan,
            "glcm_homogeneity": math.nan,
            "glcm_energy": math.nan,
            "glcm_correlation": math.nan,
            "glcm_entropy": math.nan,
        }
    p = glcm / glcm.sum()
    i, j = np.indices(p.shape)
    contrast = float(((i - j) ** 2 * p).sum())
    homogeneity = float((p / (1.0 + np.abs(i - j))).sum())
    energy = float(np.sqrt((p**2).sum()))
    entropy = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    mu_i = float((i * p).sum())
    mu_j = float((j * p).sum())
    std_i = float(np.sqrt((((i - mu_i) ** 2) * p).sum()))
    std_j = float(np.sqrt((((j - mu_j) ** 2) * p).sum()))
    correlation = float((((i - mu_i) * (j - mu_j) * p).sum()) / (std_i * std_j)) if std_i > 0 and std_j > 0 else 0.0
    return {
        "glcm_contrast": contrast,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "glcm_correlation": correlation,
        "glcm_entropy": entropy,
    }


def image_quality_features(crop: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    padded = np.pad(crop, 1, mode="edge")
    lap = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4 * padded[1:-1, 1:-1]
    )
    gy, gx = np.gradient(crop)
    grad = np.sqrt(gx**2 + gy**2)
    valid_values = crop[mask]
    std = float(valid_values.std())
    return {
        "laplacian_sharpness": float(lap[mask].var()) if bool(mask.any()) else math.nan,
        "edge_strength": float(grad[mask].mean()) if bool(mask.any()) else math.nan,
        "snr_proxy": float(valid_values.mean() / (std + 1e-6)) if len(valid_values) else math.nan,
    }


def extract_image_features(row: pd.Series) -> dict[str, Any]:
    path = Path(row["image_path"])
    base = {
        "image_path": str(path),
        "subject_id": int(row["subject_id"]),
        "view": int(row.get("view", -1)),
        "age": float(row["age"]),
        "label": int(row["label"]),
        "class_name": str(row["class_name"]),
    }
    try:
        gray = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        crop, mask, geom = valid_crop(gray)
        values = crop[mask]
        intensity = {
            "mean_intensity": float(values.mean()),
            "median_intensity": float(np.median(values)),
            "std_intensity": float(values.std()),
            "iqr_intensity": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "p25_intensity": float(np.percentile(values, 25)),
            "p75_intensity": float(np.percentile(values, 75)),
        }
        base.update(intensity)
        base.update(glcm_features(crop, mask))
        base.update(image_quality_features(crop, mask))
        base.update(geom)
        base["read_error"] = ""
    except Exception as exc:  # noqa: BLE001
        base["read_error"] = str(exc)
    return base


def extract_all_image_features(manifest: pd.DataFrame) -> pd.DataFrame:
    unique = manifest.drop_duplicates("image_path").sort_values(["subject_id", "view"]).reset_index(drop=True)
    rows = []
    total = len(unique)
    for idx, row in unique.iterrows():
        if idx % 250 == 0:
            print(f"extracting image features {idx}/{total}")
        rows.append(extract_image_features(row))
    return pd.DataFrame(rows)


def aggregate_subject_features(image_features: pd.DataFrame) -> pd.DataFrame:
    feature_cols = numeric_feature_columns(image_features)
    agg = image_features.groupby("subject_id")[feature_cols].agg(["mean", "median", "std"])
    agg.columns = [f"{feature}_{stat}" for feature, stat in agg.columns]
    agg = agg.reset_index()
    meta = (
        image_features.sort_values("subject_id")
        .groupby("subject_id")
        .agg(true_age=("age", "first"), true_class=("label", "first"), coarse_age_group=("class_name", "first"), n_images=("image_path", "count"))
        .reset_index()
    )
    meta["fine_age_group"] = meta["true_age"].map(fine_age_group)
    meta["coarse_age_group"] = meta["true_age"].map(coarse_age_group)
    return meta.merge(agg, on="subject_id", how="left")


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"subject_id", "view", "age", "label"}
    cols = []
    for col in df.columns:
        if col in exclude or col.endswith("_id") or col in {"image_path", "class_name", "read_error"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def subject_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"subject_id", "true_age", "true_class", "n_images"}
    cols = []
    for col in df.columns:
        if col in exclude or col.endswith("_group"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def merge_seed_predictions(subject_features: pd.DataFrame, predictions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    meta = manifest.drop_duplicates(["seed", "subject_id"])[["seed", "subject_id", "age", "label", "class_name", "split"]]
    pred = predictions.merge(meta, on=["seed", "subject_id"], how="left", suffixes=("", "_manifest"))
    pred["true_age"] = pred["age"]
    pred["true_class"] = pred["label"].astype(int)
    pred["pred_class"] = pred["pred"].astype(int)
    pred["correct"] = pred["true_class"] == pred["pred_class"]
    pred["pred_direction"] = [pred_direction(label, pred_value) for label, pred_value in zip(pred["true_class"], pred["pred_class"])]
    pred["coarse_age_group"] = pred["true_age"].map(coarse_age_group)
    pred["fine_age_group"] = pred["true_age"].map(fine_age_group)
    merge_cols = ["subject_id"] + subject_feature_columns(subject_features)
    return pred.merge(subject_features[merge_cols], on="subject_id", how="left")


def consensus_predictions(seed_merged: pd.DataFrame, subject_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject_id, group in seed_merged.groupby("subject_id", sort=True):
        preds = group["pred_class"].astype(int).tolist()
        labels = group["true_class"].astype(int).tolist()
        true_class = labels[0]
        pred_counts = Counter(preds)
        most_common_pred = pred_counts.most_common(1)[0][0]
        correct_count = int((group["correct"] == True).sum())  # noqa: E712
        rows.append(
            {
                "subject_id": int(subject_id),
                "true_age": float(group["true_age"].iloc[0]),
                "true_class": int(true_class),
                "most_common_pred_class": int(most_common_pred),
                "correct_count_across_seeds": correct_count,
                "n_seed_predictions": int(len(group)),
                "always_correct": bool(correct_count == len(group)),
                "always_wrong": bool(correct_count == 0),
                "ever_wrong": bool(correct_count < len(group)),
                "consensus_correct": bool(int(most_common_pred) == int(true_class)),
                "pred_direction": pred_direction(true_class, most_common_pred),
                "coarse_age_group": coarse_age_group(float(group["true_age"].iloc[0])),
                "fine_age_group": fine_age_group(float(group["true_age"].iloc[0])),
            }
        )
    consensus = pd.DataFrame(rows)
    merge_cols = ["subject_id"] + subject_feature_columns(subject_features)
    return consensus.merge(subject_features[merge_cols], on="subject_id", how="left")


def fdr_bh(p_values: list[float]) -> list[float]:
    p = np.asarray([1.0 if not math.isfinite(v) else v for v in p_values], dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = ranked * n / (np.arange(n) + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q = np.empty(n, dtype=float)
    q[order] = np.clip(q_ranked, 0, 1)
    return q.tolist()


def kruskal_table(df: pd.DataFrame, group_col: str, feature_cols: list[str], group_order: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_cols:
        groups = []
        ns = []
        for group in group_order:
            values = df.loc[df[group_col] == group, feature].dropna().to_numpy(dtype=float)
            groups.append(values)
            ns.append(len(values))
        non_empty = [values for values in groups if len(values) > 0]
        if len(non_empty) >= 2:
            try:
                h_stat, p_value = stats.kruskal(*non_empty)
            except ValueError:
                h_stat, p_value = math.nan, 1.0
        else:
            h_stat, p_value = math.nan, 1.0
        n_total = int(sum(ns))
        k = len(non_empty)
        epsilon_sq = float((h_stat - k + 1) / (n_total - k)) if math.isfinite(h_stat) and n_total > k else math.nan
        row = {
            "feature": feature,
            "group_col": group_col,
            "kruskal_h": float(h_stat) if math.isfinite(h_stat) else math.nan,
            "p_value": float(p_value) if math.isfinite(p_value) else 1.0,
            "epsilon_squared": epsilon_sq,
            "n_total": n_total,
        }
        for group, n in zip(group_order, ns):
            values = df.loc[df[group_col] == group, feature].dropna()
            row[f"{group}_n"] = int(n)
            row[f"{group}_median"] = float(values.median()) if len(values) else math.nan
            row[f"{group}_mean"] = float(values.mean()) if len(values) else math.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out["q_value_fdr"] = fdr_bh(out["p_value"].tolist())
    return out.sort_values(["q_value_fdr", "p_value", "feature"]).reset_index(drop=True)


def spearman_table(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for feature in feature_cols:
        sub = df[["true_age", feature]].dropna()
        if len(sub) >= 3 and sub[feature].nunique() > 1:
            rho, p_value = stats.spearmanr(sub["true_age"], sub[feature])
        else:
            rho, p_value = math.nan, 1.0
        rows.append({"feature": feature, "spearman_rho": float(rho) if math.isfinite(rho) else math.nan, "p_value": float(p_value) if math.isfinite(p_value) else 1.0, "n": int(len(sub))})
    out = pd.DataFrame(rows)
    out["q_value_fdr"] = fdr_bh(out["p_value"].tolist())
    out["abs_spearman_rho"] = out["spearman_rho"].abs()
    return out.sort_values(["q_value_fdr", "abs_spearman_rho"], ascending=[True, False]).reset_index(drop=True)


def group_summary(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for group_col, order in [("coarse_age_group", COARSE_LABELS), ("fine_age_group", FINE_LABELS)]:
        for group in order:
            sub = df[df[group_col] == group]
            for feature in feature_cols:
                values = sub[feature].dropna()
                rows.append(
                    {
                        "group_col": group_col,
                        "group": group,
                        "feature": feature,
                        "n": int(len(values)),
                        "mean": float(values.mean()) if len(values) else math.nan,
                        "median": float(values.median()) if len(values) else math.nan,
                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def prediction_group_stats(seed_merged: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for group_col, order in [
        ("correct_label", ["correct", "wrong"]),
        ("pred_direction", ["predicted_younger", "correct", "predicted_older"]),
    ]:
        df = seed_merged.copy()
        if group_col == "correct_label":
            df[group_col] = np.where(df["correct"], "correct", "wrong")
        for feature in feature_cols:
            groups = [df.loc[df[group_col] == group, feature].dropna().to_numpy(dtype=float) for group in order]
            non_empty = [values for values in groups if len(values) > 0]
            if len(non_empty) >= 2:
                try:
                    h_stat, p_value = stats.kruskal(*non_empty)
                except ValueError:
                    h_stat, p_value = math.nan, 1.0
            else:
                h_stat, p_value = math.nan, 1.0
            row = {"comparison": group_col, "feature": feature, "kruskal_h": h_stat, "p_value": p_value}
            for group, values in zip(order, groups):
                row[f"{group}_n"] = int(len(values))
                row[f"{group}_mean"] = float(np.mean(values)) if len(values) else math.nan
                row[f"{group}_median"] = float(np.median(values)) if len(values) else math.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    out["q_value_fdr"] = fdr_bh(out["p_value"].tolist())
    return out.sort_values(["q_value_fdr", "p_value", "feature"]).reset_index(drop=True)


def error_45_64_stats(seed_merged: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = seed_merged[seed_merged["true_class"] == 1].copy()
    order = ["predicted_younger", "correct", "predicted_older"]
    rows = []
    for feature in feature_cols:
        groups = [df.loc[df["pred_direction"] == group, feature].dropna().to_numpy(dtype=float) for group in order]
        non_empty = [values for values in groups if len(values) > 0]
        if len(non_empty) >= 2:
            try:
                h_stat, p_value = stats.kruskal(*non_empty)
            except ValueError:
                h_stat, p_value = math.nan, 1.0
        else:
            h_stat, p_value = math.nan, 1.0
        row = {"feature": feature, "kruskal_h": h_stat, "p_value": p_value}
        for group, values in zip(order, groups):
            row[f"{group}_n"] = int(len(values))
            row[f"{group}_mean"] = float(np.mean(values)) if len(values) else math.nan
            row[f"{group}_median"] = float(np.median(values)) if len(values) else math.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out["q_value_fdr"] = fdr_bh(out["p_value"].tolist())
    return out.sort_values(["q_value_fdr", "p_value", "feature"]).reset_index(drop=True)


def zscore_by_feature(matrix: pd.DataFrame) -> pd.DataFrame:
    out = matrix.copy()
    for idx in out.index:
        values = out.loc[idx].to_numpy(dtype=float)
        mean = np.nanmean(values)
        std = np.nanstd(values)
        out.loc[idx] = (values - mean) / (std + 1e-9)
    return out


def choose_top_features(feature_cols: list[str], spearman: pd.DataFrame, significance: pd.DataFrame, max_features: int = 30) -> list[str]:
    ranked = []
    for df, score_col in [(spearman, "abs_spearman_rho"), (significance, "kruskal_h")]:
        if score_col not in df.columns:
            continue
        for feature in df.sort_values(["q_value_fdr", score_col], ascending=[True, False])["feature"].tolist():
            if feature in feature_cols and feature not in ranked:
                ranked.append(feature)
            if len(ranked) >= max_features:
                return ranked
    return feature_cols[:max_features]


def plot_heatmap(matrix: pd.DataFrame, title: str, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    height = max(5, min(18, 0.35 * len(matrix.index) + 2.0))
    width = max(7, 1.4 * len(matrix.columns) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height), dpi=180)
    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-2, vmax=2)
    ax.set_xticks(np.arange(len(matrix.columns)), labels=matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="z-scored group mean")
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"))
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "无"
    cols = df.columns.tolist()
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for record in df.to_dict(orient="records"):
        values = []
        for col in cols:
            value = record[col]
            if isinstance(value, float):
                values.append(f"{value:.4g}" if math.isfinite(value) else "")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def heatmap_from_groups(df: pd.DataFrame, group_col: str, group_order: list[str], feature_cols: list[str], title: str, output_base: Path) -> None:
    means = []
    for group in group_order:
        means.append(df.loc[df[group_col] == group, feature_cols].mean(numeric_only=True))
    matrix = pd.DataFrame(means, index=group_order).T
    matrix = zscore_by_feature(matrix)
    plot_heatmap(matrix, title, output_base)


def write_report(
    output_dir: Path,
    experiment_dir: Path,
    seeds: list[int],
    image_features: pd.DataFrame,
    subject_features: pd.DataFrame,
    seed_merged: pd.DataFrame,
    consensus: pd.DataFrame,
    feature_cols: list[str],
    fine_sig: pd.DataFrame,
    coarse_sig: pd.DataFrame,
    spearman: pd.DataFrame,
    missing_images: int,
    unmatched_predictions: int,
) -> None:
    top_age = spearman.head(10)[["feature", "spearman_rho", "q_value_fdr"]]
    top_fine = fine_sig.head(10)[["feature", "kruskal_h", "epsilon_squared", "q_value_fdr"]]
    pred_counts = seed_merged["pred_direction"].value_counts().to_dict()
    group_counts = subject_features["fine_age_group"].value_counts().reindex(FINE_LABELS, fill_value=0).to_dict()
    lines = [
        "# Subject-Level Ultrasound Feature Profile Analysis",
        "",
        "## 1. 分析目的",
        "",
        "本分析用于做 subject-level ultrasound feature profiling，探索不同年龄段的可解释图像特征差异，以及模型预测正确/错误、预测偏老/偏年轻与这些特征之间的关系。它不直接训练模型，而是帮助解释当前年龄三分类任务为什么在 45-64 中间组和边界年龄段更难。",
        "",
        "## 2. 输入数据",
        "",
        f"- image_dir: /home/szdx/LNX/data/TA/Healthy/Images",
        f"- experiment_dir: {experiment_dir}",
        f"- seeds: {seeds}",
        "- 分析单位：subject-level；image-level features 先提取，再按 subject 聚合。",
        "",
        "## 3. 特征提取方法",
        "",
        "每张图像先转灰度，使用简单非黑像素阈值去除黑边并定义有效图像区域。当前没有 mask/ROI，因此本轮特征是 heuristic ultrasound/image features，不是手工标注 ROI 或 muscle ROI 特征。后续如果有 mask，可替换为 ROI-level 特征。",
        "",
        "特征包括：强度/回声特征、GLCM 纹理特征、清晰度/边缘质量特征，以及有效区域几何特征。每个 image-level 特征在 subject 内聚合为 mean / median / std。",
        "",
        "## 4. 年龄分组",
        "",
        f"- coarse groups: {COARSE_LABELS}",
        f"- fine groups: {FINE_LABELS}",
        f"- fine group subject counts: {group_counts}",
        "",
        "## 5. 统计方法",
        "",
        "连续特征使用 Kruskal-Wallis 检验比较不同年龄组，Spearman correlation 分析 feature vs true_age。所有 p-value 同时输出 FDR q-value，多重比较后以 q-value 为主。报告中不只看 p 值，也结合 Spearman rho、Kruskal H 和 epsilon-squared 判断趋势和效应大小。",
        "",
        "模型表现分析包括 correct vs wrong，以及 predicted_younger / correct / predicted_older。45-64 组单独输出错误方向分析。",
        "",
        "## 6. 主要结果摘要",
        "",
        f"- image_features rows: {len(image_features)}",
        f"- subject_features rows: {len(subject_features)}",
        f"- seed-level merged prediction rows: {len(seed_merged)}",
        f"- consensus subjects: {len(consensus)}",
        f"- pred_direction counts: {pred_counts}",
        "",
        "### 与年龄相关性最强的特征 Top 10",
        "",
        markdown_table(top_age),
        "",
        "### fine_age_group 差异最明显的特征 Top 10",
        "",
        markdown_table(top_fine),
        "",
        "## 7. 文件说明",
        "",
        "- image_features.csv：每张图像的 heuristic ultrasound/image features。",
        "- subject_features.csv：按 subject 聚合后的特征。",
        "- seed_level_subject_feature_predictions.csv：每个 seed 的 test subject 预测与 subject features 合并表。",
        "- subject_consensus_features.csv：跨 seed 的 subject consensus 预测表现表。",
        "- feature_group_summary.csv：年龄组内每个特征的均值/中位数/标准差。",
        "- feature_significance_fine_age.csv / feature_significance_coarse_age.csv：年龄组 Kruskal-Wallis 统计。",
        "- feature_spearman_age.csv：feature vs true_age 的 Spearman 相关。",
        "- feature_prediction_group_stats.csv：模型表现组特征差异。",
        "- feature_45_64_error_analysis.csv：45-64 专项错误方向分析。",
        "- figures/*.png / *.pdf：年龄组和预测表现相关特征热力图。",
        "",
        "## 8. 局限性",
        "",
        "- 没有 mask/ROI 时，特征可能受背景、黑边、设备显示参数、增益和图像裁剪影响。",
        "- 本统计分析是探索性分析，不证明因果。",
        "- 多重比较后应优先看 FDR q-value。",
        "- 后续建议加入 ROI/mask，或结合 subject-level pooling 模型做更精确的模型特征解释。",
        "",
        "## 9. 数据质量",
        "",
        f"- unreadable_or_failed_images: {missing_images}",
        f"- predictions_without_matched_age: {unmatched_predictions}",
    ]
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> Path:
    experiment_dir = resolve_experiment_dir(args.experiment_dir)
    output_dir = resolve_output_dir(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, seeds = read_seed_manifests(experiment_dir)
    predictions = read_seed_predictions(experiment_dir)
    image_features = extract_all_image_features(manifest)
    missing_images = int((image_features.get("read_error", pd.Series(dtype=str)).fillna("") != "").sum())
    image_features.to_csv(output_dir / "image_features.csv", index=False)

    subject_features = aggregate_subject_features(image_features)
    subject_features.to_csv(output_dir / "subject_features.csv", index=False)
    feature_cols = subject_feature_columns(subject_features)

    seed_merged = merge_seed_predictions(subject_features, predictions, manifest)
    unmatched_predictions = int(seed_merged["true_age"].isna().sum())
    seed_merged.to_csv(output_dir / "seed_level_subject_feature_predictions.csv", index=False)
    consensus = consensus_predictions(seed_merged, subject_features)
    consensus.to_csv(output_dir / "subject_consensus_features.csv", index=False)

    group_summary(subject_features, feature_cols).to_csv(output_dir / "feature_group_summary.csv", index=False)
    fine_sig = kruskal_table(subject_features, "fine_age_group", feature_cols, FINE_LABELS)
    coarse_sig = kruskal_table(subject_features, "coarse_age_group", feature_cols, COARSE_LABELS)
    spearman = spearman_table(subject_features, feature_cols)
    pred_stats = prediction_group_stats(seed_merged, feature_cols)
    mid_stats = error_45_64_stats(seed_merged, feature_cols)
    fine_sig.to_csv(output_dir / "feature_significance_fine_age.csv", index=False)
    coarse_sig.to_csv(output_dir / "feature_significance_coarse_age.csv", index=False)
    spearman.to_csv(output_dir / "feature_spearman_age.csv", index=False)
    pred_stats.to_csv(output_dir / "feature_prediction_group_stats.csv", index=False)
    mid_stats.to_csv(output_dir / "feature_45_64_error_analysis.csv", index=False)

    top_features = choose_top_features(feature_cols, spearman, fine_sig, max_features=int(args.max_heatmap_features))
    heatmap_from_groups(subject_features, "fine_age_group", FINE_LABELS, top_features, "Fine Age Group Feature Heatmap", figures_dir / "fine_age_feature_heatmap")
    heatmap_from_groups(subject_features, "coarse_age_group", COARSE_LABELS, top_features, "Coarse Age Group Feature Heatmap", figures_dir / "coarse_age_feature_heatmap")

    pred_features = choose_top_features(feature_cols, spearman, pred_stats.rename(columns={"kruskal_h": "kruskal_h"}), max_features=int(args.max_heatmap_features))
    seed_merged["correct_label"] = np.where(seed_merged["correct"], "correct", "wrong")
    heatmap_from_groups(seed_merged, "correct_label", ["correct", "wrong"], pred_features, "Prediction Group Feature Heatmap", figures_dir / "prediction_group_feature_heatmap")

    mid = seed_merged[seed_merged["true_class"] == 1].copy()
    if not mid.empty:
        mid_features = choose_top_features(feature_cols, spearman, mid_stats, max_features=int(args.max_heatmap_features))
        heatmap_from_groups(mid, "pred_direction", ["predicted_younger", "correct", "predicted_older"], mid_features, "45-64 Error Direction Feature Heatmap", figures_dir / "age45_64_error_feature_heatmap")

    config = {
        "image_dir": str(args.image_dir),
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "seeds": seeds,
        "coarse_age_groups": COARSE_LABELS,
        "fine_age_groups": FINE_LABELS,
        "feature_count": len(feature_cols),
        "max_heatmap_features": int(args.max_heatmap_features),
    }
    (output_dir / "analysis_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(
        output_dir,
        experiment_dir,
        seeds,
        image_features,
        subject_features,
        seed_merged,
        consensus,
        feature_cols,
        fine_sig,
        coarse_sig,
        spearman,
        missing_images,
        unmatched_predictions,
    )
    print(f"output_dir={output_dir}")
    print(f"image_features_rows={len(image_features)}")
    print(f"subject_features_rows={len(subject_features)}")
    print(f"missing_images={missing_images}")
    print(f"unmatched_predictions={unmatched_predictions}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="/home/szdx/LNX/data/TA/Healthy/Images")
    parser.add_argument("--experiment_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_heatmap_features", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
