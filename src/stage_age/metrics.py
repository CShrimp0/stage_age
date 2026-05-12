from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def image_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def subject_level_predictions(
    probs: np.ndarray,
    y_true: np.ndarray,
    subject_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.DataFrame({"subject_id": subject_ids, "label": y_true})
    prob_cols = [f"p_{i}" for i in range(probs.shape[1])]
    for idx, col in enumerate(prob_cols):
        df[col] = probs[:, idx]

    grouped = df.groupby("subject_id", sort=True)
    subject_true = grouped["label"].first().to_numpy(dtype=int)
    subject_probs = grouped[prob_cols].mean().to_numpy(dtype=float)
    subject_pred = subject_probs.argmax(axis=1)
    subject_ids_out = grouped["label"].first().index.to_numpy(dtype=int)
    return subject_true, subject_pred, subject_ids_out
