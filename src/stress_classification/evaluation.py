"""Consistent classification metrics, aggregation, and plots."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from .labels import CLASS_NAMES


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Return finite class scores normalized onto the probability simplex.

    Mixed-precision inference can make float16 softmax rows differ slightly
    from one.  Recent scikit-learn releases reject those otherwise valid rows
    for multiclass ROC-AUC, so normalize once before every metric, aggregate,
    and serialized prediction.
    """
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] < 2:
        raise ValueError("probabilities must have shape (samples, classes)")
    if not np.isfinite(probability).all():
        raise ValueError("probabilities contain non-finite values")
    if np.any(probability < 0):
        raise ValueError("probabilities contain negative values")
    row_sums = probability.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("probability rows must have a positive sum")
    return probability / row_sums


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
    *,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, object]:
    true = np.asarray(y_true, dtype=np.int64)
    predicted = np.asarray(y_pred, dtype=np.int64)
    labels = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        true, predicted, labels=labels, zero_division=0
    )
    cm = confusion_matrix(true, predicted, labels=labels)
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(f1_score(true, predicted, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(true, predicted, average="weighted", zero_division=0)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(true, predicted)),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            class_names[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in labels
        },
        "classification_report": classification_report(
            true,
            predicted,
            labels=labels,
            target_names=list(class_names),
            zero_division=0,
            output_dict=True,
        ),
    }
    if probabilities is not None:
        probability = normalize_probabilities(probabilities)
        binary = label_binarize(true, classes=labels)
        try:
            result["roc_auc_ovr_macro"] = float(
                roc_auc_score(true, probability, labels=labels, multi_class="ovr", average="macro")
            )
        except ValueError:
            result["roc_auc_ovr_macro"] = None
        try:
            result["average_precision_macro"] = float(
                average_precision_score(binary, probability, average="macro")
            )
        except ValueError:
            result["average_precision_macro"] = None
    return result


def aggregate_probabilities(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    level: str = "trial",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Average window probabilities at trial or subject level."""

    if level == "trial":
        keys = ["subject_id", "trial_id"]
    elif level == "subject":
        keys = ["subject_id"]
    else:
        raise ValueError("level must be 'trial' or 'subject'")
    frame = metadata.reset_index(drop=True)[keys].copy()
    frame["y_true"] = np.asarray(y_true, dtype=np.int64)
    for index in range(probabilities.shape[1]):
        frame[f"prob_{index}"] = probabilities[:, index]
    rows = []
    for values, group in frame.groupby(keys, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        labels = group.y_true.unique()
        if len(labels) != 1 and level == "trial":
            raise ValueError("a trial has inconsistent labels")
        row = dict(zip(keys, values))
        row["y_true"] = int(group.y_true.mode().iloc[0])
        for index in range(probabilities.shape[1]):
            row[f"prob_{index}"] = float(group[f"prob_{index}"].mean())
        rows.append(row)
    aggregate = pd.DataFrame(rows)
    probability_columns = [f"prob_{index}" for index in range(probabilities.shape[1])]
    aggregated_probabilities = aggregate[probability_columns].to_numpy(dtype=np.float64)
    predicted = aggregated_probabilities.argmax(axis=1)
    return aggregate, aggregate.y_true.to_numpy(dtype=np.int64), predicted, aggregated_probabilities


def plot_confusion_matrix(
    matrix: np.ndarray,
    destination: Path,
    *,
    class_names: tuple[str, ...] = CLASS_NAMES,
    normalized: bool = False,
) -> None:
    values = np.asarray(matrix)
    annotations = values.copy()
    if normalized:
        values = values.astype(np.float64) / np.maximum(values.sum(axis=1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        values,
        annot=annotations if normalized else True,
        fmt="d" if not normalized else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set(xlabel="Predicted", ylabel="True", title="Normalized confusion matrix" if normalized else "Confusion matrix")
    figure.tight_layout()
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def save_evaluation(
    output_dir: str | Path,
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, object]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    probability = normalize_probabilities(probabilities)
    predicted = probability.argmax(axis=1)
    window_metrics = classification_metrics(y_true, predicted, probability, class_names=class_names)
    aggregate, trial_true, trial_pred, trial_prob = aggregate_probabilities(
        metadata, y_true, probability, level="trial"
    )
    trial_metrics = classification_metrics(
        trial_true, trial_pred, trial_prob, class_names=class_names
    )
    metrics = {"window_level": window_metrics, "trial_level": trial_metrics}
    (destination / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    predictions = metadata.reset_index(drop=True).copy()
    predictions["y_true"] = np.asarray(y_true, dtype=np.int64)
    predictions["y_pred"] = predicted
    for index, name in enumerate(class_names):
        predictions[f"prob_{name}"] = probability[:, index]
    predictions.to_csv(destination / "predictions.csv", index=False)
    aggregate["y_pred"] = trial_pred
    aggregate.to_csv(destination / "trial_predictions.csv", index=False)
    matrix = np.asarray(window_metrics["confusion_matrix"])
    plot_confusion_matrix(matrix, destination / "confusion_matrix.png", class_names=class_names)
    plot_confusion_matrix(
        matrix,
        destination / "confusion_matrix_normalized.png",
        class_names=class_names,
        normalized=True,
    )
    return metrics
