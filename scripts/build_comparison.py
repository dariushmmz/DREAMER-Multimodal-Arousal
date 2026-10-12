"""Build a single comparable table from corrected experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "corrected" / "dreamer_arousal_3class"


def metric_row(name: str, family: str, path: Path) -> dict[str, object]:
    metrics = json.loads(path.read_text(encoding="utf-8"))["window_level"]
    return {
        "model": name,
        "family": family,
        "validation_f1_macro": None,
        "test_accuracy": metrics["accuracy"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_f1_macro": metrics["f1_macro"],
        "test_roc_auc_ovr_macro": metrics.get("roc_auc_ovr_macro"),
        "test_average_precision_macro": metrics.get("average_precision_macro"),
        "status": "completed",
    }


def per_class_rows(name: str, family: str, path: Path) -> list[dict[str, object]]:
    metrics = json.loads(path.read_text(encoding="utf-8"))["window_level"]
    return [
        {
            "model": name,
            "family": family,
            "class": class_name,
            **values,
        }
        for class_name, values in metrics["per_class"].items()
    ]


def main() -> None:
    classical = pd.read_csv(RESULTS / "classical" / "comparison.csv")
    rows = []
    for row in classical.itertuples():
        result = metric_row(
            row.model,
            "classical_ml",
            RESULTS / "classical" / row.model / "metrics.json",
        )
        result["validation_f1_macro"] = row.validation_f1_macro
        rows.append(result)
    stacking = RESULTS / "paper_2023_adapted_stacking" / "metrics.json"
    if stacking.is_file():
        result = metric_row("paper_2023_adapted_stacking", "classical_ensemble", stacking)
        protocol = json.loads(stacking.with_name("protocol.json").read_text(encoding="utf-8"))
        result["validation_f1_macro"] = protocol["validation_f1_macro"]
        rows.append(result)
    for model in ("base_cnn", "paper_inspired_attention"):
        metrics = RESULTS / model / "metrics.json"
        if metrics.is_file():
            result = metric_row(model, "deep_learning", metrics)
            metadata = json.loads(metrics.with_name("run_metadata.json").read_text(encoding="utf-8"))
            result["validation_f1_macro"] = metadata["best_validation_f1_macro"]
            rows.append(result)
        else:
            rows.append(
                {
                    "model": model,
                    "family": "deep_learning",
                    "validation_f1_macro": None,
                    "test_accuracy": None,
                    "test_balanced_accuracy": None,
                    "test_f1_macro": None,
                    "test_roc_auc_ovr_macro": None,
                    "test_average_precision_macro": None,
                    "status": "pending_kaggle_execution",
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS / "comparison_all.csv", index=False)
    completed = comparison[comparison.status == "completed"].copy()
    figure, axis = plt.subplots(figsize=(12, 6))
    completed.set_index("model")[[
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1_macro",
        "test_roc_auc_ovr_macro",
    ]].plot.bar(
        ax=axis
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Corrected subject-independent three-class arousal results")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(RESULTS / "comparison_all.png", dpi=200)
    plt.close(figure)

    artifact_paths = {
        row.model: (RESULTS / "classical" / row.model / "metrics.json", "classical_ml")
        for row in classical.itertuples()
    }
    artifact_paths["paper_2023_adapted_stacking"] = (stacking, "classical_ensemble")
    for model in ("base_cnn", "paper_inspired_attention"):
        artifact_paths[model] = (RESULTS / model / "metrics.json", "deep_learning")
    class_rows: list[dict[str, object]] = []
    for name, (path, family) in artifact_paths.items():
        if path.is_file():
            class_rows.extend(per_class_rows(name, family, path))
    per_class = pd.DataFrame(class_rows)
    per_class.to_csv(RESULTS / "per_class_metrics.csv", index=False)
    if not per_class.empty:
        pivot = per_class.pivot(index="model", columns="class", values="f1")
        pivot = pivot.reindex(columns=["low", "medium", "high"])
        figure, axis = plt.subplots(figsize=(7.5, max(4.5, 0.55 * len(pivot))))
        image = axis.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns)
        axis.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
        for row_index in range(len(pivot.index)):
            for column_index in range(len(pivot.columns)):
                value = pivot.iloc[row_index, column_index]
                axis.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", color="white" if value < 0.55 else "black")
        axis.set_title("Per-class F1 on fixed subject-independent test set")
        axis.set_xlabel("Arousal class")
        figure.colorbar(image, ax=axis, label="F1")
        figure.tight_layout()
        figure.savefig(RESULTS / "per_class_f1_comparison.png", dpi=200)
        plt.close(figure)

    subject_rows: list[dict[str, object]] = []
    for name, (metrics_path, family) in artifact_paths.items():
        predictions_path = metrics_path.with_name("predictions.csv")
        if not predictions_path.is_file():
            continue
        predictions = pd.read_csv(predictions_path)
        for subject_id, group in predictions.groupby("subject_id", sort=True):
            true = group["y_true"].to_numpy(dtype=np.int64)
            predicted = group["y_pred"].to_numpy(dtype=np.int64)
            subject_rows.append(
                {
                    "model": name,
                    "family": family,
                    "subject_id": int(subject_id),
                    "n_windows": len(group),
                    "accuracy": accuracy_score(true, predicted),
                    "balanced_accuracy": balanced_accuracy_score(true, predicted),
                    "f1_macro": f1_score(true, predicted, average="macro", zero_division=0),
                }
            )
    subject_metrics = pd.DataFrame(subject_rows)
    subject_metrics.to_csv(RESULTS / "subject_metrics.csv", index=False)
    if not subject_metrics.empty:
        pivot = subject_metrics.pivot(index="model", columns="subject_id", values="f1_macro")
        figure, axis = plt.subplots(figsize=(7.5, max(4.5, 0.55 * len(pivot))))
        image = axis.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap="magma", aspect="auto")
        axis.set_xticks(np.arange(len(pivot.columns)), labels=[f"S{value:02d}" for value in pivot.columns])
        axis.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
        for row_index in range(len(pivot.index)):
            for column_index in range(len(pivot.columns)):
                value = pivot.iloc[row_index, column_index]
                axis.text(column_index, row_index, f"{value:.3f}", ha="center", va="center", color="white" if value < 0.6 else "black")
        axis.set_title("Macro-F1 by held-out subject")
        axis.set_xlabel("Test subject")
        figure.colorbar(image, ax=axis, label="Macro-F1")
        figure.tight_layout()
        figure.savefig(RESULTS / "subject_f1_comparison.png", dpi=200)
        plt.close(figure)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
