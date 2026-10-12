"""Verify saved predictions, metrics, comparison rows, and deep artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stress_classification.evaluation import classification_metrics  # noqa: E402

RESULTS = ROOT / "results" / "corrected" / "dreamer_arousal_3class"
MODELS = {
    "knn": RESULTS / "classical" / "knn",
    "random_forest": RESULTS / "classical" / "random_forest",
    "logistic_regression": RESULTS / "classical" / "logistic_regression",
    "svm_rbf": RESULTS / "classical" / "svm_rbf",
    "paper_2023_adapted_stacking": RESULTS / "paper_2023_adapted_stacking",
    "base_cnn": RESULTS / "base_cnn",
    "paper_inspired_attention": RESULTS / "paper_inspired_attention",
}
SCALARS = ("accuracy", "balanced_accuracy", "f1_macro", "roc_auc_ovr_macro")


def main() -> None:
    index = pd.read_csv(ROOT / "data" / "processed" / "window_index.csv")
    expected = index.loc[index["partition"] == "test"].reset_index(drop=True)
    comparison = pd.read_csv(RESULTS / "comparison_all.csv").set_index("model")
    verified: dict[str, object] = {}
    for name, directory in MODELS.items():
        predictions = pd.read_csv(directory / "predictions.csv")
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))[
            "window_level"
        ]
        if predictions["sample_id"].astype(str).tolist() != expected["sample_id"].astype(str).tolist():
            raise RuntimeError(f"{name}: sample IDs/order differ from fixed test index")
        if not np.array_equal(predictions["y_true"].to_numpy(), expected["label"].to_numpy()):
            raise RuntimeError(f"{name}: saved labels differ from fixed test labels")
        probability_columns = ["prob_low", "prob_medium", "prob_high"]
        probability = predictions[probability_columns].to_numpy(dtype=np.float64)
        predicted = probability.argmax(axis=1)
        if not np.array_equal(predicted, predictions["y_pred"].to_numpy()):
            raise RuntimeError(f"{name}: class predictions disagree with probabilities")
        recomputed = classification_metrics(predictions["y_true"], predicted, probability)
        for key in SCALARS:
            if not np.isclose(float(recomputed[key]), float(metrics[key]), rtol=0, atol=1e-12):
                raise RuntimeError(f"{name}: {key} differs from saved predictions")
        if recomputed["confusion_matrix"] != metrics["confusion_matrix"]:
            raise RuntimeError(f"{name}: confusion matrix differs from saved predictions")
        comparison_row = comparison.loc[name]
        mapping = {
            "accuracy": "test_accuracy",
            "balanced_accuracy": "test_balanced_accuracy",
            "f1_macro": "test_f1_macro",
            "roc_auc_ovr_macro": "test_roc_auc_ovr_macro",
        }
        for metric_key, column in mapping.items():
            if not np.isclose(float(metrics[metric_key]), float(comparison_row[column]), rtol=0, atol=1e-12):
                raise RuntimeError(f"{name}: comparison {column} is inconsistent")
        verified[name] = {
            "n_test_windows": len(predictions),
            **{key: metrics[key] for key in SCALARS},
        }

    deep_required = (
        "best_checkpoint.pt",
        "last_checkpoint.pt",
        "training_history.csv",
        "training_curves.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "run_metadata.json",
        "resolved_config.json",
        "model_architecture.txt",
        "artifact_validation.json",
    )
    for name in ("base_cnn", "paper_inspired_attention"):
        missing = [item for item in deep_required if not (MODELS[name] / item).is_file()]
        if missing:
            raise FileNotFoundError(f"{name}: missing artifacts {missing}")

    summary = {
        "status": "passed",
        "fixed_test_windows": len(expected),
        "models_verified": verified,
        "deep_required_artifacts_present": True,
    }
    (RESULTS / "verification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
