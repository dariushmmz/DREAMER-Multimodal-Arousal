"""Build a single comparable table from corrected experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "corrected" / "dreamer_arousal_3class"


def metric_row(name: str, family: str, path: Path) -> dict[str, object]:
    metrics = json.loads(path.read_text(encoding="utf-8"))["window_level"]
    return {
        "model": name,
        "family": family,
        "test_accuracy": metrics["accuracy"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_f1_macro": metrics["f1_macro"],
        "test_roc_auc_ovr_macro": metrics.get("roc_auc_ovr_macro"),
        "status": "completed",
    }


def main() -> None:
    classical = pd.read_csv(RESULTS / "classical" / "comparison.csv")
    rows = [
        {
            "model": row.model,
            "family": "classical_ml",
            "test_accuracy": row.test_accuracy,
            "test_balanced_accuracy": row.test_balanced_accuracy,
            "test_f1_macro": row.test_f1_macro,
            "test_roc_auc_ovr_macro": row.test_roc_auc_ovr_macro,
            "status": "completed",
        }
        for row in classical.itertuples()
    ]
    stacking = RESULTS / "paper_2023_adapted_stacking" / "metrics.json"
    if stacking.is_file():
        rows.append(metric_row("paper_2023_adapted_stacking", "classical_ensemble", stacking))
    for model in ("base_cnn", "paper_inspired_attention"):
        metrics = RESULTS / model / "metrics.json"
        if metrics.is_file():
            rows.append(metric_row(model, "deep_learning", metrics))
        else:
            rows.append(
                {
                    "model": model,
                    "family": "deep_learning",
                    "test_accuracy": None,
                    "test_balanced_accuracy": None,
                    "test_f1_macro": None,
                    "test_roc_auc_ovr_macro": None,
                    "status": "pending_modal_cost_approval",
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS / "comparison_all.csv", index=False)
    completed = comparison[comparison.status == "completed"].copy()
    figure, axis = plt.subplots(figsize=(11, 5.5))
    completed.set_index("model")[["test_accuracy", "test_balanced_accuracy", "test_f1_macro"]].plot.bar(
        ax=axis
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Corrected subject-independent three-class arousal results")
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(RESULTS / "comparison_all.png", dpi=200)
    plt.close(figure)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()

