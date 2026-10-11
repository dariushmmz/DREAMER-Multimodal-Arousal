"""Quantify EEG/ECG/HRV contributions without calling correlation leakage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stress_classification.config import load_config  # noqa: E402
from stress_classification.evaluation import save_evaluation  # noqa: E402
from stress_classification.splits import SubjectSplit  # noqa: E402


def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    seed = int(config["experiment"]["seed"])
    index = pd.read_csv(args.window_index)
    cache = np.load(args.feature_cache, allow_pickle=False)
    features = cache["features"]
    labels = cache["labels"]
    names = cache["feature_names"].astype(str)
    if not np.array_equal(cache["sample_ids"].astype(str), index.sample_id.astype(str)):
        raise ValueError("feature cache and window index differ")
    split = SubjectSplit.from_mapping(config["split"])
    masks = split.masks(index.subject_id.to_numpy())
    blocks = {
        "eeg_only": np.flatnonzero(np.char.startswith(names, "eeg_")),
        "ecg_raw_only": np.flatnonzero(np.char.startswith(names, "ecg_")),
        "hrv_only": np.flatnonzero(np.char.startswith(names, "hrv_")),
        "ecg_plus_hrv": np.flatnonzero(
            np.char.startswith(names, "ecg_") | np.char.startswith(names, "hrv_")
        ),
        "eeg_plus_ecg_raw": np.flatnonzero(
            np.char.startswith(names, "eeg_") | np.char.startswith(names, "ecg_")
        ),
        "eeg_plus_ecg_plus_hrv": np.arange(features.shape[1]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for block_name, columns in blocks.items():
        best = None
        for leaf in (1, 2, 5):
            model = RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=leaf,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(features[masks["train"]][:, columns], labels[masks["train"]])
            validation = model.predict(features[masks["validation"]][:, columns])
            score = float(
                f1_score(labels[masks["validation"]], validation, average="macro", zero_division=0)
            )
            if best is None or score > best[0]:
                best = (score, leaf, model)
        assert best is not None
        score, leaf, model = best
        probability = model.predict_proba(features[masks["test"]][:, columns])
        metrics = save_evaluation(
            args.output_dir / block_name,
            index.loc[masks["test"]].reset_index(drop=True),
            labels[masks["test"]],
            probability,
        )
        window = metrics["window_level"]
        rows.append(
            {
                "feature_block": block_name,
                "n_features": len(columns),
                "selected_min_samples_leaf": leaf,
                "validation_f1_macro": score,
                "test_accuracy": window["accuracy"],
                "test_balanced_accuracy": window["balanced_accuracy"],
                "test_f1_macro": window["f1_macro"],
            }
        )
    comparison = pd.DataFrame(rows).sort_values("test_f1_macro", ascending=False)
    comparison.to_csv(args.output_dir / "comparison.csv", index=False)
    figure, axis = plt.subplots(figsize=(11, 5.5))
    comparison.set_index("feature_block")[["test_accuracy", "test_balanced_accuracy", "test_f1_macro"]].plot.bar(
        ax=axis
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Random-forest modality ablation — untouched test subjects")
    figure.tight_layout()
    figure.savefig(args.output_dir / "comparison.png", dpi=200)
    plt.close(figure)

    ecg_columns = blocks["ecg_raw_only"]
    hrv_columns = blocks["hrv_only"]
    training_frame = pd.DataFrame(
        features[masks["train"]][:, np.r_[ecg_columns, hrv_columns]],
        columns=names[np.r_[ecg_columns, hrv_columns]],
    )
    correlations = training_frame.corr(method="spearman").loc[names[ecg_columns], names[hrv_columns]]
    correlations.to_csv(args.output_dir / "ecg_hrv_spearman_train_only.csv")
    absolute = correlations.abs()
    relationship = {
        "interpretation": (
            "HRV features are deterministic transformations of ECG; association is expected physiological "
            "and mathematical dependence, not evidence that test labels entered training."
        ),
        "fit_partition": "train only",
        "max_absolute_spearman": float(absolute.max().max()),
        "median_absolute_spearman": float(np.nanmedian(absolute.to_numpy())),
        "strong_pair_count_abs_ge_0_8": int(np.sum(absolute.to_numpy() >= 0.8)),
    }
    (args.output_dir / "ecg_hrv_dependency_summary.json").write_text(
        json.dumps(relationship, indent=2) + "\n", encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(json.dumps(relationship, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "dreamer_arousal_3class.yaml")
    parser.add_argument("--window-index", type=Path, default=PROJECT_ROOT / "data" / "processed" / "window_index.csv")
    parser.add_argument("--feature-cache", type=Path, default=PROJECT_ROOT / "data" / "processed" / "handcrafted_features.npz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "corrected" / "dreamer_arousal_3class" / "modality_ablation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
