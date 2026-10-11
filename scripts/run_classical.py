"""Train leakage-safe classical baselines on the fixed DREAMER split."""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stress_classification.config import load_config  # noqa: E402
from stress_classification.data import load_dreamer  # noqa: E402
from stress_classification.evaluation import save_evaluation  # noqa: E402
from stress_classification.features import build_feature_matrix, feature_names  # noqa: E402
from stress_classification.splits import SubjectSplit  # noqa: E402


def parameter_grid(values: dict[str, list]) -> list[dict[str, object]]:
    keys = list(values)
    return [dict(zip(keys, combination)) for combination in itertools.product(*(values[key] for key in keys))]


def create_pipeline(name: str, parameters: dict[str, object], seed: int) -> Pipeline:
    if name == "logistic_regression":
        classifier = LogisticRegression(
            C=float(parameters["C"]),
            max_iter=3000,
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
    if name == "svm_rbf":
        classifier = SVC(
            C=float(parameters["C"]),
            gamma=parameters["gamma"],
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
    if name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=int(parameters["n_estimators"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
        return Pipeline([("classifier", classifier)])
    if name == "knn":
        classifier = KNeighborsClassifier(
            n_neighbors=int(parameters["n_neighbors"]), weights="distance", n_jobs=-1
        )
        return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
    raise KeyError(name)


def load_or_build_features(config: dict, index: pd.DataFrame, cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        sample_ids = cached["sample_ids"].astype(str)
        if not np.array_equal(sample_ids, index.sample_id.astype(str).to_numpy()):
            raise ValueError("feature cache sample IDs do not match the window index")
        return cached["features"], cached["labels"]
    data = load_dreamer(PROJECT_ROOT / config["dataset"]["mat_path"])

    def progress(done: int, total: int) -> None:
        print(f"[features] {done}/{total}", flush=True)

    features, labels = build_feature_matrix(data, index, progress=progress)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        features=features,
        labels=labels,
        sample_ids=np.asarray(index.sample_id.astype(str).tolist(), dtype="U32"),
        feature_names=np.asarray(feature_names(), dtype="U64"),
    )
    return features, labels


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    config = load_config(args.config)
    seed = int(config["experiment"]["seed"])
    np.random.seed(seed)
    split = SubjectSplit.from_mapping(config["split"])
    index = pd.read_csv(args.window_index)
    split.validate(index.subject_id.unique())
    if index.duplicated("sample_id").any():
        raise ValueError("duplicate sample identifiers")
    features, labels = load_or_build_features(config, index, args.feature_cache)
    if not np.array_equal(labels, index.label.to_numpy(dtype=np.int64)):
        raise ValueError("feature labels do not match the tracked window index")
    masks = split.masks(index.subject_id.to_numpy())
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    selection_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for model_name, grid in config["classical"]["candidates"].items():
        best: tuple[float, Pipeline, dict[str, object]] | None = None
        for parameters in parameter_grid(grid):
            pipeline = create_pipeline(model_name, parameters, seed)
            pipeline.fit(features[masks["train"]], labels[masks["train"]])
            validation_prediction = pipeline.predict(features[masks["validation"]])
            score = float(
                f1_score(labels[masks["validation"]], validation_prediction, average="macro")
            )
            row = {"model": model_name, "validation_f1_macro": score, **parameters}
            selection_rows.append(row)
            if best is None or score > best[0]:
                best = (score, pipeline, parameters)
        assert best is not None
        _, pipeline, parameters = best
        model_dir = output / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        probabilities = pipeline.predict_proba(features[masks["test"]])
        metrics = save_evaluation(
            model_dir,
            index.loc[masks["test"]].reset_index(drop=True),
            labels[masks["test"]],
            probabilities,
        )
        joblib.dump(pipeline, model_dir / "model.joblib")
        (model_dir / "hyperparameters.json").write_text(
            json.dumps(parameters, indent=2) + "\n", encoding="utf-8"
        )
        window = metrics["window_level"]
        comparison_rows.append(
            {
                "model": model_name,
                "validation_f1_macro": best[0],
                "test_accuracy": window["accuracy"],
                "test_balanced_accuracy": window["balanced_accuracy"],
                "test_f1_macro": window["f1_macro"],
                "test_roc_auc_ovr_macro": window.get("roc_auc_ovr_macro"),
            }
        )
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(output / "validation_model_selection.csv", index=False)
    comparison = pd.DataFrame(comparison_rows).sort_values("test_f1_macro", ascending=False)
    comparison.to_csv(output / "comparison.csv", index=False)
    figure, axis = plt.subplots(figsize=(9, 5))
    comparison.set_index("model")[["test_accuracy", "test_balanced_accuracy", "test_f1_macro"]].plot.bar(
        ax=axis
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Classical models — subject-independent test set")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output / "comparison.png", dpi=200)
    plt.close(figure)
    runtime = time.perf_counter() - started
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "runtime_seconds": runtime,
        "execution_environment": "local_cpu",
    }
    (output / "run_metadata.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(f"runtime_seconds={runtime:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "dreamer_arousal_3class.yaml")
    parser.add_argument("--window-index", type=Path, default=PROJECT_ROOT / "data" / "processed" / "window_index.csv")
    parser.add_argument("--feature-cache", type=Path, default=PROJECT_ROOT / "data" / "processed" / "handcrafted_features.npz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "corrected" / "dreamer_arousal_3class" / "classical",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
