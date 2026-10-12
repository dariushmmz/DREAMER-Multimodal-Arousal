"""Subject-aware adaptation of the provided 2023 paper's stacking method."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stress_classification.config import load_config  # noqa: E402
from stress_classification.evaluation import save_evaluation  # noqa: E402
from stress_classification.splits import SubjectSplit  # noqa: E402


def base_estimators(seed: int) -> dict[str, Pipeline]:
    """Algorithms reported in the paper, with fixed train-only ANOVA selection."""

    select = lambda: SelectKBest(score_func=f_classif, k=64)
    scale = lambda: StandardScaler()
    return {
        "naive_bayes": Pipeline([("select", select()), ("scale", scale()), ("classifier", GaussianNB())]),
        "logistic_regression": Pipeline(
            [
                ("select", select()),
                ("scale", scale()),
                (
                    "classifier",
                    LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", random_state=seed),
                ),
            ]
        ),
        "adaboost": Pipeline(
            [
                ("select", select()),
                ("classifier", AdaBoostClassifier(n_estimators=100, random_state=seed)),
            ]
        ),
        "knn": Pipeline(
            [
                ("select", select()),
                ("scale", scale()),
                ("classifier", KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("select", select()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            [
                ("select", select()),
                ("scale", scale()),
                (
                    "classifier",
                    SVC(
                        kernel="linear",
                        C=1.0,
                        probability=True,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "rbf_svm": Pipeline(
            [
                ("select", select()),
                ("scale", scale()),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        probability=True,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def stacked_features(models: dict[str, Pipeline], x: np.ndarray) -> np.ndarray:
    return np.concatenate([model.predict_proba(x) for model in models.values()], axis=1)


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    config = load_config(args.config)
    seed = int(config["experiment"]["seed"])
    index = pd.read_csv(args.window_index)
    cache = np.load(args.feature_cache, allow_pickle=False)
    features, labels = cache["features"], cache["labels"]
    if not np.array_equal(cache["sample_ids"].astype(str), index.sample_id.astype(str)):
        raise ValueError("feature cache and index sample IDs differ")
    split = SubjectSplit.from_mapping(config["split"])
    masks = split.masks(index.subject_id.to_numpy())
    train_indices = np.flatnonzero(masks["train"])
    validation_indices = np.flatnonzero(masks["validation"])
    test_indices = np.flatnonzero(masks["test"])
    train_groups = index.subject_id.to_numpy()[train_indices]
    templates = base_estimators(seed)
    oof = np.zeros((len(train_indices), len(templates) * 3), dtype=np.float64)
    folds = GroupKFold(n_splits=5)
    fold_inventory = []
    for fold_id, (fit_local, holdout_local) in enumerate(
        folds.split(features[train_indices], labels[train_indices], groups=train_groups), start=1
    ):
        fit_subjects = sorted(np.unique(train_groups[fit_local]).astype(int).tolist())
        holdout_subjects = sorted(np.unique(train_groups[holdout_local]).astype(int).tolist())
        if set(fit_subjects) & set(holdout_subjects):
            raise RuntimeError("group leakage in stacking fold")
        fold_inventory.append(
            {"fold": fold_id, "fit_subjects": fit_subjects, "holdout_subjects": holdout_subjects}
        )
        for model_index, (name, template) in enumerate(templates.items()):
            print(f"[stacking] fold={fold_id} model={name}", flush=True)
            model = clone(template)
            model.fit(features[train_indices[fit_local]], labels[train_indices[fit_local]])
            oof[holdout_local, model_index * 3 : (model_index + 1) * 3] = model.predict_proba(
                features[train_indices[holdout_local]]
            )
    fitted_bases = {}
    for name, template in templates.items():
        print(f"[stacking] full-train model={name}", flush=True)
        model = clone(template)
        model.fit(features[train_indices], labels[train_indices])
        fitted_bases[name] = model
    validation_stack = stacked_features(fitted_bases, features[validation_indices])
    test_stack = stacked_features(fitted_bases, features[test_indices])
    best = None
    for c_value in (0.1, 1.0, 10.0):
        meta = LogisticRegression(
            C=c_value, max_iter=3000, class_weight="balanced", random_state=seed
        )
        meta.fit(oof, labels[train_indices])
        score = float(
            __import__("sklearn.metrics", fromlist=["f1_score"]).f1_score(
                labels[validation_indices], meta.predict(validation_stack), average="macro", zero_division=0
            )
        )
        if best is None or score > best[0]:
            best = (score, c_value, meta)
    assert best is not None
    validation_score, meta_c, meta = best
    probabilities = meta.predict_proba(test_stack)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = save_evaluation(
        args.output_dir,
        index.iloc[test_indices].reset_index(drop=True),
        labels[test_indices],
        probabilities,
    )
    joblib.dump({"base_models": fitted_bases, "meta_model": meta}, args.output_dir / "stack.joblib")
    protocol = {
        "status": "adapted_not_faithful_reproduction",
        "reason": (
            "DREAMER arousal is not the paper's EO/AC1/AC2 stress target, and DREAMER hardware/population differ."
        ),
        "paper_method_elements": [
            "ANOVA F-value feature ranking",
            "NB, LR, AdaBoost, kNN, RF, linear SVM, and RBF-SVM base learners",
            "logistic-regression stacking",
        ],
        "scientific_corrections": [
            "fixed subject-disjoint train/validation/test split",
            "group-disjoint out-of-fold predictions for meta-model training",
            "feature selection fit inside each training fold",
            "meta-model C selected on validation subjects only",
        ],
        "selected_feature_count": 64,
        "selected_meta_C": meta_c,
        "validation_f1_macro": validation_score,
        "runtime_seconds": time.perf_counter() - started,
        "base_models": list(templates),
        "folds": fold_inventory,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics["window_level"], indent=2))
    print(json.dumps(protocol, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "dreamer_arousal_3class.yaml")
    parser.add_argument("--window-index", type=Path, default=PROJECT_ROOT / "data" / "processed" / "window_index.csv")
    parser.add_argument("--feature-cache", type=Path, default=PROJECT_ROOT / "data" / "processed" / "handcrafted_features.npz")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "corrected" / "dreamer_arousal_3class" / "paper_2023_adapted_stacking",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
