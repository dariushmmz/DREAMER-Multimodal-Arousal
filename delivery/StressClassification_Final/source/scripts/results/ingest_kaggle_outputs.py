"""Validate downloaded Kaggle artifacts and regenerate normalized evaluation files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stress_classification.evaluation import save_evaluation  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--kernel-log", type=Path)
    parser.add_argument("--model", choices=("base_cnn", "paper_inspired_attention"), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "corrected" / "dreamer_arousal_3class",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "data" / "processed" / "window_index.csv",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    destination = (args.output_root / args.model).resolve()
    if not (source / "metrics.json").is_file() or not (source / "predictions.csv").is_file():
        raise FileNotFoundError(f"incomplete Kaggle output: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(item, destination / item.name)
    embedded_source_sha256 = None
    if args.kernel_log is not None:
        kernel_log = args.kernel_log.resolve()
        shutil.copy2(kernel_log, destination / "kaggle_kernel.log")
        match = re.search(
            r"Embedded source SHA-256:\s*([0-9a-f]{64})",
            kernel_log.read_text(encoding="utf-8", errors="replace"),
        )
        if match is None:
            raise ValueError("kernel log does not contain the embedded source SHA-256")
        embedded_source_sha256 = match.group(1)

    remote_metrics = destination / "metrics.json"
    remote_predictions = destination / "predictions.csv"
    shutil.copy2(remote_metrics, destination / "metrics_kaggle_raw.json")
    shutil.copy2(remote_predictions, destination / "predictions_kaggle_raw.csv")
    original_metrics_sha = sha256(remote_metrics)
    original_predictions_sha = sha256(remote_predictions)

    predictions = pd.read_csv(remote_predictions)
    probability_columns = ["prob_low", "prob_medium", "prob_high"]
    required = ["sample_id", "y_true", "y_pred", *probability_columns]
    missing = [name for name in required if name not in predictions]
    if missing:
        raise ValueError(f"missing prediction columns: {missing}")
    expected = pd.read_csv(args.index)
    expected = expected.loc[expected["partition"] == "test", "sample_id"].astype(str).tolist()
    actual = predictions["sample_id"].astype(str).tolist()
    if actual != expected:
        raise RuntimeError("Kaggle predictions do not match the fixed test sample order")

    probabilities = predictions[probability_columns].to_numpy(dtype=np.float64)
    sums = probabilities.sum(axis=1)
    predicted = probabilities.argmax(axis=1)
    if not np.array_equal(predicted, predictions["y_pred"].to_numpy(dtype=np.int64)):
        raise RuntimeError("saved predictions disagree with saved class probabilities")
    metadata_columns = [
        name
        for name in predictions.columns
        if name not in {"y_true", "y_pred", *probability_columns}
    ]
    metrics = save_evaluation(
        destination,
        predictions[metadata_columns],
        predictions["y_true"].to_numpy(dtype=np.int64),
        probabilities,
    )
    validation = {
        "source_directory": str(source),
        "model": args.model,
        "embedded_source_sha256": embedded_source_sha256,
        "test_samples": len(predictions),
        "test_sample_order_matches_fixed_index": True,
        "saved_predictions_match_probability_argmax": True,
        "raw_probability_max_abs_sum_error": float(np.max(np.abs(sums - 1.0))),
        "postprocessing": "float64 row normalization before metrics and serialization",
        "remote_metrics_sha256": original_metrics_sha,
        "remote_predictions_sha256": original_predictions_sha,
        "normalized_metrics_sha256": sha256(destination / "metrics.json"),
        "normalized_predictions_sha256": sha256(destination / "predictions.csv"),
        "window_metrics": metrics["window_level"],
    }
    (destination / "artifact_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
