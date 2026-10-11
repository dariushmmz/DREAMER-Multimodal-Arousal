"""Audit DREAMER and create a deterministic subject-independent split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stress_classification.data import build_window_index, load_dreamer  # noqa: E402
from stress_classification.splits import SubjectSplit  # noqa: E402


def choose_balanced_split(
    frame,
    *,
    seed: int,
    validation_subjects: int = 3,
    test_subjects: int = 3,
    iterations: int = 100_000,
) -> tuple[SubjectSplit, float]:
    """Search deterministic group partitions for similar label distributions."""

    subjects = np.array(sorted(frame.subject_id.unique()), dtype=np.int32)
    counts = (
        frame.groupby(["subject_id", "label"]).size().unstack(fill_value=0)
        .reindex(index=subjects, columns=[0, 1, 2], fill_value=0)
        .to_numpy(dtype=np.float64)
    )
    total_counts = counts.sum(axis=0)
    global_distribution = total_counts / total_counts.sum()
    by_subject = {int(subject): counts[index] for index, subject in enumerate(subjects)}
    desired_fractions = np.array(
        [
            (len(subjects) - validation_subjects - test_subjects) / len(subjects),
            validation_subjects / len(subjects),
            test_subjects / len(subjects),
        ]
    )

    rng = np.random.default_rng(seed)
    best: tuple[float, SubjectSplit] | None = None
    for _ in range(iterations):
        shuffled = rng.permutation(subjects)
        validation = np.sort(shuffled[:validation_subjects])
        test = np.sort(shuffled[validation_subjects : validation_subjects + test_subjects])
        train = np.sort(shuffled[validation_subjects + test_subjects :])
        partitions = (train, validation, test)
        partition_counts = np.stack(
            [np.sum([by_subject[int(subject)] for subject in values], axis=0) for values in partitions]
        )
        if np.any(partition_counts == 0):
            continue
        distributions = partition_counts / partition_counts.sum(axis=1, keepdims=True)
        distribution_error = float(np.abs(distributions - global_distribution).sum())
        size_fractions = partition_counts.sum(axis=1) / total_counts.sum()
        size_error = float(np.abs(size_fractions - desired_fractions).sum())
        score = distribution_error + 0.25 * size_error
        candidate = SubjectSplit(
            train=tuple(int(value) for value in train),
            validation=tuple(int(value) for value in validation),
            test=tuple(int(value) for value in test),
        )
        tie_break = (candidate.validation, candidate.test)
        if best is None or (score, tie_break) < (best[0], (best[1].validation, best[1].test)):
            best = (score, candidate)
    if best is None:
        raise RuntimeError("failed to find a split with all classes in every partition")
    return best[1], best[0]


def summary(frame, split: SubjectSplit) -> dict[str, object]:
    partitions: dict[str, object] = {}
    for name, subject_ids in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        subset = frame[frame.subject_id.isin(subject_ids)]
        class_counts = subset.label.value_counts().reindex([0, 1, 2], fill_value=0)
        partitions[name] = {
            "subjects": list(subject_ids),
            "n_subjects": len(subject_ids),
            "n_windows": int(len(subset)),
            "class_counts": {str(key): int(value) for key, value in class_counts.items()},
            "class_proportions": {
                str(key): round(float(value / len(subset)), 6) for key, value in class_counts.items()
            },
        }
    return {
        "n_subjects": int(frame.subject_id.nunique()),
        "n_trials": int(frame[["subject_id", "trial_id"]].drop_duplicates().shape[0]),
        "n_windows": int(len(frame)),
        "target": str(frame.target.iloc[0]),
        "window_seconds": float(frame.window_seconds.iloc[0]),
        "overlap_ratio": 0.0,
        "label_definition": {"0": "rating 1-2", "1": "rating 3", "2": "rating 4-5"},
        "split": split.as_dict(),
        "partitions": partitions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=PROJECT_ROOT / "data" / "DREAMER.mat")
    parser.add_argument("--target", default="arousal", choices=("arousal", "valence", "dominance"))
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--iterations", type=int, default=100_000)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dreamer(args.mat)
    frame = build_window_index(data, target=args.target, window_seconds=args.window_seconds)
    split, objective = choose_balanced_split(frame, seed=args.seed, iterations=args.iterations)
    report = summary(frame, split)
    report["split_search"] = {"seed": args.seed, "iterations": args.iterations, "objective": objective}
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, subject_ids in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        ):
            frame.loc[frame.subject_id.isin(subject_ids), "partition"] = name
        frame.to_csv(args.output_dir / "window_index.csv", index=False)
        (args.output_dir / "dataset_summary.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (args.output_dir / "subject_split.json").write_text(
            json.dumps(split.as_dict(), indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

