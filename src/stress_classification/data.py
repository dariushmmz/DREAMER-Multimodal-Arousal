"""DREAMER loading and non-overlapping window indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import scipy.io as sio

from .labels import score_to_three_class, validate_target

EEG_FS = 128
ECG_FS = 256
N_TRIALS = 18
N_SUBJECTS = 23


@dataclass(frozen=True)
class Trial:
    subject_id: int
    trial_id: int
    eeg: np.ndarray
    ecg: np.ndarray
    ratings: dict[str, int]


def load_dreamer(path: str | Path):
    """Load the MATLAB structure without mutating or pre-normalizing signals."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = sio.loadmat(source, squeeze_me=True, struct_as_record=False)
    if "DREAMER" not in raw:
        raise ValueError(f"{source} does not contain a DREAMER variable")
    data = np.atleast_1d(raw["DREAMER"].Data)
    if len(data) != N_SUBJECTS:
        raise ValueError(f"expected {N_SUBJECTS} DREAMER subjects, found {len(data)}")
    return data


def iter_trials(data) -> Iterator[Trial]:
    for subject_id, subject in enumerate(np.atleast_1d(data), start=1):
        ratings = {
            "valence": np.asarray(subject.ScoreValence, dtype=np.int8).reshape(-1),
            "arousal": np.asarray(subject.ScoreArousal, dtype=np.int8).reshape(-1),
            "dominance": np.asarray(subject.ScoreDominance, dtype=np.int8).reshape(-1),
        }
        for key, values in ratings.items():
            if len(values) != N_TRIALS:
                raise ValueError(
                    f"subject {subject_id} has {len(values)} {key} ratings; expected {N_TRIALS}"
                )
        for trial_index in range(N_TRIALS):
            eeg = np.asarray(np.atleast_1d(subject.EEG.stimuli)[trial_index], dtype=np.float32)
            ecg = np.asarray(np.atleast_1d(subject.ECG.stimuli)[trial_index], dtype=np.float32)
            if eeg.ndim != 2 or eeg.shape[1] != 14:
                raise ValueError(
                    f"unexpected EEG shape for S{subject_id:02d} T{trial_index+1:02d}: {eeg.shape}"
                )
            if ecg.ndim != 2 or ecg.shape[1] < 1:
                raise ValueError(
                    f"unexpected ECG shape for S{subject_id:02d} T{trial_index+1:02d}: {ecg.shape}"
                )
            yield Trial(
                subject_id=subject_id,
                trial_id=trial_index + 1,
                eeg=eeg,
                ecg=ecg,
                ratings={key: int(values[trial_index]) for key, values in ratings.items()},
            )


def synchronized_window_count(
    eeg_samples: int,
    ecg_samples: int,
    window_seconds: float,
) -> int:
    eeg_window = int(round(float(window_seconds) * EEG_FS))
    ecg_window = int(round(float(window_seconds) * ECG_FS))
    if eeg_window <= 0 or ecg_window <= 0:
        raise ValueError("window duration must be positive")
    return min(int(eeg_samples) // eeg_window, int(ecg_samples) // ecg_window)


def build_window_index(
    data,
    *,
    target: str = "arousal",
    window_seconds: float = 10.0,
) -> pd.DataFrame:
    """Create an auditable index of synchronized, non-overlapping windows."""

    target = validate_target(target)
    eeg_window = int(round(window_seconds * EEG_FS))
    ecg_window = int(round(window_seconds * ECG_FS))
    rows: list[dict[str, int | float | str]] = []
    for trial in iter_trials(data):
        count = synchronized_window_count(len(trial.eeg), len(trial.ecg), window_seconds)
        rating = trial.ratings[target]
        label = score_to_three_class(rating)
        for window_id in range(count):
            eeg_start = window_id * eeg_window
            ecg_start = window_id * ecg_window
            rows.append(
                {
                    "sample_id": f"S{trial.subject_id:02d}_T{trial.trial_id:02d}_W{window_id:03d}",
                    "subject_id": trial.subject_id,
                    "trial_id": trial.trial_id,
                    "window_id": window_id,
                    "eeg_start": eeg_start,
                    "eeg_end": eeg_start + eeg_window,
                    "ecg_start": ecg_start,
                    "ecg_end": ecg_start + ecg_window,
                    "window_seconds": float(window_seconds),
                    "target": target,
                    "rating": rating,
                    "label": label,
                }
            )
    frame = pd.DataFrame(rows)
    validate_window_index(frame)
    return frame


def validate_window_index(frame: pd.DataFrame) -> None:
    required = {
        "sample_id",
        "subject_id",
        "trial_id",
        "window_id",
        "eeg_start",
        "eeg_end",
        "ecg_start",
        "ecg_end",
        "label",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"window index is missing columns: {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError("duplicate sample_id values detected")
    for (_, _), group in frame.groupby(["subject_id", "trial_id"], sort=False):
        ordered = group.sort_values("window_id")
        if len(ordered) > 1:
            eeg_gap = ordered["eeg_start"].to_numpy()[1:] - ordered["eeg_end"].to_numpy()[:-1]
            ecg_gap = ordered["ecg_start"].to_numpy()[1:] - ordered["ecg_end"].to_numpy()[:-1]
            if np.any(eeg_gap < 0) or np.any(ecg_gap < 0):
                raise ValueError("overlapping windows detected")
    if not set(frame["label"].unique()).issubset({0, 1, 2}):
        raise ValueError("labels outside the three-class definition detected")


def window_arrays(trial: Trial, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Extract one synchronized window described by an index row."""

    if int(row.subject_id) != trial.subject_id or int(row.trial_id) != trial.trial_id:
        raise ValueError("window row does not belong to the supplied trial")
    eeg = trial.eeg[int(row.eeg_start) : int(row.eeg_end)]
    ecg = trial.ecg[int(row.ecg_start) : int(row.ecg_end), :1]
    return np.asarray(eeg, dtype=np.float32), np.asarray(ecg, dtype=np.float32)

