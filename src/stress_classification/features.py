"""Leakage-safe hand-crafted feature extraction for classical models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from .data import ECG_FS, EEG_FS, iter_trials
from .preprocessing import bandpass, extract_hrv_features

EEG_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _safe_shape(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float32)
    return values[:, None] if values.ndim == 1 else values


def channel_statistics(signal: np.ndarray) -> np.ndarray:
    values = _safe_shape(signal)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    median = np.median(values, axis=0)
    rms = np.sqrt(np.mean(values**2, axis=0))
    peak_to_peak = np.ptp(values, axis=0)
    asymmetry = skew(values, axis=0, bias=False, nan_policy="omit")
    peakedness = kurtosis(values, axis=0, fisher=True, bias=False, nan_policy="omit")
    features = np.stack([mean, std, median, rms, peak_to_peak, asymmetry, peakedness], axis=1)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1).astype(np.float32)


def bandpower_features(signal: np.ndarray, fs: int, bands: dict[str, tuple[float, float]]) -> np.ndarray:
    values = _safe_shape(signal)
    frequencies, power = welch(values, fs=fs, axis=0, nperseg=min(fs * 2, len(values)))
    total_mask = (frequencies >= min(v[0] for v in bands.values())) & (
        frequencies < max(v[1] for v in bands.values())
    )
    total = np.trapz(power[total_mask], frequencies[total_mask], axis=0) + 1e-12
    output: list[np.ndarray] = []
    for low, high in bands.values():
        mask = (frequencies >= low) & (frequencies < high)
        absolute = np.trapz(power[mask], frequencies[mask], axis=0)
        output.extend([np.log1p(absolute), absolute / total])
    return np.stack(output, axis=1).reshape(-1).astype(np.float32)


def eeg_features(eeg: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [channel_statistics(eeg), bandpower_features(eeg, EEG_FS, EEG_BANDS)]
    ).astype(np.float32)


def ecg_features(ecg: np.ndarray) -> np.ndarray:
    cardiac_bands = {"very_low": (0.5, 5.0), "qrs": (5.0, 15.0), "high": (15.0, 40.0)}
    return np.concatenate(
        [
            channel_statistics(ecg),
            bandpower_features(ecg, ECG_FS, cardiac_bands),
            extract_hrv_features(ecg, fs=ECG_FS),
        ]
    ).astype(np.float32)


def feature_names() -> list[str]:
    statistics = ("mean", "std", "median", "rms", "ptp", "skew", "kurtosis")
    names: list[str] = []
    for channel in range(14):
        names.extend(f"eeg_ch{channel+1:02d}_{name}" for name in statistics)
    for channel in range(14):
        for band in EEG_BANDS:
            names.extend((f"eeg_ch{channel+1:02d}_{band}_log_power", f"eeg_ch{channel+1:02d}_{band}_relative"))
    names.extend(f"ecg_{name}" for name in statistics)
    for band in ("very_low", "qrs", "high"):
        names.extend((f"ecg_{band}_log_power", f"ecg_{band}_relative"))
    from .preprocessing import HRV_FEATURE_NAMES

    names.extend(f"hrv_{name}" for name in HRV_FEATURE_NAMES)
    return names


def build_feature_matrix(
    data,
    window_index: pd.DataFrame,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract unscaled features; scaling must occur inside a training pipeline."""

    ordered = window_index.reset_index(drop=True)
    by_trial = {
        (int(subject), int(trial)): group.sort_values("window_id")
        for (subject, trial), group in ordered.groupby(["subject_id", "trial_id"], sort=False)
    }
    result = np.empty((len(ordered), len(feature_names())), dtype=np.float32)
    total = len(ordered)
    completed = 0
    for trial in iter_trials(data):
        key = (trial.subject_id, trial.trial_id)
        if key not in by_trial:
            continue
        # Filtering a whole trial is computationally efficient and safe because
        # the subject-level partition keeps the complete trial in one split.
        filtered_eeg = bandpass(trial.eeg, 1.0, 45.0, EEG_FS)
        filtered_ecg = bandpass(trial.ecg[:, :1], 0.5, 40.0, ECG_FS)
        for original_index, row in by_trial[key].iterrows():
            eeg = filtered_eeg[int(row.eeg_start) : int(row.eeg_end)]
            ecg = filtered_ecg[int(row.ecg_start) : int(row.ecg_end)]
            result[original_index] = np.concatenate([eeg_features(eeg), ecg_features(ecg)])
            completed += 1
            if progress is not None and (completed == total or completed % 250 == 0):
                progress(completed, total)
    if completed != total:
        raise RuntimeError(f"extracted {completed} rows for a {total}-row index")
    return result, ordered["label"].to_numpy(dtype=np.int64)
