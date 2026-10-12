"""Window-local physiological signal preprocessing and ECG-derived HRV."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks, welch


def bandpass(data: np.ndarray, low_hz: float, high_hz: float, fs: int, order: int = 4) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    if values.shape[0] < 3 * (2 * order + 1):
        raise ValueError("window is too short for zero-phase filtering")
    sos_b, sos_a = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs)
    return filtfilt(sos_b, sos_a, values, axis=0).astype(np.float32)


def window_zscore(data: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Normalize one window only; no information crosses sample partitions."""

    values = np.asarray(data, dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    return ((values - mean) / np.maximum(std, epsilon)).astype(np.float32)


def preprocess_eeg(eeg: np.ndarray, fs: int = 128) -> np.ndarray:
    return window_zscore(bandpass(eeg, 1.0, 45.0, fs=fs))


def preprocess_ecg(ecg: np.ndarray, fs: int = 256) -> np.ndarray:
    values = np.asarray(ecg, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    return window_zscore(bandpass(values[:, :1], 0.5, 40.0, fs=fs))


def detect_r_peaks(ecg: np.ndarray, fs: int = 256) -> np.ndarray:
    """Detect plausible R peaks, considering either signal polarity."""

    signal = np.asarray(ecg, dtype=np.float32).reshape(-1)
    if not np.isfinite(signal).all() or signal.size < fs * 2:
        return np.empty(0, dtype=np.int32)
    standardized = (signal - np.median(signal)) / (np.std(signal) + 1e-6)
    distance = max(1, int(round(0.30 * fs)))
    candidates = []
    for polarity in (1.0, -1.0):
        peaks, properties = find_peaks(
            polarity * standardized,
            distance=distance,
            prominence=0.5,
        )
        prominence = float(np.sum(properties.get("prominences", np.array([]))))
        rr = np.diff(peaks) / float(fs)
        plausible = int(np.sum((rr >= 0.30) & (rr <= 2.0)))
        candidates.append((plausible, prominence, peaks.astype(np.int32)))
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def rr_intervals(ecg: np.ndarray, fs: int = 256) -> tuple[np.ndarray, np.ndarray]:
    peaks = detect_r_peaks(ecg, fs=fs)
    if len(peaks) < 3:
        return np.empty(0, dtype=np.float32), peaks
    rr = np.diff(peaks).astype(np.float32) / float(fs)
    mask = (rr >= 0.30) & (rr <= 2.0)
    return rr[mask], peaks


def hrv_waveform(ecg: np.ndarray, fs: int = 256) -> np.ndarray:
    """Interpolate RR intervals inside one window for the deep HRV branch."""

    signal = np.asarray(ecg, dtype=np.float32).reshape(-1)
    peaks = detect_r_peaks(signal, fs=fs)
    if len(peaks) < 3:
        return np.zeros((len(signal), 1), dtype=np.float32)
    rr = np.diff(peaks).astype(np.float32) / float(fs)
    times = peaks[1:].astype(np.float32) / float(fs)
    mask = (rr >= 0.30) & (rr <= 2.0)
    if np.sum(mask) < 2:
        return np.zeros((len(signal), 1), dtype=np.float32)
    interpolation = interp1d(
        times[mask],
        rr[mask],
        kind="linear",
        bounds_error=False,
        fill_value=(float(rr[mask][0]), float(rr[mask][-1])),
    )
    axis = np.arange(len(signal), dtype=np.float32) / float(fs)
    return window_zscore(interpolation(axis)[:, None].astype(np.float32))


HRV_FEATURE_NAMES = (
    "mean_nn",
    "median_nn",
    "sdnn",
    "rmssd",
    "sdsd",
    "cvnn",
    "cvsd",
    "nn20",
    "pnn20",
    "nn50",
    "pnn50",
    "min_nn",
    "max_nn",
    "range_nn",
    "iqr_nn",
    "mean_hr",
    "std_hr",
    "min_hr",
    "max_hr",
    "vlf_power",
    "lf_power",
    "hf_power",
    "lf_hf_ratio",
    "sd1",
    "sd2",
)


def _band_power(frequencies: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies < high)
    if np.sum(mask) < 2:
        return 0.0
    return float(np.trapz(power[mask], frequencies[mask]))


def extract_hrv_features(ecg: np.ndarray, fs: int = 256) -> np.ndarray:
    """Return 25 ECG-derived HRV descriptors from a single window.

    Frequency-domain values from 10-second windows are included only for
    architectural compatibility. They are not interpreted as clinical HRV
    estimates because such short windows do not resolve VLF/LF reliably.
    """

    rr, _ = rr_intervals(ecg, fs=fs)
    if len(rr) < 2:
        return np.zeros(len(HRV_FEATURE_NAMES), dtype=np.float32)
    diff = np.diff(rr)
    mean_nn = float(np.mean(rr))
    median_nn = float(np.median(rr))
    sdnn = float(np.std(rr, ddof=1)) if len(rr) > 1 else 0.0
    rmssd = float(np.sqrt(np.mean(diff**2))) if len(diff) else 0.0
    sdsd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    cvnn = sdnn / (mean_nn + 1e-8)
    cvsd = rmssd / (mean_nn + 1e-8)
    nn20 = float(np.sum(np.abs(diff) > 0.020))
    nn50 = float(np.sum(np.abs(diff) > 0.050))
    pnn20 = nn20 / max(len(diff), 1)
    pnn50 = nn50 / max(len(diff), 1)
    minimum = float(np.min(rr))
    maximum = float(np.max(rr))
    iqr = float(np.percentile(rr, 75) - np.percentile(rr, 25))
    heart_rate = 60.0 / np.maximum(rr, 1e-6)

    vlf = lf = hf = 0.0
    if len(rr) >= 4:
        beat_times = np.cumsum(np.r_[0.0, rr]).astype(np.float64)[1:]
        if beat_times[-1] - beat_times[0] >= 2.0:
            time_axis = np.arange(beat_times[0], beat_times[-1], 0.25)
            if len(time_axis) >= 4:
                tachogram = np.interp(time_axis, beat_times, rr)
                frequencies, power = welch(
                    tachogram - np.mean(tachogram),
                    fs=4.0,
                    nperseg=min(256, len(tachogram)),
                )
                vlf = _band_power(frequencies, power, 0.0033, 0.04)
                lf = _band_power(frequencies, power, 0.04, 0.15)
                hf = _band_power(frequencies, power, 0.15, 0.40)

    variance = sdnn**2
    sd1 = float(np.sqrt(max(0.0, 0.5 * rmssd**2)))
    sd2 = float(np.sqrt(max(0.0, 2.0 * variance - 0.5 * rmssd**2)))
    result = np.array(
        [
            mean_nn,
            median_nn,
            sdnn,
            rmssd,
            sdsd,
            cvnn,
            cvsd,
            nn20,
            pnn20,
            nn50,
            pnn50,
            minimum,
            maximum,
            maximum - minimum,
            iqr,
            float(np.mean(heart_rate)),
            float(np.std(heart_rate)),
            float(np.min(heart_rate)),
            float(np.max(heart_rate)),
            vlf,
            lf,
            hf,
            lf / (hf + 1e-8),
            sd1,
            sd2,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

