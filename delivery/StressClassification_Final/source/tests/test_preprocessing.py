import numpy as np

from stress_classification.preprocessing import HRV_FEATURE_NAMES, extract_hrv_features, hrv_waveform


def synthetic_ecg(fs: int = 256, seconds: int = 10) -> np.ndarray:
    values = np.zeros(fs * seconds, dtype=np.float32)
    for location in range(fs, len(values), fs):
        values[location : location + 3] = (1.0, 2.0, 1.0)
    return values[:, None]


def test_hrv_outputs_are_finite_and_shaped():
    ecg = synthetic_ecg()
    vector = extract_hrv_features(ecg)
    waveform = hrv_waveform(ecg)
    assert vector.shape == (len(HRV_FEATURE_NAMES),)
    assert waveform.shape == ecg.shape
    assert np.isfinite(vector).all()
    assert np.isfinite(waveform).all()

