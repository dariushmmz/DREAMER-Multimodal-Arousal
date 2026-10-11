import pandas as pd
import pytest

from stress_classification.data import validate_window_index


def _frame(second_start: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "S01_T01_W000",
                "subject_id": 1,
                "trial_id": 1,
                "window_id": 0,
                "eeg_start": 0,
                "eeg_end": 1280,
                "ecg_start": 0,
                "ecg_end": 2560,
                "label": 0,
            },
            {
                "sample_id": "S01_T01_W001",
                "subject_id": 1,
                "trial_id": 1,
                "window_id": 1,
                "eeg_start": second_start,
                "eeg_end": second_start + 1280,
                "ecg_start": second_start * 2,
                "ecg_end": second_start * 2 + 2560,
                "label": 0,
            },
        ]
    )


def test_nonoverlapping_index_passes():
    validate_window_index(_frame(1280))


def test_overlapping_index_fails():
    with pytest.raises(ValueError, match="overlapping"):
        validate_window_index(_frame(640))

