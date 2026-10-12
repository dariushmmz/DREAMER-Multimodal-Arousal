from __future__ import annotations

import numpy as np
import pytest

from stress_classification.evaluation import classification_metrics, normalize_probabilities


def test_float16_softmax_scores_are_normalized_for_multiclass_auc() -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    scores = np.array(
        [
            [0.8003, 0.1499, 0.0499],
            [0.6997, 0.2001, 0.1000],
            [0.1000, 0.7998, 0.1001],
            [0.2000, 0.7002, 0.0999],
            [0.0500, 0.1500, 0.7998],
            [0.1000, 0.2000, 0.7002],
        ],
        dtype=np.float16,
    )

    normalized = normalize_probabilities(scores)
    metrics = classification_metrics(y_true, normalized.argmax(axis=1), scores)

    np.testing.assert_allclose(normalized.sum(axis=1), 1.0)
    assert metrics["roc_auc_ovr_macro"] == pytest.approx(1.0)


def test_invalid_probability_rows_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive sum"):
        normalize_probabilities(np.zeros((2, 3)))
