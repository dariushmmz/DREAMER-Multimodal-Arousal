"""Target definitions used consistently by every corrected experiment."""

from __future__ import annotations

from typing import Final

import numpy as np

SUPPORTED_TARGETS: Final = ("arousal", "valence", "dominance")
CLASS_NAMES: Final = ("low", "medium", "high")
LABEL_DEFINITION: Final = {
    0: "rating 1-2 (low)",
    1: "rating 3 (medium)",
    2: "rating 4-5 (high)",
}


def score_to_three_class(score: int | np.integer) -> int:
    """Map DREAMER's ordinal 1–5 self-rating to low/medium/high.

    This mapping does not turn arousal into a stress label. It only discretizes
    the original self-reported affect dimension.
    """

    value = int(score)
    if not 1 <= value <= 5:
        raise ValueError(f"DREAMER ratings must be in [1, 5], got {value}")
    if value <= 2:
        return 0
    if value == 3:
        return 1
    return 2


def validate_target(target: str) -> str:
    value = target.lower().strip()
    if value not in SUPPORTED_TARGETS:
        raise ValueError(f"target must be one of {SUPPORTED_TARGETS}, got {target!r}")
    return value

