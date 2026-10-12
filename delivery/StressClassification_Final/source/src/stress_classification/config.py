"""Configuration loading with subject-split validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from .splits import SubjectSplit


def load_config(path: str | Path) -> dict:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {source}")
    SubjectSplit.from_mapping(config["split"]).validate(range(1, 24))
    if float(config["dataset"].get("overlap_ratio", -1)) != 0.0:
        raise ValueError("corrected protocol requires overlap_ratio=0.0")
    if config["dataset"].get("target") != "arousal":
        raise ValueError("comparison configuration must use the common arousal target")
    return config

