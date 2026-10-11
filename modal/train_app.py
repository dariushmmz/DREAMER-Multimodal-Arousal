"""Bounded Modal L4 training entry point.

This file prepares the job but does not run unless explicitly invoked with
`modal run modal/train_app.py --model ...`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

APP_NAME = "dreamer-arousal-subject-independent"
DATA_VOLUME_NAME = "stress-classification-dreamer-v1"
RESULT_VOLUME_NAME = "stress-classification-results-v1"
REMOTE_DATA_ROOT = Path("/input/dreamer_arousal_v1")
REMOTE_RESULT_ROOT = Path("/output/dreamer_arousal_v1")

project_root = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "scipy==1.15.2",
        "pandas==2.2.3",
        "scikit-learn==1.6.1",
        "matplotlib==3.9.2",
        "seaborn==0.13.2",
        "PyYAML==6.0.3",
        "torch==2.2.2",
    )
    .add_local_dir(project_root / "src", remote_path="/opt/project/src", copy=True)
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
result_volume = modal.Volume.from_name(RESULT_VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=16_384,
    timeout=2 * 60 * 60,
    scaledown_window=10,
    retries=0,
    max_containers=1,
    volumes={
        "/input": data_volume.with_mount_options(read_only=True),
        "/output": result_volume,
    },
)
def train(model_name: str) -> dict:
    if model_name not in {"base_cnn", "paper_inspired_attention"}:
        raise ValueError(f"unsupported model: {model_name}")
    sys.path.insert(0, "/opt/project/src")
    import pandas as pd

    from stress_classification.config import load_config
    from stress_classification.training import train_experiment

    os.environ["EXECUTION_ENVIRONMENT"] = "modal_l4"
    data_volume.reload()
    config = load_config(REMOTE_DATA_ROOT / "config.yaml")
    index = pd.read_csv(REMOTE_DATA_ROOT / "window_index.csv")
    output = REMOTE_RESULT_ROOT / model_name
    result = train_experiment(
        config,
        model_name=model_name,
        cache_dir=REMOTE_DATA_ROOT / "deep_cache",
        window_index=index,
        output_dir=output,
        device_name="cuda",
    )
    result_volume.commit()
    return result["metadata"]


@app.local_entrypoint()
def main(model: str = "base_cnn") -> None:
    print(train.remote(model))

