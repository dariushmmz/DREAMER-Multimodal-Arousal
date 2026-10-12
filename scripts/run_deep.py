"""Run one corrected deep experiment locally or inside Modal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stress_classification.config import load_config  # noqa: E402
from stress_classification.training import train_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("base_cnn", "paper_inspired_attention"), required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "dreamer_arousal_3class.yaml")
    parser.add_argument("--window-index", type=Path, default=PROJECT_ROOT / "data" / "processed" / "window_index.csv")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "deep_cache")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    output = args.output_dir or (
        PROJECT_ROOT / "results" / "corrected" / "dreamer_arousal_3class" / args.model
    )
    result = train_experiment(
        load_config(args.config),
        model_name=args.model,
        cache_dir=args.cache_dir,
        window_index=pd.read_csv(args.window_index),
        output_dir=output,
        device_name=args.device,
        resume_checkpoint=args.resume_checkpoint,
    )
    print(json.dumps(result["metadata"], indent=2))


if __name__ == "__main__":
    main()
