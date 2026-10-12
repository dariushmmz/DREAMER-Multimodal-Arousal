"""Create reusable float16 signal arrays for local or Modal deep training."""

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
from stress_classification.data import load_dreamer  # noqa: E402
from stress_classification.deep_data import write_deep_cache  # noqa: E402
from stress_classification.splits import SubjectSplit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "dreamer_arousal_3class.yaml")
    parser.add_argument("--mat", type=Path, help="Override the raw DREAMER .mat path from the config")
    parser.add_argument("--window-index", type=Path, default=PROJECT_ROOT / "data" / "processed" / "window_index.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "deep_cache")
    args = parser.parse_args()
    config = load_config(args.config)
    split = SubjectSplit.from_mapping(config["split"])
    index = pd.read_csv(args.window_index)
    mat_path = args.mat or (PROJECT_ROOT / config["dataset"]["mat_path"])
    data = load_dreamer(mat_path)
    manifest = write_deep_cache(data, index, args.output_dir, split)
    print(json.dumps(manifest["shapes"], indent=2))


if __name__ == "__main__":
    main()
