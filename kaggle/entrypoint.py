"""Entrypoint embedded into the reproducible Kaggle notebook kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXPECTED_RAW_BYTES = 452_847_973
EXPECTED_RAW_SHA256 = "35eed939e6f206a0938b13ba063178e2795ef1739b9cac8e5704a968393f3463"
EXPECTED_SAMPLE_ID_SHA256 = "91cb51e83cb5fa130d6bd9095b93047d9b383f01b213eb0ed334437adee742be"
EXPECTED_SHAPES = {
    "eeg": [8372, 1280, 14],
    "ecg": [8372, 2560, 1],
    "hrv": [8372, 25],
    "labels": [8372],
}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKING = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd=PROJECT_ROOT)


def find_raw_dreamer() -> tuple[Path, list[dict[str, object]]]:
    candidates = sorted(INPUT.rglob("DREAMER.mat"), key=lambda path: (len(path.parts), str(path)))
    if not candidates:
        raise FileNotFoundError(f"DREAMER.mat not found below {INPUT}")
    checks = []
    selected = None
    for path in candidates:
        digest = sha256(path)
        record = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest}
        checks.append(record)
        if path.stat().st_size == EXPECTED_RAW_BYTES and digest == EXPECTED_RAW_SHA256 and selected is None:
            selected = path
    if selected is None:
        raise RuntimeError(f"none of the Kaggle DREAMER files match the corrected source: {checks}")
    return selected, checks


def write_environment(path: Path, **extra: object) -> None:
    try:
        import numpy
        import pandas
        import scipy
        import sklearn
        import torch

        packages = {
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        }
    except Exception as error:  # pragma: no cover - diagnostic path
        packages = {"inspection_error": repr(error)}
    payload = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_environment": "kaggle",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def preprocess() -> None:
    started = time.perf_counter()
    raw, raw_checks = find_raw_dreamer()
    output = WORKING / "cache_bundle"
    output.mkdir(parents=True, exist_ok=True)
    config = PROJECT_ROOT / "configs" / "dreamer_arousal_3class.yaml"
    run(
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "audit_dreamer.py"),
        "--mat",
        str(raw),
        "--target",
        "arousal",
        "--window-seconds",
        "10",
        "--seed",
        "20261011",
        "--output-dir",
        str(output),
    )
    run(
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "prepare_deep_cache.py"),
        "--config",
        str(config),
        "--mat",
        str(raw),
        "--window-index",
        str(output / "window_index.csv"),
        "--output-dir",
        str(output / "deep_cache"),
    )
    manifest = json.loads((output / "deep_cache" / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "dataset_summary.json").read_text(encoding="utf-8"))
    split = json.loads((output / "subject_split.json").read_text(encoding="utf-8"))
    if manifest["sample_id_sha256"] != EXPECTED_SAMPLE_ID_SHA256:
        raise RuntimeError("Kaggle sample identifiers differ from the corrected protocol")
    if manifest["shapes"] != EXPECTED_SHAPES:
        raise RuntimeError(f"unexpected cache shapes: {manifest['shapes']}")
    expected_split = {
        "train": [2, 3, 4, 7, 8, 9, 10, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        "validation": [1, 6, 11],
        "test": [5, 13, 14],
    }
    if {key: split[key] for key in expected_split} != expected_split:
        raise RuntimeError(f"subject split mismatch: {split}")
    shutil.copy2(config, output / "config.yaml")
    verification = {
        "dataset_source": "phhasian0710/dreamer",
        "selected_raw_file": str(raw),
        "raw_candidates": raw_checks,
        "expected_raw_size_bytes": EXPECTED_RAW_BYTES,
        "expected_raw_sha256": EXPECTED_RAW_SHA256,
        "raw_source_verified": True,
        "sample_id_sha256": manifest["sample_id_sha256"],
        "shapes": manifest["shapes"],
        "n_windows": summary["n_windows"],
        "partitions": summary["partitions"],
        "split": split,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "kaggle_dataset_verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    write_environment(output / "kaggle_preprocessing_environment.json")
    print(json.dumps(verification, indent=2), flush=True)


def find_cache_bundle() -> Path:
    manifests = sorted(INPUT.rglob("deep_cache/manifest.json"))
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("sample_id_sha256") == EXPECTED_SAMPLE_ID_SHA256
            and manifest.get("shapes") == EXPECTED_SHAPES
        ):
            return manifest_path.parent.parent
    raise FileNotFoundError("verified stressclassification cache kernel output is not attached")


def train(model: str) -> None:
    import torch

    if model not in {"base_cnn", "paper_inspired_attention"}:
        raise ValueError(model)
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is not available to this notebook")
    raw, raw_checks = find_raw_dreamer()
    cache = find_cache_bundle()
    destination = WORKING / "results" / model
    destination.mkdir(parents=True, exist_ok=True)
    os.environ["EXECUTION_ENVIRONMENT"] = "kaggle_gpu"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_deep.py"),
        "--model",
        model,
        "--config",
        str(cache / "config.yaml"),
        "--window-index",
        str(cache / "window_index.csv"),
        "--cache-dir",
        str(cache / "deep_cache"),
        "--output-dir",
        str(destination),
        "--device",
        "cuda",
    ]
    resume_candidates = sorted(INPUT.rglob(f"results/{model}/last_checkpoint.pt"))
    if resume_candidates:
        command.extend(["--resume-checkpoint", str(resume_candidates[0])])
    run(*command)
    manifest = json.loads((cache / "deep_cache" / "manifest.json").read_text(encoding="utf-8"))
    infrastructure = {
        "dataset_source": "phhasian0710/dreamer",
        "raw_file": str(raw),
        "raw_candidates": raw_checks,
        "raw_sha256": EXPECTED_RAW_SHA256,
        "cache_manifest_sha256": sha256(cache / "deep_cache" / "manifest.json"),
        "cache_sample_id_sha256": manifest["sample_id_sha256"],
        "model": model,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "resume_checkpoint": str(resume_candidates[0]) if resume_candidates else None,
    }
    (destination / "kaggle_infrastructure.json").write_text(
        json.dumps(infrastructure, indent=2) + "\n", encoding="utf-8"
    )
    write_environment(destination / "kaggle_environment.json", **infrastructure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preprocess", "train"), required=True)
    parser.add_argument("--model", choices=("base_cnn", "paper_inspired_attention"))
    args = parser.parse_args()
    if args.mode == "preprocess":
        preprocess()
    else:
        if args.model is None:
            parser.error("--model is required for train mode")
        train(args.model)


if __name__ == "__main__":
    main()
