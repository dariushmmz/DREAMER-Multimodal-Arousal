"""Build self-contained private Kaggle notebooks from the canonical project source."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle" / "kernels"
OWNER = "amirmz"

PAYLOAD_PATHS = [
    ROOT / "configs" / "dreamer_arousal_3class.yaml",
    ROOT / "kaggle" / "entrypoint.py",
    ROOT / "scripts" / "data" / "audit_dataset.py",
    ROOT / "scripts" / "data" / "build_deep_cache.py",
    ROOT / "scripts" / "experiments" / "train_deep_model.py",
    *sorted((ROOT / "src" / "stress_classification").glob("*.py")),
]


def payload() -> tuple[str, str]:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in PAYLOAD_PATHS:
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    data = stream.getvalue()
    return base64.b64encode(data).decode("ascii"), hashlib.sha256(data).hexdigest()


def notebook_source(encoded: str, digest: str, *, mode: str, model: str | None) -> str:
    arguments = ["--mode", mode]
    if model is not None:
        arguments.extend(["--model", model])
    return f'''import base64, hashlib, io, subprocess, sys, zipfile
from pathlib import Path

PAYLOAD_SHA256 = {digest!r}
payload = base64.b64decode({encoded!r})
assert hashlib.sha256(payload).hexdigest() == PAYLOAD_SHA256
project = Path("/tmp/stressclassification")
project.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(payload)) as archive:
    archive.extractall(project)
command = [sys.executable, str(project / "kaggle" / "entrypoint.py"), {', '.join(repr(value) for value in arguments)}]
print("Embedded source SHA-256:", PAYLOAD_SHA256)
subprocess.run(command, check=True, cwd=project)
'''


def build(
    name: str,
    title: str,
    *,
    mode: str,
    model: str | None,
    gpu: bool,
    kernel_sources: list[str],
    id_slug: str | None = None,
) -> None:
    encoded, digest = payload()
    destination = OUTPUT / name
    destination.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# StressClassification corrected DREAMER experiment\\n",
                    "Generated from the canonical repository source. Do not edit this generated notebook directly.\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": notebook_source(encoded, digest, mode=mode, model=model).splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (destination / "notebook.ipynb").write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": f"{OWNER}/{id_slug or name}",
        "title": title,
        "code_file": "notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": False,
        "dataset_sources": ["phhasian0710/dreamer"],
        "competition_sources": [],
        "kernel_sources": kernel_sources,
    }
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "source_manifest.json").write_text(
        json.dumps(
            {
                "embedded_zip_sha256": digest,
                "source_files": [path.relative_to(ROOT).as_posix() for path in PAYLOAD_PATHS],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    cache_kernel = f"{OWNER}/stressclassification-dreamer-corrected-cache"
    build(
        "stressclassification-dreamer-cache",
        "StressClassification DREAMER Corrected Cache",
        mode="preprocess",
        model=None,
        gpu=False,
        kernel_sources=[],
        id_slug="stressclassification-dreamer-corrected-cache",
    )
    build(
        "stressclassification-base-cnn",
        "StressClassification Base CNN",
        mode="train",
        model="base_cnn",
        gpu=True,
        kernel_sources=[cache_kernel],
    )
    build(
        "stressclassification-attention",
        "StressClassification Paper-Inspired Attention",
        mode="train",
        model="paper_inspired_attention",
        gpu=True,
        kernel_sources=[cache_kernel],
        id_slug="stressclassification-paper-inspired-attention",
    )


if __name__ == "__main__":
    main()
