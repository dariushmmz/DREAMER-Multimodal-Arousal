"""Assemble the verified, client-ready delivery tree and deterministic ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DELIVERY_ROOT = PROJECT_ROOT / "delivery"
PACKAGE_NAME = "StressClassification_Final"
PACKAGE_ROOT = DELIVERY_ROOT / PACKAGE_NAME
ARCHIVE_PATH = DELIVERY_ROOT / f"{PACKAGE_NAME}.zip"

EXCLUDED_NAMES = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pt", ".joblib"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part in EXCLUDED_NAMES for part in relative.parts):
            continue
        if path.is_file() and path.suffix.lower() not in EXCLUDED_SUFFIXES:
            copy_file(path, destination / relative)


def prepare_destination() -> None:
    delivery = DELIVERY_ROOT.resolve()
    destination = PACKAGE_ROOT.resolve()
    if destination.parent != delivery or destination.name != PACKAGE_NAME:
        raise RuntimeError(f"unsafe delivery destination: {destination}")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)


def copy_reports() -> None:
    mapping = {
        "09_final_experimental_report.md": "FINAL_EXPERIMENTAL_REPORT.md",
        "09_final_experimental_report_fa.md": "FINAL_EXPERIMENTAL_REPORT_FA.md",
        "07_responses_to_reviewers_fa.md": "REVIEWER_RESPONSES_FA.md",
        "05_reference_paper_reproduction_and_differences.md": "REFERENCE_PAPER_COMPARISON.md",
        "02_dataset_target_overlap_and_leakage.md": "DATA_AND_LEAKAGE_AUDIT.md",
        "03_models_and_experiment_configuration.md": "MODEL_CONFIGURATIONS.md",
        "04_performance_and_comparison.md": "PERFORMANCE_SUMMARY.md",
        "06_modal_environment.md": "KAGGLE_EXECUTION_ENVIRONMENT.md",
        "08_artifact_inventory.md": "REPOSITORY_ARTIFACT_INVENTORY.md",
    }
    for source_name, destination_name in mapping.items():
        copy_file(PROJECT_ROOT / ".docs" / source_name, PACKAGE_ROOT / "reports" / destination_name)
    repository_manifest = PROJECT_ROOT / ".docs" / "artifact_manifest.json"
    if repository_manifest.is_file():
        copy_file(
            repository_manifest,
            PACKAGE_ROOT / "manifests" / "repository_artifact_manifest.json",
        )


def copy_source() -> None:
    source_root = PACKAGE_ROOT / "source"
    for file_name in ("README.md", "pyproject.toml", ".gitignore"):
        copy_file(PROJECT_ROOT / file_name, source_root / file_name)
    for directory in ("configs", "notebooks", "src", "scripts", "kaggle", "tests", "modal"):
        copy_tree(PROJECT_ROOT / directory, source_root / directory)


def copy_data_metadata() -> None:
    destination = PACKAGE_ROOT / "data"
    readme = PROJECT_ROOT / "data" / "README.md"
    if readme.is_file():
        copy_file(readme, destination / "README.md")
    processed = PROJECT_ROOT / "data" / "processed"
    for path in sorted(processed.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".json"}:
            copy_file(path, destination / "processed" / path.relative_to(processed))


def copy_results_and_models() -> None:
    corrected = PROJECT_ROOT / "results" / "corrected"
    copy_tree(corrected, PACKAGE_ROOT / "results" / "corrected")

    binary_root = PACKAGE_ROOT / "artifacts" / "model_binaries"
    for suffix in ("*.pt", "*.joblib"):
        for path in sorted(corrected.rglob(suffix)):
            copy_file(path, binary_root / path.relative_to(corrected))


def write_delivery_readme() -> None:
    text = """# StressClassification final client delivery

This package contains the final experimental reports in English and Persian
Markdown, Persian reviewer responses in Markdown, the 2026 reference-paper
comparison, reproducible source, validated corrected results, and fitted model
files. DOCX files are intentionally excluded from the delivery package.

## Start here

1. Read `reports/FINAL_EXPERIMENTAL_REPORT_FA.md` for the Persian client report
   or `reports/FINAL_EXPERIMENTAL_REPORT.md` for the English report.
2. Reviewer responses are in `reports/REVIEWER_RESPONSES_FA.md`.
3. `reports/REFERENCE_PAPER_COMPARISON.md` explains why the 2026 result is not
   directly comparable and why the implemented attention model is an adaptation.
4. Reproduction commands and project layout are in `source/README.md`.
5. Every delivered file is hashed in `manifests/package_manifest.json`.

The raw DREAMER file and derived 344 MiB tensor cache are not redistributed in
this package. Their verified hashes, expected locations, and reconstruction
method are documented in `manifests/CHECKPOINTS_AND_DATA.md`. Fitted PyTorch
and classical model binaries are included under `artifacts/model_binaries/`.
"""
    (PACKAGE_ROOT / "README_DELIVERY.md").write_text(text, encoding="utf-8")


def write_delivery_root_readme() -> None:
    text = """# Client delivery

`StressClassification_Final/` is the canonical delivery tree. It contains the
final Markdown reports, organized reproducible source snapshot, processed
metadata, validated result artifacts, figures, and local fitted-model files.
DOCX reports are intentionally excluded.

Run the following to rebuild the directory and deterministic archive:

```powershell
python scripts/reporting/build_delivery_package.py --zip
```

Large `*.pt` and `*.joblib` files inside the delivery tree and the generated ZIP
are intentionally excluded from Git. They remain available locally and are
covered by `StressClassification_Final/manifests/package_manifest.json` and
`delivery_checksums.json`.
"""
    (DELIVERY_ROOT / "README.md").write_text(text, encoding="utf-8")


def write_large_artifact_inventory() -> None:
    cache_manifest = json.loads(
        (PROJECT_ROOT / "data" / "processed" / "deep_cache" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    raw = PROJECT_ROOT / "data" / "DREAMER.mat"
    if not raw.is_file():
        raise FileNotFoundError("data/DREAMER.mat is required to verify delivery provenance")
    model_files = sorted((PACKAGE_ROOT / "artifacts" / "model_binaries").rglob("*"))
    lines = [
        "# Checkpoints, fitted models, and dataset inventory",
        "",
        "## Raw DREAMER dataset (distributed separately)",
        "",
        f"- Expected repository location: `data/DREAMER.mat`",
        f"- Size: {raw.stat().st_size:,} bytes",
        f"- SHA-256: `{sha256(raw)}`",
        "- Verified Kaggle source: `phhasian0710/dreamer`",
        "- The raw dataset is omitted from the client ZIP because of size and dataset distribution considerations.",
        "",
        "## Derived deep cache (reproducible, distributed separately)",
        "",
        "The cache is rebuilt with `python scripts/data/build_deep_cache.py` after placing the raw file above.",
        "",
        "| Array | Shape | Size (bytes) | SHA-256 |",
        "|---|---|---:|---|",
    ]
    for name, details in cache_manifest["files"].items():
        shape = "×".join(str(value) for value in cache_manifest["shapes"][name])
        lines.append(f"| {name} | {shape} | {details['size_bytes']:,} | `{details['sha256']}` |")
    lines.extend(
        [
            "",
            "## Delivered fitted-model files",
            "",
            "| File | Size (bytes) | SHA-256 |",
            "|---|---:|---|",
        ]
    )
    for path in model_files:
        if path.is_file():
            relative = path.relative_to(PACKAGE_ROOT).as_posix()
            lines.append(f"| `{relative}` | {path.stat().st_size:,} | `{sha256(path)}` |")
    lines.extend(
        [
            "",
            "The `best_checkpoint.pt` files are the validation-selected inference checkpoints.",
            "The `last_checkpoint.pt` files are included only for interruption-safe resume provenance.",
            "Classical `.joblib` files are the fitted estimators retained with the corrected experiments.",
        ]
    )
    destination = PACKAGE_ROOT / "manifests" / "CHECKPOINTS_AND_DATA.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_package_manifest() -> Path:
    manifest_path = PACKAGE_ROOT / "manifests" / "package_manifest.json"
    entries = []
    for path in sorted(PACKAGE_ROOT.rglob("*"), key=lambda item: item.as_posix().lower()):
        if path.is_file() and path != manifest_path:
            entries.append(
                {
                    "path": path.relative_to(PACKAGE_ROOT).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "package": PACKAGE_NAME,
        "source_git_head": git_head(),
        "manifest_excludes_itself": True,
        "file_count": len(entries),
        "entries": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def create_deterministic_zip() -> None:
    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    with zipfile.ZipFile(ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(PACKAGE_ROOT.rglob("*"), key=lambda item: item.as_posix().lower()):
            if not path.is_file():
                continue
            relative = Path(PACKAGE_NAME) / path.relative_to(PACKAGE_ROOT)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())


def write_delivery_checksums(manifest_path: Path, include_zip: bool) -> None:
    checksums = {
        "package_directory": PACKAGE_ROOT.name,
        "package_manifest_sha256": sha256(manifest_path),
        "archive": None,
    }
    if include_zip:
        checksums["archive"] = {
            "path": ARCHIVE_PATH.name,
            "size_bytes": ARCHIVE_PATH.stat().st_size,
            "sha256": sha256(ARCHIVE_PATH),
        }
    (DELIVERY_ROOT / "delivery_checksums.json").write_text(
        json.dumps(checksums, indent=2) + "\n", encoding="utf-8"
    )


def build(include_zip: bool) -> None:
    prepare_destination()
    copy_reports()
    copy_source()
    copy_data_metadata()
    copy_results_and_models()
    write_delivery_readme()
    write_delivery_root_readme()
    write_large_artifact_inventory()
    manifest_path = write_package_manifest()
    if include_zip:
        create_deterministic_zip()
    write_delivery_checksums(manifest_path, include_zip)
    print(f"delivery package: {PACKAGE_ROOT}")
    print(f"files: {json.loads(manifest_path.read_text())['file_count']}")
    if include_zip:
        print(f"archive: {ARCHIVE_PATH} ({ARCHIVE_PATH.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", action="store_true", help="also build a deterministic ZIP archive")
    args = parser.parse_args()
    build(args.zip)


if __name__ == "__main__":
    main()
