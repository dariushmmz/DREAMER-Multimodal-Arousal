"""Build a checksum inventory of source, data, and experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / ".docs" / "artifact_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True).strip()


def main() -> None:
    tracked = set(git("ls-files").splitlines())
    candidates: set[Path] = set()
    for relative in (
        "README.md",
        "pyproject.toml",
        "configs",
        "notebooks",
        "src/stress_classification",
        "scripts",
        "kaggle",
        "modal",
        "tests",
        "data/processed",
        "results/corrected",
        ".paper",
        ".docs",
    ):
        path = PROJECT_ROOT / relative
        if path.is_file():
            candidates.add(path)
        elif path.is_dir():
            candidates.update(item for item in path.rglob("*") if item.is_file())
    dreamer = PROJECT_ROOT / "data" / "DREAMER.mat"
    if dreamer.is_file():
        candidates.add(dreamer)
    candidates.discard(OUTPUT)
    candidates = {path for path in candidates if ".temp" not in path.parts and "__pycache__" not in path.parts}
    entries = []
    for path in sorted(candidates, key=lambda value: value.as_posix().lower()):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if relative.startswith("results/legacy/") or relative.startswith(".docs/legacy/"):
            continue
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "git_tracked": relative in tracked,
            }
        )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git("rev-parse", "HEAD"),
        "baseline_commit": "2c7b2fd",
        "restructure_commit": "afc21a8",
        "pipeline_commit": "43ba248",
        "corrected_results_commit": "8aa7772",
        "kaggle_workflow_commit": "0509a1e",
        "probability_normalization_fix_commit": "60b84e8",
        "base_cnn_results_commit": "7055739",
        "attention_results_commit": "e035199",
        "entries": entries,
        "pending_after_approval": [],
    }
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
