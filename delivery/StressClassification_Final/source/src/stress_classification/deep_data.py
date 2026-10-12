"""Precomputed non-overlapping arrays and PyTorch datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data import ECG_FS, EEG_FS, iter_trials
from .preprocessing import bandpass, extract_hrv_features, window_zscore
from .splits import SubjectSplit


ARRAY_FILES = {
    "eeg": "eeg.npy",
    "ecg": "ecg.npy",
    "hrv": "hrv.npy",
    "labels": "labels.npy",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_deep_cache(
    data,
    window_index: pd.DataFrame,
    output_dir: str | Path,
    split: SubjectSplit,
    *,
    progress_interval: int = 250,
) -> dict[str, object]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    index = window_index.reset_index(drop=True)
    count = len(index)
    eeg_length = int(index.eeg_end.iloc[0] - index.eeg_start.iloc[0])
    ecg_length = int(index.ecg_end.iloc[0] - index.ecg_start.iloc[0])
    eeg_map = np.lib.format.open_memmap(
        destination / ARRAY_FILES["eeg"], mode="w+", dtype=np.float16, shape=(count, eeg_length, 14)
    )
    ecg_map = np.lib.format.open_memmap(
        destination / ARRAY_FILES["ecg"], mode="w+", dtype=np.float16, shape=(count, ecg_length, 1)
    )
    hrv_map = np.lib.format.open_memmap(
        destination / ARRAY_FILES["hrv"], mode="w+", dtype=np.float32, shape=(count, 25)
    )
    label_map = np.lib.format.open_memmap(
        destination / ARRAY_FILES["labels"], mode="w+", dtype=np.int64, shape=(count,)
    )
    by_trial = {
        (int(subject), int(trial)): group.sort_values("window_id")
        for (subject, trial), group in index.groupby(["subject_id", "trial_id"], sort=False)
    }
    completed = 0
    for trial in iter_trials(data):
        rows = by_trial.get((trial.subject_id, trial.trial_id))
        if rows is None:
            continue
        # Every trial belongs to exactly one subject partition, so trial-level
        # filtering cannot cross train/validation/test boundaries.
        filtered_eeg = bandpass(trial.eeg, 1.0, 45.0, EEG_FS)
        filtered_ecg = bandpass(trial.ecg[:, :1], 0.5, 40.0, ECG_FS)
        for original_index, row in rows.iterrows():
            eeg = filtered_eeg[int(row.eeg_start) : int(row.eeg_end)]
            ecg = filtered_ecg[int(row.ecg_start) : int(row.ecg_end)]
            eeg_map[original_index] = window_zscore(eeg).astype(np.float16)
            ecg_map[original_index] = window_zscore(ecg).astype(np.float16)
            hrv_map[original_index] = extract_hrv_features(ecg, fs=ECG_FS)
            label_map[original_index] = int(row.label)
            completed += 1
            if completed == count or completed % progress_interval == 0:
                print(f"[deep-cache] {completed}/{count}", flush=True)
    for values in (eeg_map, ecg_map, hrv_map, label_map):
        values.flush()
    del eeg_map, ecg_map, hrv_map, label_map
    if completed != count:
        raise RuntimeError(f"wrote {completed} arrays for {count} index rows")

    hrv = np.load(destination / ARRAY_FILES["hrv"], mmap_mode="r")
    train_mask = np.isin(index.subject_id.to_numpy(), split.train)
    hrv_mean = np.asarray(hrv[train_mask], dtype=np.float64).mean(axis=0)
    hrv_std = np.asarray(hrv[train_mask], dtype=np.float64).std(axis=0)
    hrv_std = np.maximum(hrv_std, 1e-6)
    manifest = {
        "n_samples": count,
        "sample_id_sha256": hashlib.sha256(
            "\n".join(index.sample_id.astype(str)).encode("utf-8")
        ).hexdigest(),
        "shapes": {
            "eeg": [count, eeg_length, 14],
            "ecg": [count, ecg_length, 1],
            "hrv": [count, 25],
            "labels": [count],
        },
        "dtypes": {"eeg": "float16", "ecg": "float16", "hrv": "float32", "labels": "int64"},
        "hrv_scaler": {"fit_partition": "train", "mean": hrv_mean.tolist(), "std": hrv_std.tolist()},
        "split": split.as_dict(),
        "files": {},
    }
    for key, filename in ARRAY_FILES.items():
        path = destination / filename
        manifest["files"][key] = {
            "name": filename,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


class CachedMultimodalDataset(Dataset):
    def __init__(
        self,
        cache_dir: str | Path,
        indices: np.ndarray,
        *,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.augment = augment
        self.seed = int(seed)
        manifest = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        self.hrv_mean = torch.tensor(manifest["hrv_scaler"]["mean"], dtype=torch.float32)
        self.hrv_std = torch.tensor(manifest["hrv_scaler"]["std"], dtype=torch.float32)
        self.eeg = np.load(self.cache_dir / ARRAY_FILES["eeg"], mmap_mode="r")
        self.ecg = np.load(self.cache_dir / ARRAY_FILES["ecg"], mmap_mode="r")
        self.hrv = np.load(self.cache_dir / ARRAY_FILES["hrv"], mmap_mode="r")
        self.labels = np.load(self.cache_dir / ARRAY_FILES["labels"], mmap_mode="r")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        index = int(self.indices[item])
        eeg = torch.tensor(np.asarray(self.eeg[index], dtype=np.float32))
        ecg = torch.tensor(np.asarray(self.ecg[index], dtype=np.float32))
        hrv = torch.tensor(np.asarray(self.hrv[index], dtype=np.float32))
        hrv = (hrv - self.hrv_mean) / self.hrv_std
        label = torch.tensor(int(self.labels[index]), dtype=torch.long)
        if self.augment:
            # Electrode/channel order is never shuffled: EEG channels are not
            # exchangeable. Noise and amplitude changes are window-local.
            eeg = eeg * torch.empty(1).uniform_(0.9, 1.1) + torch.randn_like(eeg) * 0.02
            ecg = ecg * torch.empty(1).uniform_(0.9, 1.1) + torch.randn_like(ecg) * 0.02
        return eeg, ecg, hrv, label, index

