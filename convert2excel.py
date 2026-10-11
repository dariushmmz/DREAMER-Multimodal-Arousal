"""Convert DREAMER.mat to memory-safe, partitioned CSV files.

This writes one EEG and one ECG CSV per subject, plus small metadata files.
Only one trial/condition is converted to a DataFrame at a time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

N_TRIALS = 18
SAMPLING_RATES = {"EEG": 128, "ECG": 256}


def default_mat_path() -> Path:
    for path in (Path("data/DREAMER.mat"), Path(".data/DREAMER.mat"), Path("DREAMER.mat")):
        if path.exists():
            return path
    return Path("data/DREAMER.mat")


def scalar_text(value: object) -> str:
    value = np.asarray(value).squeeze()
    return str(value.item() if value.ndim == 0 else value)


def trial_frame(values: np.ndarray, subject: int, trial: int, condition: str,
                signal: str, labels: tuple[int, int, int]) -> pd.DataFrame:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    n_rows, n_channels = values.shape
    frame = pd.DataFrame(values, columns=[f"ch_{i}" for i in range(1, n_channels + 1)])
    frame.insert(0, "dominance", labels[2])
    frame.insert(0, "arousal", labels[1])
    frame.insert(0, "valence", labels[0])
    frame.insert(0, "time_sec", np.arange(n_rows, dtype=np.float32) / SAMPLING_RATES[signal])
    frame.insert(0, "time_idx", np.arange(n_rows, dtype=np.int32))
    frame.insert(0, "condition", condition)
    frame.insert(0, "trial", trial)
    frame.insert(0, "subject", subject)
    return frame


def convert(mat_path: Path, output_dir: Path, selected: set[int] | None,
            overwrite: bool, resume: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[LOAD] {mat_path} (loaded once)")
    raw = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    data = np.atleast_1d(raw["DREAMER"].Data)
    subject_rows, trial_rows, manifest_rows = [], [], []

    for subject_id, subj in enumerate(data, start=1):
        if selected is not None and subject_id not in selected:
            continue
        subject_rows.append({"subject": subject_id, "age": scalar_text(subj.Age),
                             "gender": scalar_text(subj.Gender)})
        scores = tuple(np.asarray(x, dtype=np.int8).reshape(-1) for x in
                       (subj.ScoreValence, subj.ScoreArousal, subj.ScoreDominance))
        paths = {signal: output_dir / f"subject_{subject_id:02d}_{signal.lower()}.csv"
                 for signal in SAMPLING_RATES}
        complete = resume and all(path.exists() for path in paths.values())
        work_paths = {signal: path.with_suffix(".csv.part") for signal, path in paths.items()}
        for path in paths.values():
            if path.exists() and not (overwrite or complete):
                raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
            if not complete:
                path.unlink(missing_ok=True)
        if not complete:
            for path in work_paths.values():
                path.unlink(missing_ok=True)

        counts = {"EEG": 0, "ECG": 0}
        headers = {"EEG": True, "ECG": True}
        for trial_index in range(N_TRIALS):
            trial_id = trial_index + 1
            labels = tuple(int(score[trial_index]) for score in scores)
            trial_rows.append(dict(subject=subject_id, trial=trial_id, valence=labels[0],
                                   arousal=labels[1], dominance=labels[2]))
            for signal, source in (("EEG", subj.EEG), ("ECG", subj.ECG)):
                for condition in ("baseline", "stimuli"):
                    values = np.atleast_1d(getattr(source, condition))[trial_index]
                    if complete:
                        counts[signal] += len(values)
                        continue
                    frame = trial_frame(values, subject_id, trial_id, condition, signal, labels)
                    frame.to_csv(work_paths[signal], mode="a", index=False, header=headers[signal],
                                 float_format="%.8g")
                    headers[signal] = False
                    counts[signal] += len(frame)
                    del frame

        if not complete:
            for signal in SAMPLING_RATES:
                work_paths[signal].replace(paths[signal])
        for signal, path in paths.items():
            manifest_rows.append(dict(subject=subject_id, signal=signal,
                                      sampling_rate_hz=SAMPLING_RATES[signal], rows=counts[signal],
                                      file=path.name, size_bytes=path.stat().st_size))
        action = "SKIP" if complete else "WRITE"
        print(f"[{action}] subject {subject_id:02d}: EEG={counts['EEG']:,}, ECG={counts['ECG']:,} rows")

    pd.DataFrame(subject_rows).to_csv(output_dir / "subjects.csv", index=False)
    pd.DataFrame(trial_rows).to_csv(output_dir / "trials.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(output_dir / "manifest.csv", index=False)
    total = sum(row["size_bytes"] for row in manifest_rows)
    print(f"[DONE] {len(manifest_rows)} files, {total / 1024**3:.2f} GiB in {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, default=default_mat_path())
    parser.add_argument("--output", type=Path, default=Path("data/dreamer_csv"))
    parser.add_argument("--subjects", type=int, nargs="+", help="optional 1-based IDs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="keep subjects whose EEG and ECG files both exist")
    args = parser.parse_args()
    if not args.mat.is_file():
        parser.error(f"MAT file not found: {args.mat}")
    if args.subjects and any(not 1 <= value <= 23 for value in args.subjects):
        parser.error("subject IDs must be between 1 and 23")
    return args


if __name__ == "__main__":
    args = parse_args()
    convert(args.mat, args.output, set(args.subjects) if args.subjects else None,
            args.overwrite, args.resume)
