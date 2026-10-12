# Command-line workflows

Each entry point has one responsibility and resolves paths from the repository
root, so commands may be launched from any working directory.

## Data

- `data/audit_dataset.py`: build and validate the non-overlapping sample index.
- `data/build_deep_cache.py`: materialize EEG, ECG, HRV, and label arrays.
- `data/export_dreamer_csv.py`: optional memory-safe raw MATLAB-to-CSV export.

## Experiments

- `experiments/train_classical_models.py`: train/select the four classical models.
- `experiments/train_adapted_stacking.py`: run the adapted 2023 stacking baseline.
- `experiments/evaluate_modality_ablation.py`: evaluate handcrafted feature blocks.
- `experiments/train_deep_model.py`: train one configured deep model.

## Results

- `results/ingest_kaggle_outputs.py`: validate and normalize downloaded GPU artifacts.
- `results/build_model_comparison.py`: build unified tables and publication figures.
- `results/verify_experiment_artifacts.py`: replay metrics from saved predictions.

## Reporting

- `reporting/build_artifact_manifest.py`: hash curated and local artifacts.
- `reporting/generate_reviewer_docx.py`: export the Persian reviewer responses.
- `reporting/build_delivery_package.py`: assemble the client delivery snapshot.
