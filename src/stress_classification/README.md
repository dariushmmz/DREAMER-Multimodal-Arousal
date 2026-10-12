# Core package responsibilities

- `config.py`: load and validate versioned experiment configuration.
- `labels.py`: define the three-class self-reported arousal target.
- `splits.py`: validate the fixed subject-disjoint partition.
- `data.py`: read DREAMER and construct the auditable window index.
- `preprocessing.py`: signal filters, normalization, and HRV extraction.
- `features.py`: handcrafted EEG, ECG, and HRV feature matrices.
- `deep_data.py`: cached multimodal arrays and PyTorch datasets.
- `models.py`: Base CNN and paper-inspired attention architectures.
- `training.py`: seeded deep training, early stopping, checkpoints, and logs.
- `evaluation.py`: probability normalization, metrics, aggregation, and
  confusion-matrix export.

Classical estimators and experiment orchestration remain in the single-purpose
entry points under `scripts/experiments/`; reporting and delivery tools are
under `scripts/results/` and `scripts/reporting/`.
