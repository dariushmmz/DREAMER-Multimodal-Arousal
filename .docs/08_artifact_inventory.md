# Artifact inventory and Git traceability

## Git milestones

| Commit | Purpose |
|---|---|
| `2c7b2fd` | Initial preservation baseline; original code/results untouched |
| `afc21a8` | Repository restructuring and explicit legacy labeling |
| `43ba248` | Non-overlapping, subject-independent pipeline, tests, configs, cache manifest, and Modal job |
| `8aa7772` | Corrected classical results, modality audit, adapted 2023 baseline, and reviewer reports |

The machine-readable checksum inventory is `.docs/artifact_manifest.json`.

## Core reproducibility artifacts

- Configuration: `configs/dreamer_arousal_3class.yaml`
- Split: `data/processed/subject_split.json`
- Dataset summary: `data/processed/dataset_summary.json`
- Window provenance: `data/processed/window_index.csv`
- Deep-array shapes, training-only scaler, and SHA-256 hashes:
  `data/processed/deep_cache/manifest.json`
- Source package: `src/stress_classification/`
- Automated tests: `tests/`
- Corrected notebooks: `notebooks/01_*` through `notebooks/03_*`
- Classical outputs: `results/corrected/dreamer_arousal_3class/classical/`
- Modality audit: `results/corrected/dreamer_arousal_3class/modality_ablation/`
- Adapted 2023 baseline:
  `results/corrected/dreamer_arousal_3class/paper_2023_adapted_stacking/`
- Modal definition and cost gate: `modal/`

Large raw data, signal caches, fitted classical binaries, and future PyTorch
checkpoints remain local and are excluded from ordinary Git history. Their
paths, byte sizes, and hashes are included in the JSON inventory so the exact
local artifacts remain verifiable without inflating repository history.

## Pending artifacts

The two deep-training result directories, their checkpoints/logs/curves, actual
GPU/runtime consumption, and the final ML-versus-deep figure are pending Modal
cost approval. They are not represented as completed deliverables.
