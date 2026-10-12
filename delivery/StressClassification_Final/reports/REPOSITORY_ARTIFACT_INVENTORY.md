# Artifact inventory and Git traceability

## Git milestones

| Commit | Purpose |
|---|---|
| `2c7b2fd` | Initial preservation baseline; original code/results untouched |
| `afc21a8` | Repository restructuring and explicit legacy labeling |
| `43ba248` | Non-overlapping, subject-independent pipeline, tests, config, cache, and bounded remote job |
| `8aa7772` | Corrected classical results, modality audit, adapted 2023 baseline, and initial reviewer reports |
| `0509a1e` | Reproducible Kaggle CPU/GPU notebook workflow |
| `60b84e8` | Float64 probability normalization and AUROC regression test |
| `7055739` | Validated Kaggle Base CNN artifacts and preprocessing evidence |
| `e035199` | Validated Attention artifacts, exact replay check, and final comparison figures |
| `fa2814a` | Delivery-oriented source, notebook, and validation-tool organization |

The machine-readable inventory `.docs/artifact_manifest.json` records byte size,
SHA-256, and Git tracking state for every curated artifact and local large file.

## Dataset and protocol artifacts

- Configuration: `configs/dreamer_arousal_3class.yaml`
- Split: `data/processed/subject_split.json`
- Dataset summary: `data/processed/dataset_summary.json`
- Window provenance: `data/processed/window_index.csv`
- Local deep-cache shapes, scaler, and hashes:
  `data/processed/deep_cache/manifest.json`
- Kaggle raw-data/cache verification:
  `results/corrected/dreamer_arousal_3class/kaggle_preprocessing/`

## Implementation and execution artifacts

- Reusable source: `src/stress_classification/`
- Corrected notebooks: `notebooks/workflows/01_dataset_protocol_audit.ipynb`,
  `notebooks/workflows/02_classical_ml_evaluation.ipynb`, and
  `notebooks/workflows/03_deep_learning_evaluation.ipynb`
- Preserved historical notebooks: `notebooks/legacy/`
- Kaggle notebook generator, entrypoint, and private-kernel metadata: `kaggle/`
- Artifact ingestion and consistency check: `scripts/results/ingest_kaggle_outputs.py`
- Tests: `tests/`
- Preserved, unused-after-migration Modal implementation: `modal/`

## Experiment outputs

- Classical models: `results/corrected/dreamer_arousal_3class/classical/`
- Modality ablation: `results/corrected/dreamer_arousal_3class/modality_ablation/`
- Adapted 2023 stack:
  `results/corrected/dreamer_arousal_3class/paper_2023_adapted_stacking/`
- Base CNN: `results/corrected/dreamer_arousal_3class/base_cnn/`
- Paper-inspired attention:
  `results/corrected/dreamer_arousal_3class/paper_inspired_attention/`
- Unified metrics and figures: `comparison_all.csv`, `comparison_all.png`,
  `per_class_metrics.csv`, `per_class_f1_comparison.png`,
  `subject_metrics.csv`, and `subject_f1_comparison.png` under the corrected
  experiment root.

Each deep result contains the resolved config, architecture, environment,
infrastructure and runtime metadata, epoch history/log, best and last
checkpoints, predictions, window/trial metrics, confusion matrices, and curves.
The original Kaggle metrics/predictions are retained beside normalized versions,
with both digests in `artifact_validation.json`.

Raw DREAMER data, signal arrays, fitted model binaries, and PyTorch checkpoints
remain local and are excluded from ordinary Git history. They are not absent:
their paths, sizes, and hashes are captured by the manifest, while the small
reproduction metadata, predictions, metrics, and figures are Git-versioned.
Historical leakage-affected outputs remain isolated in `results/legacy/` and
are never used by the comparison builder.

## Final reports and client delivery

- Final experimental report: `.docs/09_final_experimental_report.md`
- Final Persian reviewer responses: `.docs/07_responses_to_reviewers_fa.md`
  and `.docs/07_responses_to_reviewers_fa.docx`
- Reference-paper comparison:
  `.docs/05_reference_paper_reproduction_and_differences.md`
- Client package directory: `delivery/StressClassification_Final/`
- Portable package archive: `delivery/StressClassification_Final.zip`
- Package checksum record: `delivery/delivery_checksums.json`

The package contains the reproducible source, curated corrected results,
reports, figures, predictions, logs, and fitted-model binaries. The raw DREAMER
file and derived deep tensor cache remain outside the package; the package's
`manifests/CHECKPOINTS_AND_DATA.md` records their exact sizes, SHA-256 values,
expected locations, and reconstruction procedure. This makes the omission
explicit rather than silently dropping required inputs.
