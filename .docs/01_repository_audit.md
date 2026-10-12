# Repository audit and restructuring

## Baseline and preservation

- Audit date: 2026-10-11 (Asia/Tehran).
- The supplied directory was not a Git repository and no parent Git repository
  was found.
- Git was initialized without deleting or overwriting any existing artifact.
- Commit `2c7b2fd` preserves the original notebooks, source helpers, reports,
  reference PDFs, and result files.
- Approximately 23.5 GB under `.data/`, 1.4 GB under `data/`, local archives,
  caches, and IDE state were excluded from Git but left untouched.

## Original implementation inventory

| Artifact | Role | Audit finding |
|---|---|---|
| `notebooks/legacy/legacy_base_multimodal_cnn.ipynb` | Three-branch temporal CNN | Three-class valence; 10 s windows with 50% overlap; random 70/15/15 window split |
| `notebooks/legacy/legacy_attention_multimodal_cnn.ipynb` | Multi-scale CNN with attention-like fusion | Three-class valence; same overlap and split leakage |
| `notebooks/legacy/legacy_classical_ml.ipynb` | SVM, random forest, kNN | Three-class arousal; random 80/20 window split; not comparable to deep models |
| `src/legacy/eval_metrics.py` | Deep-model plots and metrics | Weighted metrics only; no provenance or subject-level audit |
| `src/legacy/metrics.py` | Classical metrics | Macro metrics, but no group-aware evaluation |
| `results/legacy/window_level_random_split/` | Historical outputs | Preserved; scientifically invalid for subject-independent claims |

The legacy pipeline generated 16,606 windows. Its saved test results were
86.43% accuracy for the base CNN and 91.53% for the attention model on valence.
The ML notebook recorded 91.84% SVM, 95.36% random forest, and 88.50% kNN
accuracy on arousal. These numbers use different targets/splits and substantial
within-trial and within-subject leakage, so they are not valid comparisons.

## Reference-paper provenance

The client-requested reference is the 2026 *Scientific Reports* article
*An attention-based multimodal deep learning framework integrating EEG and ECG
for enhanced stress detection*, stored locally as
`.paper/An attention-based multimodal.pdf`. It reports VGG16,
EfficientNetB0, and ResNeXt50 feature extractors, attention fusion, nonoverlap
10-second epochs, subject-independent 80/10/10 partitioning, and 95.7%
three-class accuracy. However, it states that WESAD and CASE contain EEG. Their
primary dataset documentation does not describe the EEG configurations
attributed to them by the paper. The exact reported EEG–ECG experiment therefore
cannot be reconstructed from the documented public releases alone. This
evidence boundary remains explicit in all comparison reports.

## Restructuring

- Primary notebooks moved to `notebooks/` without content changes.
- Existing reports moved to `.docs/legacy/`.
- Existing results moved to
  `results/legacy/window_level_random_split/` and labeled as historical.
- The optional CSV conversion utility is `scripts/data/export_dreamer_csv.py`.
- Reusable code, configurations, tests, Kaggle execution, and corrected results
  are separated into dedicated directories.
