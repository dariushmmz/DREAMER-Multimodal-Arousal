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
| `notebooks/base_multimodal.ipynb` | Three-branch temporal CNN | Three-class valence; 10 s windows with 50% overlap; random 70/15/15 window split |
| `notebooks/attention_multimodal.ipynb` | Multi-scale CNN with attention-like fusion | Three-class valence; same overlap and split leakage |
| `notebooks/ml_models.ipynb` | SVM, random forest, kNN | Three-class arousal; random 80/20 window split; not comparable to deep models |
| `src/eval_metrics.py` | Deep-model plots and metrics | Weighted metrics only; no provenance or subject-level audit |
| `src/metrics.py` | Classical metrics | Macro metrics, but no group-aware evaluation |
| `results/legacy/window_level_random_split/` | Historical outputs | Preserved; scientifically invalid for subject-independent claims |

The legacy pipeline generated 16,606 windows. Its saved test results were
86.43% accuracy for the base CNN and 91.53% for the attention model on valence.
The ML notebook recorded 91.84% SVM, 95.36% random forest, and 88.50% kNN
accuracy on arousal. These numbers use different targets/splits and substantial
within-trial and within-subject leakage, so they are not valid comparisons.

## Reference-file provenance

The path specified as the “2026 reference paper,”
`.paper/ECG and EEG based detection and multilevel.pdf`, is actually the 2023
PLOS ONE article *ECG and EEG based detection and multilevel classification of
stress using machine learning for specified genders: A preliminary study* by
Hemakom, Atiwiwat, and Israsena. It uses a private 40-subject mental-arithmetic
dataset and classical feature selection/classification; its mixed-gender
three-level stacking accuracy is 64.08%.

A separate local file, `.paper/An attention-based multimodal.pdf`, is the 2026
Scientific Reports article named in the request. It reports VGG16,
EfficientNetB0, and ResNeXt50 feature extractors, attention fusion, nonoverlap
10-second epochs, subject-independent 80/10/10 partitioning, and 95.7%
three-class accuracy. However, it states that WESAD and CASE contain EEG. Their
primary dataset documentation shows that neither dataset contains EEG. The
paper's exact EEG–ECG experiment therefore cannot be reproduced from those
datasets as described. This limitation must remain explicit in all comparison
reports.

## Restructuring

- Primary notebooks moved to `notebooks/` without content changes.
- Existing reports moved to `.docs/legacy/`.
- Existing results moved to
  `results/legacy/window_level_random_split/` and labeled as historical.
- The CSV conversion utility moved to `scripts/convert_dreamer_csv.py`.
- Reusable code, configurations, tests, Kaggle execution, and corrected results
  are separated into dedicated directories.
