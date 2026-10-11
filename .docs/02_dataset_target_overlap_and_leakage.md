# Dataset, target, overlap removal, and leakage assessment

## Scientific target

DREAMER contains trial-level self-ratings of **valence**, **arousal**, and
**dominance** on a 1–5 scale. It contains no experimentally assigned or
clinically assessed stress label. The corrected common target is therefore
**three-class self-reported arousal**, mapped as follows:

| Class | DREAMER rating | Interpretation |
|---|---:|---|
| 0 | 1–2 | low arousal |
| 1 | 3 | medium arousal |
| 2 | 4–5 | high arousal |

Arousal was selected because it is the affect dimension most closely related to
activation of the autonomic nervous system, but it is not synonymous with
stress. High arousal may accompany stress, excitement, fear, or amusement;
stress also depends on context, valence/appraisal, and the stressor protocol.
Accordingly, corrected files use “arousal classification,” not “stress
detection,” for DREAMER results. The [DREAMER publication](https://doi.org/10.1109/JBHI.2017.2688239)
likewise describes valence/arousal/dominance affect recognition.

## Window construction and split

- Stimulus recordings only; baseline recordings are not mislabeled as the
  participant's trial self-rating.
- Synchronized 10-second EEG (1,280 samples) and ECG (2,560 samples) windows.
- Hop equals window length; overlap ratio is exactly 0.0.
- Each row has a stable `Sxx_Txx_Wxxx` sample identifier and raw start/end
  indices in `data/processed/window_index.csv`.
- The subject split is made before scaling, feature selection, model selection,
  or evaluation.

| Partition | Subjects | Windows | Low | Medium | High |
|---|---|---:|---:|---:|---:|
| Train | 2,3,4,7,8,9,10,12,15,16,17,18,19,20,21,22,23 | 6,188 | 1,441 | 1,778 | 2,969 |
| Validation | 1,6,11 | 1,092 | 247 | 322 | 523 |
| Test | 5,13,14 | 1,092 | 253 | 316 | 523 |

The index contains 8,372 windows from 414 trials and 23 subjects. Automated
checks reject duplicate IDs, overlapping time intervals, subject sharing, split
coverage gaps, and labels outside the declared mapping.

## Leakage audit

| Stage | Legacy issue | Corrected control |
|---|---|---|
| Windowing | 50% overlap created near-duplicate samples | Non-overlapping windows and an interval assertion |
| Splitting | Random windows put each subject/trial in multiple partitions | Immutable subject IDs with disjoint train/validation/test sets |
| Normalization | Full trial was normalized before random window splitting | Deep signals are normalized per window; HRV scaler is fit on train only |
| Classical scaling | Random-split pipeline | `StandardScaler` is inside each training pipeline |
| Feature selection | No consistent protocol | ANOVA selection is fit inside each group-disjoint stacking fold |
| Model selection | Different targets and splits | Validation subjects only; one untouched common test set |
| Evaluation | Weighted metrics and incompatible tasks | Macro/per-class metrics, confusion matrices, probabilities, and trial aggregation |

Filtering is applied to a complete trial for efficiency, but every complete
trial belongs to one subject partition, so filtering cannot cross a
train/validation/test boundary. Window normalization never uses another sample.

## ECG and HRV dependency

HRV is calculated from R–R intervals detected in ECG. Thus ECG and HRV are
mathematically and physiologically dependent; this dependence is not leakage
unless information from held-out subjects or labels is used during processing.
The corrected pipeline computes HRV within each window and fits its scaler only
on training subjects.

On training subjects, raw-ECG versus HRV features had maximum absolute Spearman
correlation 0.439 and median absolute correlation 0.128; zero pairs exceeded
0.8. Random-forest test macro-F1 was 0.310 for HRV alone, 0.209 for raw ECG
alone, 0.213 for ECG+HRV, and 0.293 for all modalities. This indicates that the
derived block can add redundancy/noise; it does not demonstrate label leakage.

Standard VLF/LF HRV power cannot be reliably estimated from 10-second windows.
The generated compatibility vector therefore contains three constant/near-zero
frequency features that are ignored by training-only feature selection and are
not interpreted clinically. Time-domain ultra-short HRV results are also
treated as experimental descriptors rather than clinical HRV measurements.

