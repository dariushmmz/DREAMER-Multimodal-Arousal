# Model performance and comparison

## Historical results (not corrected measurements)

The legacy window-random workflow recorded base CNN accuracy 0.864, attention
CNN accuracy 0.915, and random-forest arousal accuracy 0.954. These results used
overlapping windows, shared subjects/trials across partitions, and inconsistent
targets. They remain under `results/legacy/window_level_random_split/` only for
traceability.

## Corrected classical results

Untouched test subjects: 5, 13, and 14 (1,092 windows). Hyperparameters were
selected on subjects 1, 6, and 11.

| Model | Validation macro-F1 | Test accuracy | Balanced accuracy | Test macro-F1 | Macro AUROC |
|---|---:|---:|---:|---:|---:|
| kNN | 0.359 | 0.390 | 0.316 | 0.302 | 0.458 |
| Random forest | 0.311 | 0.456 | 0.345 | 0.293 | 0.437 |
| Logistic regression | 0.318 | 0.336 | 0.288 | 0.279 | 0.424 |
| RBF SVM | 0.289 | 0.452 | 0.327 | 0.253 | 0.395 |

Accuracy alone is misleading because 47.9% of test windows are high-arousal.
For example, random forest reaches 0.456 accuracy but recalls only 0.012 of the
low class and 0.830 of the high class. Macro-F1 and balanced accuracy are the
primary summary measures.

## Three-class stacking comparator

The leakage-corrected stacking ensemble achieved accuracy 0.470,
balanced accuracy 0.393, macro-F1 0.376, and macro one-vs-rest AUROC 0.510. Its
per-class F1 values were 0.089 (low), 0.443 (medium), and 0.595 (high). This is
the strongest corrected classical macro-F1. It is retained as an independent
comparator and is not a reproduction of the 2026 attention paper.

## Modality ablation (random forest)

| Feature block | Features | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| HRV only | 25 | 0.341 | 0.313 | 0.310 |
| EEG + ECG + HRV | 276 | 0.456 | 0.345 | 0.293 |
| EEG + raw ECG | 251 | 0.430 | 0.327 | 0.275 |
| EEG only | 238 | 0.405 | 0.300 | 0.249 |
| ECG + HRV | 38 | 0.217 | 0.212 | 0.213 |
| Raw ECG only | 13 | 0.214 | 0.228 | 0.209 |

The ablation does not support a claim that multimodal fusion improves
subject-independent generalization in this feature/model configuration.

## ML versus deep learning

All methods below use the same byte-verified DREAMER source, three-class arousal
mapping, window IDs, 1,092-window untouched test set, and subject split. The two
deep models share the same Kaggle cache; classical features were produced by the
corresponding leakage-safe local pipeline. Model/checkpoint selection used
validation macro-F1 only.

| Model | Family | Validation macro-F1 | Accuracy | Balanced accuracy | Macro-F1 | Macro AUROC |
|---|---|---:|---:|---:|---:|---:|
| Adapted stacking | Classical ensemble | 0.342 | **0.470** | **0.393** | **0.376** | 0.510 |
| Paper-inspired attention | Deep learning | 0.307 | 0.403 | 0.373 | 0.355 | **0.512** |
| kNN | Classical | **0.359** | 0.390 | 0.316 | 0.302 | 0.458 |
| Random forest | Classical | 0.311 | 0.456 | 0.345 | 0.293 | 0.437 |
| Logistic regression | Classical | 0.318 | 0.336 | 0.288 | 0.279 | 0.424 |
| RBF SVM | Classical | 0.289 | 0.452 | 0.327 | 0.253 | 0.395 |
| Base CNN | Deep learning | 0.341 | 0.330 | 0.358 | 0.239 | 0.484 |

The adapted stacking model has the highest primary test macro-F1 (0.3755). The
attention model is 0.0201 lower, but improves by 0.0534 over the best individual
classical model (kNN) and by 0.1160 over Base CNN. Attention has the highest
macro AUROC, but the 0.0019 difference from stacking is too small to support a
strong superiority claim, especially with only three held-out subjects. Base
CNN does not outperform the classical models on macro-F1.

## Deep-model class behavior

| Model/class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Base / low | 0.579 | 0.043 | 0.081 | 253 |
| Base / medium | 0.297 | 0.921 | 0.449 | 316 |
| Base / high | 0.624 | 0.111 | 0.188 | 523 |
| Attention / low | 0.311 | 0.126 | 0.180 | 253 |
| Attention / medium | 0.328 | 0.538 | 0.407 | 316 |
| Attention / high | 0.506 | 0.455 | 0.479 | 523 |

Base predicts medium for 980/1,092 test windows, explaining the disagreement
between its 0.330 accuracy and 0.358 balanced accuracy. Attention distributes
predictions more evenly and materially improves low/high recall, but low-arousal
recall remains only 0.126. Its held-out-subject macro-F1 is 0.313, 0.359, and
0.226 for subjects 5, 13, and 14 respectively; this spread shows that
cross-subject generalization remains unstable. The adapted stack shows the same
problem (0.403, 0.396, and 0.207).

At trial level, Base and Attention macro-F1 are 0.217 and 0.340 respectively.
The attention model's mean learned fusion weights are 0.546 for EEG and 0.454
for ECG/HRV. These weights describe model behavior and are not interpreted as
causal physiological importance.

## Training evidence and figures

Base selected epoch 25 and stopped after 37 epochs; Attention selected epoch 12
and stopped after 24. Final train/validation macro-F1 gaps (0.859/0.139 and
0.955/0.295) show overfitting despite augmentation, class weights, dropout, and
early stopping. No test-driven retraining was performed.

The unified source table is `results/corrected/dreamer_arousal_3class/comparison_all.csv`.
Overall, class-level, and subject-level figures are `comparison_all.png`,
`per_class_f1_comparison.png`, and `subject_f1_comparison.png`. Each deep result
directory also contains its training curve and confusion matrices. Exact replay
of all seven models from saved predictions is recorded in
`verification_summary.json`.
