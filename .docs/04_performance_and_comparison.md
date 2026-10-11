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

## Three-class paper-method adaptation

The leakage-corrected 2023-paper stacking adaptation achieved accuracy 0.470,
balanced accuracy 0.393, macro-F1 0.376, and macro one-vs-rest AUROC 0.510. Its
per-class F1 values were 0.089 (low), 0.443 (medium), and 0.595 (high). This is
the strongest corrected classical macro-F1, but it remains an arousal-task
adaptation rather than reproduction of the paper's stress experiment.

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

The classical side is complete under the common protocol. Deep-learning code,
cache, architecture documentation, and Modal job are complete, but the actual
GPU runs are pending explicit billing approval. Until those runs finish, an
ML-versus-deep performance conclusion is **not available** and no legacy deep
score is substituted.

