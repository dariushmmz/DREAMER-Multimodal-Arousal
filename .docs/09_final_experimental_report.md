# Final Experimental Report

## Multimodal three-class arousal classification on DREAMER

**Delivery status:** Final validated experiment set

**Task:** Subject-independent classification of self-reported arousal

**Modalities:** 14-channel EEG, raw ECG, and ECG-derived HRV

**Primary metric:** Test macro-F1
**Important scope:** These experiments do not constitute clinical stress diagnosis.

## 1. Executive summary

Seven methods were evaluated under one fixed, subject-disjoint protocol. The
adapted 2023 classical stacking method obtained the best test macro-F1 (0.3755)
and balanced accuracy (0.3930). The paper-inspired attention network was the
strongest deep model, with macro-F1 0.3554 and the highest macro-AUROC (0.5117).
It improved macro-F1 by 0.1160 over Base CNN, but it did not outperform adapted
stacking. Performance varied substantially across held-out subjects and all
methods struggled with the low-arousal class.

Historical scores as high as 0.915–0.954 are not comparable with these results:
the historical workflow used overlapping windows, random window-level splits,
shared subjects across partitions, and inconsistent targets. Those outputs are
retained only as evidence of the pre-correction state.

![Unified model comparison](../results/corrected/dreamer_arousal_3class/comparison_all.png)

## 2. Dataset, target, and split

DREAMER contains 23 participants and 18 emotion-eliciting film trials per
participant. The corrected task uses the trial-level self-reported **arousal**
rating; DREAMER does not supply a ground-truth stress variable.

| Class | Original rating | Train windows | Validation windows | Test windows |
|---|---|---:|---:|---:|
| Low | 1–2 | 1,441 | 247 | 253 |
| Medium | 3 | 1,778 | 322 | 316 |
| High | 4–5 | 2,969 | 523 | 523 |
| **Total** | — | **6,188** | **1,092** | **1,092** |

The fixed subject split is:

- Training: 2, 3, 4, 7, 8, 9, 10, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23.
- Validation: 1, 6, 11.
- Test: 5, 13, 14.

The Kaggle DREAMER file is byte-identical to the local raw file: 452,847,973
bytes, SHA-256 `35eed939e6f206a0938b13ba063178e2795ef1739b9cac8e5704a968393f3463`.
The ordered sample-ID digest is
`91cb51e83cb5fa130d6bd9095b93047d9b383f01b213eb0ed334437adee742be`.

## 3. Leakage prevention and preprocessing

Stimulus signals were divided into 10-second windows with a 10-second stride;
there is no temporal overlap. Each sample retains participant, trial, window,
and EEG/ECG start/end indices. Splitting is performed by participant before
model fitting. Consequently, no participant or window appears in more than one
partition.

EEG is band-pass filtered and normalized per window. ECG is filtered and used
both as a raw waveform and to derive 25 HRV descriptors. Deep-model HRV scaling
is fitted on the training partition only. Classical scaling and ANOVA feature
selection are fitted inside the applicable training split/fold. Hyperparameters
and checkpoints are selected using validation data; test labels are not used
for preprocessing, feature selection, model selection, or early stopping.

HRV is mathematically derived from ECG, so ECG–HRV correlation is expected and
is not itself leakage. The largest absolute training-set Spearman correlation
between raw-ECG and HRV feature blocks was 0.439. Three spectral HRV variables
were constant or near-constant at a 10-second duration and are not given a
clinical interpretation.

## 4. Implemented methods

### 4.1 Classical models

The handcrafted representation contains 276 features: 238 EEG descriptors, 13
raw-ECG descriptors, and 25 HRV descriptors.

| Method | Validation-selected setting |
|---|---|
| Logistic regression | C=10; balanced class weights; scaled features |
| RBF SVM | C=1; gamma=`scale`; balanced class weights; scaled features |
| Random forest | 500 trees; minimum leaf size 5; `sqrt` feature sampling |
| kNN | k=5; distance weighting; scaled features |

### 4.2 Adapted 2023 stacking baseline

The adaptation uses ANOVA ranking with 64 selected features; Gaussian Naive
Bayes, logistic regression, AdaBoost, kNN, random forest, linear SVM, and RBF
SVM base learners; subject-grouped five-fold out-of-fold predictions; and a
logistic-regression meta-learner with C=10. It preserves central algorithmic
ideas from the 2023 paper but is not a reproduction of its proprietary
EO/AC1/AC2 stress experiment.

### 4.3 Base CNN

The Base CNN has separate three-block 1D convolutional EEG and ECG branches,
global average pooling, a 25→64 HRV MLP, concatenation, and a 256→128→3 head.
It has 215,843 trainable parameters.

### 4.4 Paper-inspired attention

The attention model uses three multi-scale 1D EEG branches, a raw-ECG Conv1D
pathway, a 25→64→128→256 HRV pathway, 512-dimensional EEG and ECG/HRV
projections, learned two-way modality attention, and a 512→256→128→3 head. It
has 1,846,085 trainable parameters. It adapts multi-scale processing, raw
ECG+HRV fusion, and attention concepts from the 2026 paper. It does not
implement that paper's CWT images or VGG16/EfficientNetB0/ResNeXt50 backbones.

## 5. Experimental settings

Both deep models used PyTorch, AdamW, learning rate 3×10⁻⁴, weight decay 10⁻⁴,
cosine scheduling, batch size 32, class-weighted cross-entropy with label
smoothing 0.1, mixed precision, train-only amplitude/noise augmentation, seed
20261011, a 100-epoch maximum, and early-stopping patience 12. Validation
macro-F1 selected the checkpoint.

| Model | Selected epoch | Epochs executed | Best validation macro-F1 | Tesla T4 runtime |
|---|---:|---:|---:|---:|
| Base CNN | 25 | 37 | 0.3411 | 113.39 s |
| Paper-inspired attention | 12 | 24 | 0.3070 | 136.58 s |

Kaggle reported 0.10 of the available 30 GPU-hours used after both jobs. No
direct monetary charge was incurred. The account-specific queue/allocation time
is distinct from the in-process runtimes above.

## 6. Final window-level results

| Model | Family | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUROC |
|---|---|---:|---:|---:|---:|
| Adapted 2023 stacking | Classical ensemble | **0.4698** | **0.3930** | **0.3755** | 0.5097 |
| Paper-inspired attention | Deep learning | 0.4029 | 0.3732 | 0.3554 | **0.5117** |
| kNN | Classical ML | 0.3901 | 0.3160 | 0.3020 | 0.4578 |
| Random forest | Classical ML | 0.4560 | 0.3449 | 0.2933 | 0.4367 |
| Logistic regression | Classical ML | 0.3361 | 0.2884 | 0.2791 | 0.4236 |
| RBF SVM | Classical ML | 0.4524 | 0.3275 | 0.2533 | 0.3945 |
| Base CNN | Deep learning | 0.3297 | 0.3584 | 0.2394 | 0.4843 |

The adapted stack leads the primary macro-F1 measure. Attention improves by
0.0534 over the best individual classical model and by 0.1160 over Base CNN.
Its 0.0019 AUROC advantage over stacking is too small to support a strong
superiority claim with only three held-out participants.

## 7. Per-class evaluation

![Per-class F1](../results/corrected/dreamer_arousal_3class/per_class_f1_comparison.png)

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| kNN | Low | 0.5897 | 0.0909 | 0.1575 | 253 |
| kNN | Medium | 0.2177 | 0.2184 | 0.2180 | 316 |
| kNN | High | 0.4538 | 0.6386 | 0.5306 | 523 |
| Random forest | Low | 0.2500 | 0.0119 | 0.0226 | 253 |
| Random forest | Medium | 0.3675 | 0.1930 | 0.2531 | 316 |
| Random forest | High | 0.4748 | 0.8298 | 0.6040 | 523 |
| Logistic regression | Low | 0.3571 | 0.0791 | 0.1294 | 253 |
| Logistic regression | Medium | 0.2037 | 0.3101 | 0.2459 | 316 |
| Logistic regression | High | 0.4486 | 0.4761 | 0.4620 | 523 |
| RBF SVM | Low | 0.4000 | 0.0079 | 0.0155 | 253 |
| RBF SVM | Medium | 0.3214 | 0.0854 | 0.1350 | 316 |
| RBF SVM | High | 0.4636 | 0.8891 | 0.6094 | 523 |
| Adapted stacking | Low | 0.1481 | 0.0632 | 0.0886 | 253 |
| Adapted stacking | Medium | 0.4714 | 0.4177 | 0.4430 | 316 |
| Adapted stacking | High | 0.5185 | 0.6979 | 0.5949 | 523 |
| Base CNN | Low | 0.5789 | 0.0435 | 0.0809 | 253 |
| Base CNN | Medium | 0.2969 | 0.9209 | 0.4491 | 316 |
| Base CNN | High | 0.6237 | 0.1109 | 0.1883 | 523 |
| Paper-inspired attention | Low | 0.3107 | 0.1265 | 0.1798 | 253 |
| Paper-inspired attention | Medium | 0.3276 | 0.5380 | 0.4072 | 316 |
| Paper-inspired attention | High | 0.5064 | 0.4551 | 0.4794 | 523 |

The high class comprises 47.9% of the test windows. Base CNN predicts medium
for 980 of 1,092 windows, producing medium recall 0.921 but low/high recall only
0.043/0.111. Attention is more balanced, yet low-class recall remains 0.126.

## 8. Trial-level and participant-level evaluation

Trial probabilities are the mean of all windows belonging to each participant
and trial. Test data contain 54 trials.

| Model | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUROC |
|---|---:|---:|---:|---:|
| kNN | 0.3519 | 0.3083 | 0.2698 | 0.4356 |
| Random forest | 0.4259 | 0.3619 | 0.2774 | 0.4651 |
| Logistic regression | 0.3333 | 0.3065 | 0.2843 | 0.4534 |
| RBF SVM | 0.4074 | 0.3422 | 0.2487 | 0.4075 |
| Adapted stacking | **0.4630** | **0.4171** | **0.3843** | 0.5067 |
| Base CNN | 0.3148 | 0.3315 | 0.2174 | 0.5261 |
| Paper-inspired attention | 0.3519 | 0.3403 | 0.3401 | **0.5281** |

![Held-out participant variability](../results/corrected/dreamer_arousal_3class/subject_f1_comparison.png)

Attention macro-F1 is 0.3132, 0.3593, and 0.2263 for participants 5, 13,
and 14. Adapted stacking is 0.4032, 0.3960, and 0.2065. These ranges show that
aggregate results depend strongly on which unseen participant is evaluated.

## 9. Training behavior and confusion matrices

| Base CNN training | Paper-inspired attention training |
|---|---|
| ![Base training](../results/corrected/dreamer_arousal_3class/base_cnn/training_curves.png) | ![Attention training](../results/corrected/dreamer_arousal_3class/paper_inspired_attention/training_curves.png) |

Final train/validation macro-F1 is 0.8591/0.1394 for Base and 0.9551/0.2948
for Attention. This is clear overfitting rather than evidence that additional
epochs would improve unseen-participant performance. No test-driven retraining
or hyperparameter adjustment was performed.

| Base CNN confusion matrix | Paper-inspired attention confusion matrix |
|---|---|
| ![Base confusion matrix](../results/corrected/dreamer_arousal_3class/base_cnn/confusion_matrix.png) | ![Attention confusion matrix](../results/corrected/dreamer_arousal_3class/paper_inspired_attention/confusion_matrix.png) |

The attention model assigns a mean learned fusion weight of 0.546 to EEG and
0.454 to ECG/HRV. These are model-internal weights, not causal measures of
physiological importance.

## 10. Relationship to the 2026 paper

The 2026 paper reports CWT image representations, modified VGG16,
EfficientNetB0, and ResNeXt50 backbones, hierarchical attention, approximately
42.7 million parameters, WESAD+CASE data, rest/stress/amusement labels, and
95.7% accuracy. The implemented 1.846-million-parameter model uses raw temporal
DREAMER signals and self-reported arousal labels.

A faithful reproduction cannot be claimed because the available WESAD and CASE
documentation does not support the paper's stated EEG sensor descriptions or
subject/task composition, and its executable code/data mapping is unavailable.
The reported 95.7% is therefore not directly comparable with the present 0.4029
attention accuracy. The valid conclusion is narrower: adapted attention improves
over this project's Base CNN but not over the leakage-safe adapted stack.

## 11. Validation and reproducibility evidence

All seven prediction files were replayed against the fixed 1,092-sample test
index. Sample order, labels, probability argmax, confusion matrices, scalar
metrics, and unified comparison rows match. Raw Kaggle outputs, postprocessed
metrics, environment details, source digests, checkpoints, training logs, and
plots are retained. Automated tests cover label mapping, splitting, window
overlap, preprocessing, architectures, and probability evaluation.

Primary evidence files are the unified comparison CSV, per-class CSV,
subject-level CSV, per-model `metrics.json` and `predictions.csv`, and the
machine-readable artifact manifest. Historical outputs remain isolated under
`results/legacy/` and are excluded from the comparison builder.

## 12. Limitations and conclusions

- Arousal and stress are related but not interchangeable constructs.
- Only three participants form the fixed test set; no confidence interval or
  broad population claim is justified.
- Class imbalance and participant shift materially affect recall and macro-F1.
- Ten-second windows limit reliable interpretation of low-frequency HRV.
- The attention model is a documented adaptation, not a faithful 2026 reproduction.
- Small metric differences, particularly AUROC 0.5117 versus 0.5097, should not
  be interpreted as decisive superiority.

Within these constraints, the corrected results show that attention improves
the implemented deep baseline, while the adapted classical stack provides the
best overall macro-F1 and balanced accuracy for this subject-independent
DREAMER arousal task.
