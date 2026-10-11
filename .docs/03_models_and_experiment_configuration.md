# Models and experiment configuration

All methods use `configs/dreamer_arousal_3class.yaml`, the same 8,372 samples,
three-class arousal labels, and fixed subject partition.

## Classical models

Unscaled feature extraction produces 276 descriptors: 238 EEG statistical and
band-power features, 13 raw-ECG statistical/spectral features, and 25
ECG-derived HRV features. Scaling and ANOVA selection are fitted only on the
training portion relevant to each model/fold.

| Model | Candidate settings | Selected setting |
|---|---|---|
| Logistic regression | C = 0.1, 1, 10; balanced classes | C = 10 |
| RBF SVM | C = 1, 10; gamma = scale; balanced classes | C = 1 |
| Random forest | 500 trees; min leaf = 1, 2, 5; sqrt features | min leaf = 5 |
| kNN | k = 5, 11, 21; distance weights | k = 5 |

Selection metric is validation macro-F1. No model is refit with test data.

### 2023-paper adapted stacking

The adaptation uses the paper's NB, LR, AdaBoost, kNN, RF, linear SVM, and RBF
SVM base learners, ANOVA F-value feature ranking (64 selected features), and a
logistic-regression meta learner. Five-fold out-of-fold base predictions are
created with `GroupKFold`, so held-out subjects never occur in a fold's fit
set. The meta-model C is selected from 0.1, 1, and 10 on the validation subjects;
C=10 was selected. This is not a faithful stress reproduction because DREAMER
arousal is not EO/AC1/AC2 stress and the acquisition hardware differs.

## Base multimodal CNN

| Component | Layers |
|---|---|
| EEG branch | Conv1D 14→32 (k7), 32→64 (k5), 64→128 (k3); BN/ReLU/max-pool per block; global average pool |
| ECG branch | Conv1D 1→32 (k15), 32→64 (k9), 64→128 (k5); BN/ReLU/max-pool; global average pool |
| HRV branch | Dense 25→64, ReLU, dropout 0.2 |
| Fusion/head | Concatenate 320; Dense 256→128→3; dropout 0.4/0.3 |

Trainable parameters: **215,843**.

## Paper-inspired attention model

| Component | Layers |
|---|---|
| EEG | Three parallel Conv1D branches, each 14→32→64→128; kernel sets (15,7,3), (31,15,7), (63,31,15); concatenation 384→512 |
| Raw ECG | Conv1D 1→64→128→256 with kernels 15,7,3 |
| HRV | Dense 25→64→128→256, dropout 0.3 |
| ECG projection | Concatenate 512→512 |
| Attention | Separate EEG/ECG 512-dimensional value maps and 512→128→1 modality scores; two-way softmax weighted sum |
| Head | Dense 512→256→128→3; batch normalization; ReLU; dropout 0.5 |

Trainable parameters: **1,846,085**.

This implementation transfers the paper's multi-scale, raw-ECG+HRV, and
attention-fusion concepts to DREAMER. It does **not** implement the paper's
VGG16/EfficientNetB0/ResNeXt50 CWT image system and is always named
`paper_inspired_attention`.

## Deep training settings

- PyTorch; AdamW; learning rate 3e-4; weight decay 1e-4.
- Cosine annealing to 1e-6; maximum 100 epochs.
- Batch size 32; label smoothing 0.1; inverse-frequency class weights.
- Validation macro-F1 checkpoint selection; patience 12.
- Mixed precision on CUDA; deterministic seed 20261011.
- Train-only augmentation: amplitude scaling 0.9–1.1 and Gaussian noise 0.02.
- Test set is evaluated once after restoring the best validation checkpoint.

GPU training is pending the explicit Modal cost approval recorded in
`.docs/06_modal_environment.md`; no deep score is reported before that run.

