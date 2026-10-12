# Kaggle GPU environment and migration from Modal

## Why execution moved

Modal authentication and workspace access were valid, but an L4 allocation was
rejected before startup because the workspace required a payment method even
though promotional credit was visible. No Modal GPU was allocated and no Modal
GPU charge was incurred. The existing Modal code and audit history are retained
for traceability, but no further Modal jobs were launched.

The remaining experiments were migrated to private Kaggle notebooks generated
from the canonical repository source by `kaggle/build_kernels.py`. Authentication
uses the local Kaggle profile for account `amirmz`; no OAuth credential or token
is stored in the repository. Network calls used the user-provided local SOCKS
proxy. Billing settings were not changed.

## Dataset and preprocessing verification

The Kaggle source is `phhasian0710/dreamer`. Both copies of `DREAMER.mat` in
that dataset are 452,847,973 bytes and have SHA-256
`35eed939e6f206a0938b13ba063178e2795ef1739b9cac8e5704a968393f3463`,
exactly matching the local raw file. The CPU preprocessing notebook produced
8,372 non-overlapping windows and the expected 6,188/1,092/1,092
train/validation/test counts. Its sample-ID digest is
`91cb51e83cb5fa130d6bd9095b93047d9b383f01b213eb0ed334437adee742be`.

ECG, HRV, and label arrays are byte-identical to the local cache. The EEG array
has the same shape, dtype, indices, and source data but a different byte digest
after filtering (`3d2769...` on Kaggle versus `e8c5cb...` locally), consistent
with the recorded NumPy/SciPy environment difference. The remote cache is treated
as its own versioned preprocessing artifact; its manifest is used by both deep
models, so their comparison remains internally identical. No local array was
silently substituted.

## Reproducible notebooks

| Notebook | Purpose | Accelerator |
|---|---|---|
| `amirmz/stressclassification-dreamer-corrected-cache` | Raw hash check and corrected cache | CPU |
| `amirmz/stressclassification-base-cnn` | Base CNN training and test evaluation | Tesla T4 |
| `amirmz/stressclassification-paper-inspired-attention` | Attention adaptation training and test evaluation | Tesla T4 |

The notebooks run without internet, embed a deterministic ZIP of tracked source
and configuration, attach the private output of the cache notebook, save a
last-epoch resume checkpoint after every epoch, and terminate when the batch
notebook exits. Kaggle's managed accelerator allocation incurred no direct
project charge. After both completed runs, the authenticated quota endpoint
reported 0.10 of 30.00 GPU-hours used and 29.90 hours remaining, with refresh at
2026-10-17T00:00:00. Runs were nevertheless bounded by 100 epochs and validation
patience 12. Kaggle reports quota consumption rather than a monetary invoice;
the direct monetary cost for these managed runs was USD 0.

## Recorded execution

The preprocessing run took 85.92 seconds. Base CNN used a Tesla T4 with PyTorch
2.11.0+cu128, CUDA 12.8, mixed precision, and stopped after 37 epochs; its
measured training/evaluation runtime was 113.39 seconds. Attention used the same
environment, stopped after 24 epochs, and measured 136.58 seconds. Per-run
environment, GPU, runtime, embedded-source/data hashes, and resume state are
stored with each result. The quota figure includes Kaggle allocation overhead
and is therefore not interchangeable with the in-process model runtime.

## Probability-metric compatibility correction

Base inference correctly saved class predictions, but float16 softmax rows could
differ from a unit sum by up to 0.00047. scikit-learn 1.6.1 therefore rejected
multiclass AUROC. Commit `60b84e8` normalizes finite non-negative score rows in
float64 before metrics and serialization. Base metrics were deterministically
regenerated from the saved test probabilities—without retraining or changing
any prediction—and the raw Kaggle files and their SHA-256 hashes were retained.
Attention was submitted with the normalization correction already embedded.
Final postprocessing treats the persisted CSV as canonical so rank metrics
replay exactly even for tied scores; `scripts/results/verify_experiment_artifacts.py`
confirms consistency for all seven models.
