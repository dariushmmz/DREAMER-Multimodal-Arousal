# StressClassification

Leakage-audited multimodal classification of three levels of **self-reported
arousal** using DREAMER EEG, raw ECG, and ECG-derived HRV. The final evaluation
is subject-independent and uses non-overlapping 10-second windows.

> DREAMER does not provide clinical stress labels. Results in this repository
> must be described as low/medium/high arousal classification, not stress
> diagnosis. The attention network is a documented DREAMER adaptation of ideas
> from the 2026 reference paper, not a faithful reproduction of that paper.

## Implemented methods and final results

All methods use the same target mapping, sample IDs, and fixed test participants.

| Method | Accuracy | Balanced accuracy | Macro-F1 | Macro-AUROC |
|---|---:|---:|---:|---:|
| Adapted stacking | **0.4698** | **0.3930** | **0.3755** | 0.5097 |
| Paper-inspired attention | 0.4029 | 0.3732 | 0.3554 | **0.5117** |
| kNN | 0.3901 | 0.3160 | 0.3020 | 0.4578 |
| Random forest | 0.4560 | 0.3449 | 0.2933 | 0.4367 |
| Logistic regression | 0.3361 | 0.2884 | 0.2791 | 0.4236 |
| RBF SVM | 0.4524 | 0.3275 | 0.2533 | 0.3945 |
| Base CNN | 0.3297 | 0.3584 | 0.2394 | 0.4843 |

The adapted stack is retained as an independent classical comparator; it is not
a reproduction or baseline from the client-requested 2026 paper.
The attention model uses multi-scale temporal CNNs, raw ECG+HRV fusion, and
learned modality attention; it does not implement the 2026 paper's CWT and
VGG16/EfficientNetB0/ResNeXt50 image system.

## Repository structure

```text
configs/                         fixed experiment configuration
data/                            local raw data and tracked processed metadata
notebooks/
  workflows/                     corrected, module-driven notebooks
  exploratory/                   optional raw-data inspection
  legacy/                        preserved pre-correction notebooks
src/stress_classification/       reusable scientific pipeline
src/legacy/                      helpers only for archived notebooks
scripts/
  data/                          audit, cache construction, optional CSV export
  experiments/                   classical, stacking, ablation, and deep runs
  results/                       ingestion, comparison, and replay verification
  reporting/                     DOCX, manifest, and delivery builders
kaggle/                          reproducible private CPU/GPU notebook workflow
modal/                           preserved, unused post-migration implementation
results/
  corrected/                     validated subject-independent measurements
  legacy/                        isolated leakage-affected historical outputs
.docs/                           technical reports and reviewer responses
delivery/                        client-ready delivery snapshot and inventory
```

The package modules have narrow responsibilities: `data.py` constructs the
window index; `preprocessing.py` filters signals and extracts HRV; `features.py`
builds handcrafted features; `deep_data.py` serves cached tensors; `models.py`
defines both CNNs; `training.py` handles seeded training/checkpoints; and
`evaluation.py` produces replayable metrics and confusion matrices.

The corrected notebooks are:

- `01_dataset_protocol_audit.ipynb`: target, split, non-overlap, and leakage checks.
- `02_classical_ml_evaluation.ipynb`: corrected classical results.
- `03_deep_learning_evaluation.ipynb`: architectures and validated GPU outputs.

## Dataset and protocol

Place `DREAMER.mat` at `data/DREAMER.mat`. The verified Kaggle source is
`phhasian0710/dreamer`; its raw file SHA-256 is
`35eed939e6f206a0938b13ba063178e2795ef1739b9cac8e5704a968393f3463`.

- Target mapping: ratings 1–2 = low, 3 = medium, 4–5 = high.
- Windows: 10 seconds, zero overlap.
- Train/validation/test participants: 17/3/3.
- Train/validation/test windows: 6,188/1,092/1,092.
- Model selection: validation macro-F1 only.
- Test set: evaluated after checkpoint/hyperparameter selection.

The authoritative protocol is `configs/dreamer_arousal_3class.yaml`; tracked
sample and split provenance is under `data/processed/`.

## Environment

Python 3.10 or newer is supported. Install the package and relevant extras:

```powershell
python -m pip install -e ".[test,notebooks,reporting,kaggle]"
```

PyTorch GPU builds should match the execution platform. Exact Kaggle package,
CUDA, and GPU versions are stored with each deep result.

## Reproduce and validate locally

From the repository root:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -m pytest

python scripts/data/audit_dataset.py --output-dir data/processed
python scripts/data/build_deep_cache.py
python scripts/experiments/train_classical_models.py
python scripts/experiments/train_adapted_stacking.py
python scripts/experiments/evaluate_modality_ablation.py

python scripts/results/build_model_comparison.py
python scripts/results/verify_experiment_artifacts.py
python scripts/reporting/generate_reviewer_docx.py
python scripts/reporting/build_artifact_manifest.py
```

Classical runs can be reproduced locally. The completed deep experiments were
executed on private Kaggle Tesla T4 notebooks. Regenerate their self-contained
sources with `python kaggle/build_kernels.py`; submission, resumable checkpoint,
download, and ingestion details are in `kaggle/README.md`. Do not retrain merely
to regenerate reports.

## Results, checkpoints, and reports

- Unified metrics and figures:
  `results/corrected/dreamer_arousal_3class/`.
- Deep histories, predictions, logs, confusion matrices, and metadata:
  `base_cnn/` and `paper_inspired_attention/` below that directory.
- Best/last PyTorch checkpoints remain local because `*.pt` is intentionally
  excluded from Git; their sizes and SHA-256 hashes are in
  `.docs/artifact_manifest.json` and the delivery inventory.
- Final client report: `.docs/09_final_experimental_report.md`.
- Persian final report: `.docs/09_final_experimental_report_fa.md` and the
  B Nazanin RTL DOCX edition beside it.
- Persian reviewer responses: `.docs/07_responses_to_reviewers_fa.md` and DOCX.
- Reference-paper comparison:
  `.docs/05_reference_paper_reproduction_and_differences.md`.

## Scientific limitations

Only three participants are held out for testing, cross-participant variation
is substantial, low-arousal recall remains weak, and both CNNs overfit their
training participants. Historical random-window results are not comparable with
the corrected measurements. The reported 95.7% accuracy from the 2026 paper is
also not directly comparable because its stated datasets, labels, sensor setup,
architecture, and evaluation population differ materially from this project.
