# Multimodal EEG–ECG Arousal Classification

This repository contains a leakage-audited research pipeline for three-class,
subject-independent classification using the DREAMER EEG and ECG dataset.

> **Scientific scope:** DREAMER provides self-reported valence, arousal, and
> dominance labels; it does not provide ground-truth stress labels. The
> corrected experiments therefore use three-class **arousal** as the common
> target and do not present arousal classification as clinical stress
> detection.

## Repository layout

- `notebooks/`: original notebooks plus corrected, module-driven workflows.
- `src/`: reusable preprocessing, models, training, and evaluation code.
- `configs/`: versioned experiment configurations and fixed split definitions.
- `scripts/`: command-line entry points for dataset audit and experiments.
- `kaggle/`: generated private-notebook workflow used for GPU experiments.
- `modal/`: preserved, bounded Modal implementation (not used after migration).
- `tests/`: leakage, splitting, preprocessing, and model tests.
- `data/`: local datasets and tracked metadata only; large signal files are
  intentionally ignored by Git.
- `results/`: immutable legacy outputs and versioned corrected experiment
  artifacts.
- `.paper/`: local reference PDFs.
- `.docs/`: technical reports and reviewer responses.

The pre-correction state is preserved by Git commit `2c7b2fd`. Legacy scores
are retained for traceability but are not valid leakage-free measurements.

## Reproduce local checks and classical experiments

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
python -m pytest
python scripts/audit_dreamer.py --output-dir data/processed
python scripts/run_classical.py
python scripts/run_modality_ablation.py
python scripts/run_paper_adapted_stacking.py
python scripts/build_comparison.py
python scripts/verify_corrected_results.py
```

The corrected classical and deep-learning results and their limitations are
summarized in `.docs/04_performance_and_comparison.md`. The executed Kaggle
workflow and the reason for migrating away from Modal are documented in
`.docs/06_modal_environment.md` and `kaggle/README.md`.
