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
- `modal/`: bounded Modal GPU execution entry points.
- `tests/`: leakage, splitting, preprocessing, and model tests.
- `data/`: local datasets and tracked metadata only; large signal files are
  intentionally ignored by Git.
- `results/`: immutable legacy outputs and versioned corrected experiment
  artifacts.
- `.paper/`: local reference PDFs.
- `.docs/`: technical reports and reviewer responses.

The pre-correction state is preserved by Git commit `2c7b2fd`. Legacy scores
are retained for traceability but are not valid leakage-free measurements.

