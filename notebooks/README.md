# Notebooks

The four notebooks moved from the repository root are preserved with their
original outputs. They document the legacy workflow and must not be used as
evidence for the corrected protocol because they use overlapping windows and
random window-level splits.

Corrected notebooks call reusable code from `src/` and use the fixed,
subject-disjoint split in `configs/`.

- `01_dataset_and_leakage_audit.ipynb`: target, window, and split checks.
- `02_classical_evaluation.ipynb`: corrected tables and figures.
- `03_deep_models.ipynb`: architectures and the explicit GPU execution gate.
