# Notebooks

## Corrected workflows

The notebooks in `workflows/` are lightweight, module-driven views of the final
protocol and validated results. They locate the repository root automatically
and do not start training when opened.

- `01_dataset_protocol_audit.ipynb`: target, non-overlap, split, and leakage checks.
- `02_classical_ml_evaluation.ipynb`: corrected classical table and figure.
- `03_deep_learning_evaluation.ipynb`: architectures and validated GPU results.

## Exploratory utility

`exploratory/dreamer_csv_overview.ipynb` demonstrates optional raw-data export
and bounded-memory inspection. CSV export is not required by the final pipeline.

## Historical notebooks

The four notebooks in `legacy/` preserve the original pre-correction work and
outputs. They use overlapping windows, inconsistent targets, and random
window-level splits, so their measurements are historical evidence only and
must not be mixed with `results/corrected/`. Legacy helper modules are isolated
under `src/legacy/`.
