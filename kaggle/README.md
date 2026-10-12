# Kaggle execution workflow

The active remote workflow uses three private Kaggle notebooks generated from
the canonical repository source:

1. `amirmz/stressclassification-dreamer-corrected-cache` verifies the attached
   `phhasian0710/dreamer` raw file and creates the corrected non-overlapping
   cache on CPU.
2. `amirmz/stressclassification-base-cnn` trains and evaluates Base CNN on GPU.
3. `amirmz/stressclassification-paper-inspired-attention` independently trains
   and evaluates `paper_inspired_attention` on GPU.

Both GPU notebooks consume the versioned output of the cache notebook, attach
the raw Kaggle dataset for an independent SHA-256 check, and write checkpoints,
history, predictions, metrics, figures, and environment metadata to
`/kaggle/working/results/<model>/`. A `last_checkpoint.pt` is written after
every epoch. To resume an interrupted version, add its prior output as a kernel
source; the entrypoint discovers the matching checkpoint automatically.

Regenerate notebooks after any canonical source change:

```powershell
python kaggle/build_kernels.py
```

The generated notebooks embed only source and configuration. Raw DREAMER data
is never uploaded from the local project.

After a completed run, download its output to a temporary directory and ingest
it with validation and normalized probability metrics:

```powershell
python scripts/results/ingest_kaggle_outputs.py `
  --source .temp/kaggle_attention_output/results/paper_inspired_attention `
  --kernel-log .temp/kaggle_attention_output/stressclassification-paper-inspired-attention.log `
  --model paper_inspired_attention
python scripts/results/build_model_comparison.py
python scripts/results/verify_experiment_artifacts.py
```

The local proxy and Kaggle OAuth profile are runtime-only account settings and
must not be embedded in notebooks, logs, or repository files.
