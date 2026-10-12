# Checkpoints, fitted models, and dataset inventory

## Raw DREAMER dataset (distributed separately)

- Expected repository location: `data/DREAMER.mat`
- Size: 452,847,973 bytes
- SHA-256: `35eed939e6f206a0938b13ba063178e2795ef1739b9cac8e5704a968393f3463`
- Verified Kaggle source: `phhasian0710/dreamer`
- The raw dataset is omitted from the client ZIP because of size and dataset distribution considerations.

## Derived deep cache (reproducible, distributed separately)

The cache is rebuilt with `python scripts/data/build_deep_cache.py` after placing the raw file above.

| Array | Shape | Size (bytes) | SHA-256 |
|---|---|---:|---|
| eeg | 8372×1280×14 | 300,052,608 | `e8c5cb455aa82c30871425e532909e99197ba452b0b26057f53c3ebf89e9baf9` |
| ecg | 8372×2560×1 | 42,864,768 | `32a925d6f563e95cf1a817a9d359352906b4e6a738a7045fe0327532cbfe2879` |
| hrv | 8372×25 | 837,328 | `3a535de0a382ab1cf2b48f308df4280d963f664883298adb9c0df804fb87ac32` |
| labels | 8372 | 67,104 | `9db5614e42c2417842e4b75acd5c5aa07cc8b4421771ccbd350fc344301a2c1f` |

## Delivered fitted-model files

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `artifacts/model_binaries/dreamer_arousal_3class/base_cnn/best_checkpoint.pt` | 882,671 | `bf1e862c389b26451624cc0e425b7a1805151489d0d2e5cd11010f4233b877ec` |
| `artifacts/model_binaries/dreamer_arousal_3class/base_cnn/last_checkpoint.pt` | 3,519,571 | `5070117d71f1854b2a4b77f11c35d50a120acba9610a94fb01564a5a493d6ce2` |
| `artifacts/model_binaries/dreamer_arousal_3class/classical/knn/model.joblib` | 6,889,006 | `a5a783d654624e3249f64b05b0d13e9d56ec173819b61bf5533845dbe79f3c72` |
| `artifacts/model_binaries/dreamer_arousal_3class/classical/logistic_regression/model.joblib` | 14,641 | `c7048e2b3dd858440558946e34d173ce64298736c93cbda09a83a8e077a3b714` |
| `artifacts/model_binaries/dreamer_arousal_3class/classical/random_forest/model.joblib` | 38,731,066 | `a78e38f5ee174e9cd29045550736c4827613d9ceedf769efc44e5123f11a12e8` |
| `artifacts/model_binaries/dreamer_arousal_3class/classical/svm_rbf/model.joblib` | 11,345,557 | `a94e204814e45605ff0679f51cb8f1422b3e6ca90129f38c5aec33cc34f25027` |
| `artifacts/model_binaries/dreamer_arousal_3class/paper_2023_adapted_stacking/stack.joblib` | 46,868,025 | `6276ba1ad63cd18c8945cbaaeb29cf3f7757be701ce767d6ba6709b928ac160a` |
| `artifacts/model_binaries/dreamer_arousal_3class/paper_inspired_attention/best_checkpoint.pt` | 7,432,771 | `e012474d85f294cf1282efe247f89d6d580f0e54d08986e9971aec1949e8a363` |
| `artifacts/model_binaries/dreamer_arousal_3class/paper_inspired_attention/last_checkpoint.pt` | 29,700,511 | `be35f3c7ecd2f0c6b48e300d211b829bc211d9b15a26be99242c2aa53fedf958` |

The `best_checkpoint.pt` files are the validation-selected inference checkpoints.
The `last_checkpoint.pt` files are included only for interruption-safe resume provenance.
Classical `.joblib` files are the fitted estimators retained with the corrected experiments.
