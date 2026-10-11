# Modal GPU execution

The Modal account is authenticated, but no GPU job is launched automatically.
The job is bounded to one NVIDIA L4, four CPU cores, 16 GiB RAM, a two-hour hard
timeout, zero retries, and a ten-second scale-down window.

## Cost envelope

Modal's price page on 2026-10-11 lists an L4 at USD 0.000222/s
(USD 0.7992/hour), excluding CPU and memory. The two corrected networks contain
approximately 0.216 M and 1.846 M parameters and train on 6,188 windows. A
conservative 20–45 minute estimate per 100-epoch capped run is roughly
USD 0.27–0.64 total per model after adding four CPU cores and 16 GiB RAM. Early
stopping may reduce this. The two-hour timeout creates a worst-case compute
ceiling around USD 1.70 per model.

These are estimates, not a credit assertion. The current-month billing report
showed no GPU usage, but the CLI does not establish the remaining included
credit or payment-method state. Explicit approval is required before upload and
execution.

## Staged commands (run only after approval)

```powershell
modal volume create stress-classification-dreamer-v1
modal volume create stress-classification-results-v1
modal volume put stress-classification-dreamer-v1 data/processed/deep_cache dreamer_arousal_v1/deep_cache
modal volume put stress-classification-dreamer-v1 data/processed/window_index.csv dreamer_arousal_v1/window_index.csv
modal volume put stress-classification-dreamer-v1 configs/dreamer_arousal_3class.yaml dreamer_arousal_v1/config.yaml
modal run modal/train_app.py --model base_cnn
modal run modal/train_app.py --model paper_inspired_attention
modal volume get stress-classification-results-v1 dreamer_arousal_v1 results/modal_download
```

The result volume is explicitly committed at the end of a successful run. The
function has no schedule or deployment and scales to zero after the call.

