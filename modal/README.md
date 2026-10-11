# Modal GPU execution

The Modal account is authenticated, but no GPU job is launched automatically.
The job is bounded to one NVIDIA L4, two physical CPU cores, 8 GiB RAM, a 90-minute hard
timeout, zero retries, and a ten-second scale-down window.

## Cost envelope

Modal's price page on 2026-10-12 lists an L4 at USD 0.000222/s,
CPU at USD 0.0000131/core/s, and memory at USD 0.00000222/GiB/s. The requested
resources therefore total USD 0.00026596/s (USD 0.957456/hour). The two corrected networks contain
approximately 0.216 M and 1.846 M parameters and train on 6,188 windows. A
conservative 20–45 minute estimate per 100-epoch capped run is roughly
USD 0.32–0.72 total per model. Early stopping may reduce this. The 90-minute
timeout creates a worst-case compute ceiling of approximately USD 1.44 per
model, or USD 2.88 for both models.

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
