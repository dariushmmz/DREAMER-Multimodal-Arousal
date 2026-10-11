# Modal execution environment and cost controls

## Verified local state

- Modal client: 1.5.2.
- Authentication is active for the configured workspace; no credential value
  is recorded in the repository.
- Current-month billing through 2026-10-12 showed USD 0.0016348 of CPU and
  memory usage and no GPU usage. This does **not** prove that free credits remain.
- Existing volumes and one idle deployed application were observed; no task was
  running. They were not modified.

## Planned environment

`modal/train_app.py` defines a reproducible Python 3.11 image with pinned NumPy,
SciPy, pandas, scikit-learn, Matplotlib, Seaborn, PyYAML, and PyTorch versions.
It requests one NVIDIA L4, two physical CPU cores, 8 GiB RAM, one container maximum,
zero retries, and a 90-minute hard timeout. The input volume is mounted read-only
and the result volume is committed after training. There is no deployed or
scheduled service, so the job terminates after the call.

## Estimate and approval gate

The official Modal price page listed the L4 at USD 0.000222/s, CPU at USD
0.0000131/core/s, and memory at USD 0.00000222/GiB/s on the audit date.
Based on 6,188 training windows and model sizes of 215,843 and 1,846,085
parameters, the expected duration is conservatively 20–45 minutes per capped
100-epoch run. The requested resources total USD 0.00026596/s, so estimated
compute is approximately USD 0.32–0.72 per model; the 90-minute timeout bounds
the worst case near USD 1.44 per model (USD 2.88 for both). Actual charges depend on startup, epoch
time, early stopping, and account credits.

No paid GPU execution or persistent-volume upload is authorized until the user
approves this cost envelope.
