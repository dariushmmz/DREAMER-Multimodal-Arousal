# Modal execution environment and cost controls

## Verified local state

- Modal client: 1.5.2.
- Authentication is active for the configured workspace; no credential value
  is recorded in the repository.
- Current-month billing through 2026-10-11 showed only negligible CPU and
  memory usage and no GPU usage. This does **not** prove that free credits remain.
- Existing volumes and one idle deployed application were observed; no task was
  running. They were not modified.

## Planned environment

`modal/train_app.py` defines a reproducible Python 3.11 image with pinned NumPy,
SciPy, pandas, scikit-learn, Matplotlib, Seaborn, PyYAML, and PyTorch versions.
It requests one NVIDIA L4, four CPU cores, 16 GiB RAM, one container maximum,
zero retries, and a two-hour hard timeout. The input volume is mounted read-only
and the result volume is committed after training. There is no deployed or
scheduled service, so the job terminates after the call.

## Estimate and approval gate

The official Modal price page listed the L4 at USD 0.000222/s on the audit date.
Based on 6,188 training windows and model sizes of 215,843 and 1,846,085
parameters, the expected duration is conservatively 20–45 minutes per capped
100-epoch run. Estimated total compute is approximately USD 0.27–0.64 per
model, including requested CPU and memory; the two-hour timeout bounds the
worst case near USD 1.70 per model. Actual charges depend on startup, epoch
time, early stopping, and account credits.

No paid GPU execution or persistent-volume upload is authorized until the user
approves this cost envelope.

