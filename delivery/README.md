# Client delivery

`StressClassification_Final/` is the canonical delivery tree. It contains the
final reports, organized reproducible source snapshot, processed metadata,
validated result artifacts, figures, and local fitted-model files.

Run the following after final verification to rebuild it deterministically:

```powershell
python scripts/reporting/build_delivery_package.py --zip
```

Large `*.pt` and `*.joblib` files inside the delivery tree and the generated ZIP
are intentionally excluded from Git. They remain present locally and are
covered by `StressClassification_Final/manifests/package_manifest.json` and
`delivery_checksums.json`. This avoids duplicating more than 100 MiB of fitted
models in Git history while keeping the handoff verifiable.
