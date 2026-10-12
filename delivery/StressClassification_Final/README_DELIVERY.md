# StressClassification final client delivery

This package contains the final experimental report in English Markdown and
Persian Markdown/B Nazanin RTL DOCX, Persian reviewer responses in Markdown and
B Nazanin RTL DOCX, the 2026 reference-paper comparison, reproducible source,
validated corrected results, and fitted model files.

## Start here

1. Read `reports/FINAL_EXPERIMENTAL_REPORT_FA.docx` for the Persian client report
   or `reports/FINAL_EXPERIMENTAL_REPORT.md` for the English Markdown report.
2. Reviewer responses are in `reports/REVIEWER_RESPONSES_FA.md` and `.docx`.
3. `reports/REFERENCE_PAPER_COMPARISON.md` explains why the 2026 result is not
   directly comparable and why the implemented attention model is an adaptation.
4. Reproduction commands and project layout are in `source/README.md`.
5. Every delivered file is hashed in `manifests/package_manifest.json`.

The raw DREAMER file and derived 344 MiB tensor cache are not redistributed in
this package. Their verified hashes, expected locations, and reconstruction
method are documented in `manifests/CHECKPOINTS_AND_DATA.md`. Fitted PyTorch
and classical model binaries are included under `artifacts/model_binaries/`.
