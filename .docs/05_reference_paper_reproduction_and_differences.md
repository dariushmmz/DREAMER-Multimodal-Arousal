# Reference papers, reproducibility, and differences

## File identity check

The user-specified PDF path is not the paper title stated in the request.

1. `.paper/ECG and EEG based detection and multilevel.pdf` is the 2023 PLOS ONE
   paper [*ECG and EEG based detection and multilevel classification of stress
   using machine learning for specified genders: A preliminary study*](https://doi.org/10.1371/journal.pone.0291070).
2. `.paper/An attention-based multimodal.pdf` is the 2026 Scientific Reports
   paper [*An attention-based multimodal deep learning framework integrating
   EEG and ECG for enhanced stress detection*](https://doi.org/10.1038/s41598-026-44499-0).

Both were read and are treated separately.

## 2023 PLOS ONE method

The study recorded 40 university students in no-stress eyes-open (EO),
low-stress mental arithmetic (AC1), and high-stress arithmetic with audio
distraction (AC2) conditions. It used 8-channel EEG at 2,000 Hz, ECG at 1,000
Hz, 136 EEG features, 11 ECG/HRV features, ANOVA ranking, seven classical
classifiers, repeated stratified 10-fold validation, and logistic-regression
stacking. Mixed-gender three-level accuracy was 61.00% for the best individual
model and 64.08% for stacking.

The current repository lacks those raw EO/AC1/AC2 recordings and uses DREAMER
instead. Therefore a faithful result reproduction is not claimed. The adapted
stacking run preserves key algorithmic elements while correcting subject
leakage and reports its adaptation status in
`results/corrected/dreamer_arousal_3class/paper_2023_adapted_stacking/protocol.json`.

## 2026 Scientific Reports method and internal incompatibility

The PDF reports 10-second non-overlapping epochs; EEG scalograms processed in
parallel by modified VGG16, EfficientNetB0, and ResNeXt50; a raw-ECG CNN plus 25
HRV features; hierarchical attention; a 512→256→128→3 head; Adam/cosine
training; subject-independent 80/10/10 division; 42.7 M parameters; and 95.7%
accuracy for rest/stress/amusement.

However, the stated experiment cannot be reconstructed from its stated data:

- The official [WESAD description](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)
  lists BVP, ECG, EDA, EMG, respiration, temperature, and acceleration for 15
  subjects—no EEG.
- The primary [CASE data descriptor](https://pmc.ncbi.nlm.nih.gov/articles/PMC6785543/)
  lists ECG, BVP, three EMG channels, GSR, respiration, and skin temperature
  for 30 participants—no EEG. Local CASE README files confirm the same.
- The 2026 paper instead describes WESAD as 14-channel EEG+ECG and CASE as
  20-subject 32-channel EEG. It totals 35 subjects, whereas the primary datasets
  total 45, and assigns CASE mental-math stress states not present in its data
  descriptor.
- Its 700 Hz EEG/ECG inputs and 28/4/3 subject split follow those incompatible
  descriptions. The PDF provides no executable code that resolves the mismatch.

Consequently, reporting a “faithful 2026 reproduction” would be scientifically
false. The implemented `paper_inspired_attention` model is a transparent
DREAMER adaptation and is not compared numerically with 95.7% as if datasets
and tasks were equivalent.

## Main differences

| Dimension | 2026 paper claim | Corrected project |
|---|---|---|
| Dataset | WESAD + CASE, 35 subjects | DREAMER, 23 subjects |
| Ground truth | Rest/stress/amusement protocol states | Self-reported low/medium/high arousal |
| Modalities | Claimed EEG + ECG + HRV | Actual DREAMER 14-channel EEG + ECG-derived HRV |
| Epochs | 10 s, no overlap | 10 s, no overlap |
| Split | Claimed subject-independent 80/10/10 | Fixed subject-independent 17/3/3 |
| EEG representation | CWT 128×128×3, three ImageNet backbones | Raw temporal multi-scale Conv1D adaptation |
| ECG representation | Raw CNN then three 2D backbones + 25 HRV | Raw Conv1D + 25 HRV descriptors |
| Parameters | 42.7 M reported | 1.846 M attention adaptation |
| Claim | Stress-state detection | Arousal classification only |

